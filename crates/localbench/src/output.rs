//! Machine-output discipline: structured output goes to **stdout clean**,
//! human logs go to **stderr**, a non-TTY stdout defaults to JSON, and long
//! runs stream JSONL events (`started` → `result` per persisted cell →
//! `completed`, or `error` on an abort — the stream always terminates) — so
//! `localbench ... | jq` and a supervising harness never fight log noise for
//! position 0 of the stream.

use std::io::Write;

use serde::{Deserialize, Serialize};

/// How results are rendered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    Text,
    Json,
}

/// Resolve the output format: an explicit request wins; otherwise a non-TTY
/// stdout (a pipe, a harness) gets JSON and an interactive terminal gets text.
#[must_use]
pub fn resolve_format(explicit: Option<OutputFormat>, stdout_is_tty: bool) -> OutputFormat {
    explicit.unwrap_or(if stdout_is_tty {
        OutputFormat::Text
    } else {
        OutputFormat::Json
    })
}

/// One event on the long-run JSONL stream.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "lowercase")]
pub enum RunEvent {
    /// The run began.
    Started {
        run: String,
        /// Total planned units of work, when known.
        #[serde(skip_serializing_if = "Option::is_none")]
        total: Option<u64>,
    },
    /// One unit of work finished; `payload` is the unit's structured result.
    Result {
        run: String,
        payload: serde_json::Value,
    },
    /// The run finished; `payload` is the final structured result.
    Completed {
        run: String,
        payload: serde_json::Value,
    },
    /// The run failed; the message is the operator-facing reason.
    Error { run: String, message: String },
}

/// A writer pair enforcing the stream discipline: events/results to `out`
/// (stdout in production), logs to `err` (stderr). Nothing here ever writes a
/// log line to `out`.
pub struct MachineOutput<O: Write, E: Write> {
    out: O,
    err: E,
}

impl<O: Write, E: Write> MachineOutput<O, E> {
    /// A machine-output writer over the given sinks.
    pub fn new(out: O, err: E) -> Self {
        Self { out, err }
    }

    /// Emit one JSONL event to the structured stream.
    ///
    /// # Errors
    /// Returns the underlying I/O error.
    pub fn event(&mut self, event: &RunEvent) -> std::io::Result<()> {
        let line = serde_json::to_string(event)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        writeln!(self.out, "{line}")?;
        self.out.flush()
    }

    /// Emit a final structured result document to the structured stream.
    ///
    /// # Errors
    /// Returns the underlying I/O error.
    pub fn result(&mut self, value: &serde_json::Value) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(value)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        writeln!(self.out, "{json}")?;
        self.out.flush()
    }

    /// Emit a human log line — always to the log sink, never the stream.
    ///
    /// # Errors
    /// Returns the underlying I/O error.
    pub fn log(&mut self, message: &str) -> std::io::Result<()> {
        writeln!(self.err, "{message}")
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn non_tty_defaults_to_json_and_an_explicit_choice_wins() {
        assert_eq!(resolve_format(None, false), OutputFormat::Json);
        assert_eq!(resolve_format(None, true), OutputFormat::Text);
        assert_eq!(
            resolve_format(Some(OutputFormat::Text), false),
            OutputFormat::Text
        );
        assert_eq!(
            resolve_format(Some(OutputFormat::Json), true),
            OutputFormat::Json
        );
    }

    #[test]
    fn no_log_line_contaminates_position_zero_of_the_stream() {
        let mut out: Vec<u8> = Vec::new();
        let mut err: Vec<u8> = Vec::new();
        {
            let mut output = MachineOutput::new(&mut out, &mut err);
            output.log("starting the sweep (human noise)").unwrap();
            output
                .event(&RunEvent::Started {
                    run: "sweep-1".to_string(),
                    total: Some(2),
                })
                .unwrap();
            output.log("cell 1 measured").unwrap();
            output
                .event(&RunEvent::Result {
                    run: "sweep-1".to_string(),
                    payload: serde_json::json!({ "cell": 1, "solved": true }),
                })
                .unwrap();
            output
                .event(&RunEvent::Completed {
                    run: "sweep-1".to_string(),
                    payload: serde_json::json!({ "solved": 1, "total": 2 }),
                })
                .unwrap();
        }
        // Every stdout line parses as JSON from position 0 — the golden rule.
        let stdout = String::from_utf8(out).unwrap();
        for (i, line) in stdout.lines().enumerate() {
            let value: serde_json::Value = serde_json::from_str(line)
                .unwrap_or_else(|e| panic!("stdout line {i} is not clean JSON: {e}: {line}"));
            assert!(value.get("event").is_some());
        }
        // The human noise all landed on stderr.
        let stderr = String::from_utf8(err).unwrap();
        assert!(stderr.contains("human noise"));
        assert!(!stdout.contains("human noise"));
    }

    #[test]
    fn the_event_stream_carries_the_four_kinds() {
        let events = [
            RunEvent::Started {
                run: "r".to_string(),
                total: None,
            },
            RunEvent::Result {
                run: "r".to_string(),
                payload: serde_json::json!({}),
            },
            RunEvent::Completed {
                run: "r".to_string(),
                payload: serde_json::json!({}),
            },
            RunEvent::Error {
                run: "r".to_string(),
                message: "docker wedged".to_string(),
            },
        ];
        let kinds: Vec<String> = events
            .iter()
            .map(|e| {
                serde_json::to_value(e).unwrap()["event"]
                    .as_str()
                    .unwrap()
                    .to_string()
            })
            .collect();
        assert_eq!(kinds, ["started", "result", "completed", "error"]);
        // Round-trip: a consumer can parse the stream back.
        let line = serde_json::to_string(&events[3]).unwrap();
        let back: RunEvent = serde_json::from_str(&line).unwrap();
        assert_eq!(back, events[3]);
    }
}
