//! The scripted coach: a deterministic MCP client that drives the
//! solver-under-test's `mcp serve` surface for a coached arm.
//!
//! A coach script is a versioned JSON file of rules — "when this event
//! appears, steer/cancel/answer" — so a coached cell is replayable: the same
//! event stream always produces the same interventions (the offline bar for
//! the coached arm; a frontier-model coach is a live, opportunistic leg).
//!
//! The drive is strict lockstep JSON-RPC over the child's stdio: one request
//! out, read until its response. The solver's own closeout runs when the
//! coach closes stdin, exactly as with any disconnecting MCP client. The
//! synthesized scorecard reports what the drive observed (tool calls, exit
//! reason, interventions); pass/fail stays the grader's verdict.

use std::fs::File;
use std::io::{BufRead, Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::solver::{configure_process_tree, terminate_and_reap_process_tree, SolveSpec};

/// How long each `events` poll waits server-side before returning empty.
const EVENTS_WAIT_MS: u64 = 2_000;
const PROCESS_POLL_INTERVAL: Duration = Duration::from_millis(25);

struct DeadlineFileReader {
    file: File,
    deadline: Instant,
    process_done: Arc<AtomicBool>,
    buffer: Vec<u8>,
    consumed: usize,
}

impl DeadlineFileReader {
    fn new(file: File, deadline: Instant, process_done: Arc<AtomicBool>) -> Self {
        Self {
            file,
            deadline,
            process_done,
            buffer: Vec::new(),
            consumed: 0,
        }
    }
}

impl Read for DeadlineFileReader {
    fn read(&mut self, destination: &mut [u8]) -> std::io::Result<usize> {
        let available = self.fill_buf()?;
        let length = available.len().min(destination.len());
        destination[..length].copy_from_slice(&available[..length]);
        self.consume(length);
        Ok(length)
    }
}

impl BufRead for DeadlineFileReader {
    fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
        if self.consumed < self.buffer.len() {
            return Ok(&self.buffer[self.consumed..]);
        }
        self.buffer.clear();
        self.consumed = 0;
        loop {
            let mut chunk = [0_u8; 8 * 1024];
            match self.file.read(&mut chunk)? {
                0 if self.process_done.load(Ordering::Acquire)
                    || Instant::now() >= self.deadline =>
                {
                    return Ok(&[]);
                }
                0 => std::thread::sleep(PROCESS_POLL_INTERVAL),
                length => {
                    self.buffer.extend_from_slice(&chunk[..length]);
                    return Ok(&self.buffer);
                }
            }
        }
    }

    fn consume(&mut self, amount: usize) {
        self.consumed = (self.consumed + amount).min(self.buffer.len());
    }
}

/// One trigger: an event type, optionally narrowed by a substring of the
/// event's serialized form.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuleTrigger {
    /// The event `type` to match (e.g. `tool_stuck`, `warning`,
    /// `permission_ask`, `tool_finished`).
    pub event: String,
    /// Additional substring the serialized event must contain.
    #[serde(default)]
    pub detail_contains: Option<String>,
}

/// One coach rule. Exactly one of `steer` / `reply` / `cancel` is the action.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoachRule {
    pub on: RuleTrigger,
    /// Steer text injected at the next safe boundary.
    #[serde(default)]
    pub steer: Option<String>,
    /// Answer a `permission_ask` (`true` allows, `false` denies). Only
    /// meaningful on a `permission_ask` trigger.
    #[serde(default)]
    pub reply: Option<bool>,
    /// Cancel the running turn.
    #[serde(default)]
    pub cancel: bool,
    /// How many times this rule may fire over the whole drive.
    #[serde(default = "default_max_fires")]
    pub max_fires: u32,
}

fn default_max_fires() -> u32 {
    1
}

/// A versioned coach script.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoachScript {
    pub schema: u32,
    /// The client name reported in the MCP handshake (provenance: the
    /// solver's lesson candidates name this).
    #[serde(default = "default_client")]
    pub client: String,
    pub rules: Vec<CoachRule>,
}

fn default_client() -> String {
    "localbench-coach".to_string()
}

/// Load and validate a coach script, failing loud on anything unusable.
///
/// # Errors
/// A plain-language message naming the defect.
pub fn load_coach_script(path: &Path) -> Result<CoachScript, String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| format!("could not read coach script {}: {e}", path.display()))?;
    let script: CoachScript = serde_json::from_str(&raw)
        .map_err(|e| format!("coach script {} does not parse: {e}", path.display()))?;
    if script.schema != 1 {
        return Err(format!(
            "unsupported coach-script schema {} (expected 1)",
            script.schema
        ));
    }
    for (index, rule) in script.rules.iter().enumerate() {
        let actions = usize::from(rule.steer.is_some())
            + usize::from(rule.reply.is_some())
            + usize::from(rule.cancel);
        if actions != 1 {
            return Err(format!(
                "coach rule {index} must have exactly one action (steer, reply, or cancel)"
            ));
        }
        if rule.on.event.trim().is_empty() {
            return Err(format!("coach rule {index} names no event"));
        }
    }
    Ok(script)
}

/// One action the engine decided on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoachAction {
    Steer(String),
    Reply { ask_id: String, allow: bool },
    Cancel,
}

/// The deterministic rule engine: same event sequence in, same actions out.
pub struct CoachEngine {
    script: CoachScript,
    fires: Vec<u32>,
}

impl CoachEngine {
    #[must_use]
    pub fn new(script: CoachScript) -> Self {
        let fires = vec![0; script.rules.len()];
        Self { script, fires }
    }

    /// The actions the script takes on one event, in rule order.
    pub fn on_event(&mut self, event: &Value) -> Vec<CoachAction> {
        let event_type = event["type"].as_str().unwrap_or_default().to_string();
        let serialized = event.to_string();
        let mut actions = Vec::new();
        for (index, rule) in self.script.rules.iter().enumerate() {
            if self.fires[index] >= rule.max_fires {
                continue;
            }
            if rule.on.event != event_type {
                continue;
            }
            if let Some(needle) = &rule.on.detail_contains {
                if !serialized.contains(needle.as_str()) {
                    continue;
                }
            }
            let action = if let Some(text) = &rule.steer {
                CoachAction::Steer(text.clone())
            } else if let Some(allow) = rule.reply {
                let Some(ask_id) = event["ask_id"].as_str() else {
                    continue; // a reply rule only makes sense on an ask
                };
                CoachAction::Reply {
                    ask_id: ask_id.to_string(),
                    allow,
                }
            } else {
                CoachAction::Cancel
            };
            self.fires[index] += 1;
            actions.push(action);
        }
        actions
    }
}

/// What one coached drive observed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DriveObservation {
    pub tool_calls: u32,
    pub interventions: u32,
    pub exit_reason: String,
}

/// Drive one coached session over any JSON-RPC transport, in strict lockstep.
/// `reader`/`writer` are the server's stdout/stdin; the caller owns process
/// lifetime and deadlines around this call.
///
/// # Errors
/// A plain-language message on transport failure or a malformed handshake.
pub fn drive_over<R: BufRead, W: Write>(
    reader: &mut R,
    writer: &mut W,
    engine: &mut CoachEngine,
    client: &str,
    problem: &str,
    deadline: Instant,
) -> Result<DriveObservation, String> {
    let mut next_id = 0u64;
    let mut request = |writer: &mut W, method: &str, params: Value| -> Result<u64, String> {
        next_id += 1;
        let line = json!({ "jsonrpc": "2.0", "id": next_id, "method": method, "params": params });
        writeln!(writer, "{line}").map_err(|e| format!("coach write: {e}"))?;
        writer.flush().map_err(|e| format!("coach flush: {e}"))?;
        Ok(next_id)
    };
    let read_response = |reader: &mut R, id: u64| -> Result<Value, String> {
        loop {
            let mut line = String::new();
            let read = reader
                .read_line(&mut line)
                .map_err(|e| format!("coach read: {e}"))?;
            if read == 0 {
                return Err("server closed the stream mid-drive".to_string());
            }
            let Ok(message) = serde_json::from_str::<Value>(&line) else {
                continue;
            };
            if message["id"].as_u64() == Some(id) {
                return Ok(message);
            }
        }
    };
    let call = |reader: &mut R,
                writer: &mut W,
                request: &mut dyn FnMut(&mut W, &str, Value) -> Result<u64, String>,
                tool: &str,
                arguments: Value|
     -> Result<Value, String> {
        let id = request(
            writer,
            "tools/call",
            json!({ "name": tool, "arguments": arguments }),
        )?;
        read_response(reader, id)
    };

    // Handshake.
    let id = request(
        writer,
        "initialize",
        json!({
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": { "name": client, "version": env!("CARGO_PKG_VERSION") },
        }),
    )?;
    let init = read_response(reader, id)?;
    if init["result"]["serverInfo"]["name"].is_null() {
        return Err("initialize returned no serverInfo".to_string());
    }

    // Start the task.
    let started = call(
        reader,
        writer,
        &mut request,
        "prompt",
        json!({ "text": problem }),
    )?;
    if started["result"]["isError"].as_bool() == Some(true) {
        return Err(format!("prompt refused: {}", started["result"]));
    }

    // Poll → react → repeat, until the turn stops or the deadline passes.
    let mut cursor = 0u64;
    let mut tool_calls = 0u32;
    let mut interventions = 0u32;
    let mut exit_reason = "unknown".to_string();
    'drive: loop {
        if Instant::now() >= deadline {
            let _ = call(reader, writer, &mut request, "cancel", json!({}));
            exit_reason = "coach-deadline".to_string();
            break;
        }
        let page = call(
            reader,
            writer,
            &mut request,
            "events",
            json!({ "cursor": cursor, "wait_ms": EVENTS_WAIT_MS }),
        )?;
        let result = &page["result"]["structuredContent"];
        cursor = result["next_cursor"].as_u64().unwrap_or(cursor);
        let entries = result["events"].as_array().cloned().unwrap_or_default();
        let mut stopped = false;
        for entry in &entries {
            let event = &entry["event"];
            match event["type"].as_str().unwrap_or_default() {
                "tool_started" => tool_calls += 1,
                "stopped" => {
                    exit_reason = event["reason"].as_str().unwrap_or("unknown").to_string();
                    stopped = true;
                }
                _ => {}
            }
            for action in engine.on_event(event) {
                interventions += 1;
                match action {
                    CoachAction::Steer(text) => {
                        call(
                            reader,
                            writer,
                            &mut request,
                            "prompt",
                            json!({ "text": text, "disposition": "steer" }),
                        )?;
                    }
                    CoachAction::Reply { ask_id, allow } => {
                        call(
                            reader,
                            writer,
                            &mut request,
                            "reply_permission",
                            json!({ "ask_id": ask_id, "allow": allow }),
                        )?;
                    }
                    CoachAction::Cancel => {
                        call(reader, writer, &mut request, "cancel", json!({}))?;
                    }
                }
            }
        }
        if stopped {
            break 'drive;
        }
    }

    Ok(DriveObservation {
        tool_calls,
        interventions,
        exit_reason,
    })
}

/// The synthesized scorecard for a coached drive: process facts the drive
/// observed, everything else zeroed/false. The grader owns pass/fail; the
/// diff-quality layer is not computed on this path.
#[must_use]
pub fn coached_scorecard(spec: &SolveSpec, observation: &DriveObservation, wall_ms: u64) -> String {
    use localx_eval_core::{
        ProcessBlock, QualityBlock, ResultsBlock, Scorecard, SpeedBlock, SCORECARD_SCHEMA,
    };
    let card = Scorecard {
        schema: SCORECARD_SCHEMA,
        task: spec.task.clone(),
        arm: spec.arm.clone(),
        model: spec.model.clone(),
        results: ResultsBlock {
            passed: false,
            regression_safe: false,
            partial_credit: 0.0,
            tests_total: 0,
            tests_passed: 0,
        },
        quality: QualityBlock {
            diff_added: 0,
            diff_removed: 0,
            diff_files: 0,
            vs_gold_ratio: None,
            format_clean: false,
            lint_clean: false,
            typecheck_clean: false,
            complexity_delta: None,
            tests_added: false,
        },
        process: ProcessBlock {
            tool_calls: observation.tool_calls,
            redundant_calls: 0,
            reproduce_before_fix: false,
            test_before_done: false,
            retrieval_used: false,
            retrieval_count: 0,
            exit_reason: observation.exit_reason.clone(),
            recovered_after_failure: false,
            interventions: observation.interventions,
            discipline: None,
        },
        speed: SpeedBlock {
            wall_ms,
            input_tokens: 0,
            output_tokens: 0,
        },
        judge: None,
    };
    card.to_json().unwrap_or_default()
}

/// The argv for the coached serve child.
#[must_use]
pub fn serve_args(spec: &SolveSpec) -> Vec<String> {
    vec![
        "mcp".to_string(),
        "serve".to_string(),
        "--model".to_string(),
        spec.model.clone(),
        "--permission".to_string(),
        spec.permission.clone(),
    ]
}

/// Drive one coached cell: spawn the solver's MCP serve surface in the task
/// workspace, run the script against it, close stdin (the solver's own
/// closeout runs), and synthesize the cell's scorecard. A watchdog kills the
/// child at the wall-clock bound so a hung server can never wedge the matrix.
///
/// # Errors
/// A plain-language message; the matrix isolates it as a failed cell.
pub fn drive_coached(
    bin: &str,
    workspace: &Path,
    spec: &SolveSpec,
    script_path: &Path,
    timeout: Duration,
) -> Result<String, String> {
    let script = load_coach_script(script_path)?;
    let client = script.client.clone();
    let mut engine = CoachEngine::new(script);

    // stdout is a regular file with an independently reopened reader. An MCP
    // descendant may inherit the writer, but it cannot hold a pipe reader
    // hostage after the deadline. The tail reader polls only until the same
    // cell deadline used by the process-tree watchdog.
    let stdout_capture = tempfile::NamedTempFile::new()
        .map_err(|error| format!("could not create coached stdout capture: {error}"))?;
    let stdout_writer = stdout_capture
        .as_file()
        .try_clone()
        .map_err(|error| format!("could not clone coached stdout capture: {error}"))?;
    let stdout_reader = stdout_capture
        .reopen()
        .map_err(|error| format!("could not reopen coached stdout capture: {error}"))?;

    let mut command = std::process::Command::new(bin);
    command
        .args(serve_args(spec))
        .current_dir(workspace)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::from(stdout_writer))
        // This stream was previously drained and discarded on a detached
        // thread. Discard it directly so no reader helper can outlive a cell.
        .stderr(std::process::Stdio::null());
    configure_process_tree(&mut command);
    let started = Instant::now();
    let mut child = command
        .spawn()
        .map_err(|e| format!("could not start {bin} mcp serve: {e}"))?;

    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| "no stdin pipe".to_string())?;
    // The watchdog owns the exact tree termination and direct-child reap; the
    // drive itself only reads/writes.
    let child = Arc::new(Mutex::new(child));
    let watchdog_child = Arc::clone(&child);
    let deadline = started + timeout;
    let process_done = Arc::new(AtomicBool::new(false));
    let watchdog_done = Arc::clone(&process_done);
    let watchdog_bin = bin.to_string();
    let watchdog = std::thread::spawn(move || -> (bool, Option<String>) {
        loop {
            if Instant::now() >= deadline {
                let diagnostic = match watchdog_child.lock() {
                    Ok(mut child) => terminate_and_reap_process_tree(&mut child, &watchdog_bin),
                    Err(_) => Some(
                        "coached process-tree watchdog could not lock the child at timeout"
                            .to_string(),
                    ),
                };
                watchdog_done.store(true, Ordering::Release);
                return (true, diagnostic);
            }
            match watchdog_child.lock().map(|mut child| child.try_wait()) {
                Ok(Ok(Some(_))) => {
                    watchdog_done.store(true, Ordering::Release);
                    return (false, None);
                }
                Ok(Err(error)) => {
                    watchdog_done.store(true, Ordering::Release);
                    return (
                        false,
                        Some(format!("coached process watchdog wait failed: {error}")),
                    );
                }
                Err(_) => {
                    watchdog_done.store(true, Ordering::Release);
                    return (
                        false,
                        Some("coached process watchdog lock was poisoned".to_string()),
                    );
                }
                Ok(Ok(None)) => {}
            }
            std::thread::sleep(PROCESS_POLL_INTERVAL);
        }
    });

    let mut reader = DeadlineFileReader::new(stdout_reader, deadline, process_done);
    // Leave the coach a margin to close out inside the cell bound.
    let drive_deadline = deadline
        .checked_sub(Duration::from_secs(5))
        .unwrap_or(deadline);
    let outcome = drive_over(
        &mut reader,
        &mut stdin,
        &mut engine,
        &client,
        &spec.problem,
        drive_deadline,
    );
    // Closing stdin is the MCP stdio shutdown: the solver runs its closeout
    // (lesson candidates from the coach's corrections land review-gated).
    drop(stdin);
    let (timed_out, cleanup_diagnostic) = watchdog.join().unwrap_or_else(|_| {
        (
            true,
            Some("coached process-tree watchdog panicked before cleanup completed".to_string()),
        )
    });

    if timed_out {
        let drive = outcome
            .as_ref()
            .err()
            .map_or_else(String::new, |reason| format!("; drive: {reason}"));
        let cleanup = cleanup_diagnostic
            .as_deref()
            .map_or_else(String::new, |diagnostic| format!("; cleanup: {diagnostic}"));
        return Err(format!(
            "coached drive timed out after {}s (arm '{}', task '{}'){drive}{cleanup}",
            timeout.as_secs(),
            spec.arm,
            spec.task
        ));
    }
    if outcome.is_ok() {
        if let Some(diagnostic) = &cleanup_diagnostic {
            return Err(format!(
                "coached drive completed but process completion is uncertain (arm '{}', task '{}'): {diagnostic}",
                spec.arm, spec.task
            ));
        }
    }

    let observation = outcome.map_err(|reason| {
        let cleanup = cleanup_diagnostic
            .as_deref()
            .map_or_else(String::new, |diagnostic| format!("; {diagnostic}"));
        format!(
            "coached drive failed (arm '{}', task '{}'): {reason}{cleanup}",
            spec.arm, spec.task,
        )
    })?;
    #[allow(clippy::cast_possible_truncation)]
    let wall_ms = started.elapsed().as_millis() as u64;
    Ok(coached_scorecard(spec, &observation, wall_ms))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn regular_file_tail_reader_delivers_live_lines_without_waiting_for_eof() {
        let capture = tempfile::NamedTempFile::new().unwrap();
        let mut writer = capture.as_file().try_clone().unwrap();
        let done = Arc::new(AtomicBool::new(false));
        let writer_done = Arc::clone(&done);
        let writer_thread = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            writeln!(writer, "live-response").unwrap();
            writer.flush().unwrap();
            writer_done.store(true, Ordering::Release);
        });
        let mut reader = DeadlineFileReader::new(
            capture.reopen().unwrap(),
            Instant::now() + Duration::from_secs(2),
            done,
        );
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        writer_thread.join().unwrap();
        assert_eq!(line, "live-response\n");
    }

    #[test]
    fn regular_file_tail_reader_has_a_hard_empty_stream_deadline() {
        let capture = tempfile::NamedTempFile::new().unwrap();
        let started = Instant::now();
        let mut reader = DeadlineFileReader::new(
            capture.reopen().unwrap(),
            started + Duration::from_millis(100),
            Arc::new(AtomicBool::new(false)),
        );
        let mut line = String::new();
        assert_eq!(reader.read_line(&mut line).unwrap(), 0);
        assert!(started.elapsed() < Duration::from_secs(1));
    }
    use std::io::Cursor;

    fn script(rules: Vec<CoachRule>) -> CoachScript {
        CoachScript {
            schema: 1,
            client: "localbench-coach".to_string(),
            rules,
        }
    }

    fn steer_rule(event: &str, text: &str, max_fires: u32) -> CoachRule {
        CoachRule {
            on: RuleTrigger {
                event: event.to_string(),
                detail_contains: None,
            },
            steer: Some(text.to_string()),
            reply: None,
            cancel: false,
            max_fires,
        }
    }

    #[test]
    fn the_engine_replays_identically_and_respects_max_fires() {
        let events = vec![
            json!({ "type": "tool_stuck", "name": "run_shell", "count": 3 }),
            json!({ "type": "warning", "message": "provider retry" }),
            json!({ "type": "tool_stuck", "name": "run_shell", "count": 4 }),
        ];
        let run = |events: &[Value]| {
            let mut engine = CoachEngine::new(script(vec![steer_rule(
                "tool_stuck",
                "try a different command",
                1,
            )]));
            events
                .iter()
                .flat_map(|event| engine.on_event(event))
                .collect::<Vec<_>>()
        };
        let first = run(&events);
        let second = run(&events);
        assert_eq!(first, second, "same events, same actions");
        assert_eq!(first.len(), 1, "max_fires bounds the rule");
    }

    #[test]
    fn a_reply_rule_answers_the_ask_it_saw() {
        let mut engine = CoachEngine::new(script(vec![CoachRule {
            on: RuleTrigger {
                event: "permission_ask".to_string(),
                detail_contains: Some("rm -rf".to_string()),
            },
            steer: None,
            reply: Some(false),
            cancel: false,
            max_fires: 5,
        }]));
        let actions = engine.on_event(&json!({
            "type": "permission_ask", "ask_id": "ask-7",
            "tool": "run_shell", "detail": "rm -rf build", "risk": "run a command",
        }));
        assert_eq!(
            actions,
            [CoachAction::Reply {
                ask_id: "ask-7".to_string(),
                allow: false
            }]
        );
        // A non-matching ask does not fire the narrowed rule.
        let actions = engine.on_event(&json!({
            "type": "permission_ask", "ask_id": "ask-8",
            "tool": "run_shell", "detail": "cargo test", "risk": "run a command",
        }));
        assert!(actions.is_empty());
    }

    #[test]
    fn a_script_with_zero_or_two_actions_per_rule_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("coach.json");
        std::fs::write(
            &path,
            r#"{ "schema": 1, "rules": [ { "on": { "event": "warning" } } ] }"#,
        )
        .unwrap();
        let err = load_coach_script(&path).unwrap_err();
        assert!(err.contains("exactly one action"), "{err}");

        std::fs::write(
            &path,
            r#"{ "schema": 1, "rules": [ { "on": { "event": "warning" }, "steer": "x", "cancel": true } ] }"#,
        )
        .unwrap();
        let err = load_coach_script(&path).unwrap_err();
        assert!(err.contains("exactly one action"), "{err}");
    }

    /// The lockstep drive is deterministic, so a whole scenario's server side
    /// can be precomputed: responses in exactly the order the coach asks.
    #[test]
    fn a_lockstep_drive_steers_counts_and_stops() {
        // Coach: init(1), prompt(2), events(3) -> [tool_started],
        // then the tool_stuck page(4) fires a steer(5), then events(6) -> stopped.
        let responses = [
            json!({ "jsonrpc": "2.0", "id": 1, "result": {
                "protocolVersion": "2025-06-18",
                "serverInfo": { "name": "localpilot", "version": "0" } } }),
            json!({ "jsonrpc": "2.0", "id": 2, "result": {
                "isError": false,
                "structuredContent": { "started": true } } }),
            json!({ "jsonrpc": "2.0", "id": 3, "result": { "structuredContent": {
                "events": [ { "seq": 1, "event": { "type": "tool_started", "id": "c1", "name": "run_shell" } } ],
                "next_cursor": 1, "dropped": 0, "busy": true } } }),
            json!({ "jsonrpc": "2.0", "id": 4, "result": { "structuredContent": {
                "events": [ { "seq": 2, "event": { "type": "tool_stuck", "name": "run_shell", "count": 3 } } ],
                "next_cursor": 2, "dropped": 0, "busy": true } } }),
            json!({ "jsonrpc": "2.0", "id": 5, "result": {
                "isError": false,
                "structuredContent": { "queued": "steer" } } }),
            json!({ "jsonrpc": "2.0", "id": 6, "result": { "structuredContent": {
                "events": [ { "seq": 3, "event": { "type": "stopped", "reason": "done" } } ],
                "next_cursor": 3, "dropped": 0, "busy": false } } }),
        ];
        let feed = responses.iter().fold(String::new(), |mut feed, value| {
            use std::fmt::Write as _;
            let _ = writeln!(feed, "{value}");
            feed
        });
        let mut reader = Cursor::new(feed.into_bytes());
        let mut sent: Vec<u8> = Vec::new();
        let mut engine = CoachEngine::new(script(vec![steer_rule(
            "tool_stuck",
            "try a different command",
            1,
        )]));

        let observation = drive_over(
            &mut reader,
            &mut sent,
            &mut engine,
            "localbench-coach",
            "fix the failing test",
            Instant::now() + Duration::from_secs(30),
        )
        .unwrap();

        assert_eq!(observation.tool_calls, 1);
        assert_eq!(observation.interventions, 1);
        assert_eq!(observation.exit_reason, "done");

        let sent = String::from_utf8(sent).unwrap();
        assert!(sent.contains("\"initialize\""));
        assert!(sent.contains("fix the failing test"));
        assert!(sent.contains("try a different command"));
        assert!(sent.contains("\"disposition\":\"steer\""));
    }

    #[test]
    fn the_coached_scorecard_parses_as_a_capability_card() {
        let spec = SolveSpec::new("apex", "coached", "t-1", "fix it");
        let observation = DriveObservation {
            tool_calls: 4,
            interventions: 2,
            exit_reason: "done".to_string(),
        };
        let json = coached_scorecard(&spec, &observation, 1234);
        let card = localbench_measure::runner::parse_capability_scorecard(&json).unwrap();
        assert_eq!(card.tool_calls, 4);
        assert_eq!(card.interventions, 2);
        assert_eq!(card.exit_reason, "done");
        assert!(!card.passed, "pass/fail belongs to the grader");
    }

    #[test]
    fn the_serve_argv_shape_is_pinned() {
        let spec = SolveSpec::new("apex", "coached", "t-1", "fix it");
        assert_eq!(
            serve_args(&spec),
            ["mcp", "serve", "--model", "apex", "--permission", "bypass"]
        );
    }

    /// Live leg (opt-in): drives a real solver child end-to-end with no model —
    /// the provider endpoint is dead, so the turn emits warnings and the
    /// script's cancel rule fires. Proves spawn → handshake → prompt → events →
    /// intervention → shutdown → synthesized scorecard against the real binary.
    /// Requires `LOCALBENCH_LIVE_LOCALPILOT` to name the solver binary.
    #[test]
    #[ignore = "live leg: needs a localpilot binary (set LOCALBENCH_LIVE_LOCALPILOT)"]
    fn live_coached_drive_cancels_a_dead_provider_run() {
        let Ok(bin) = std::env::var("LOCALBENCH_LIVE_LOCALPILOT") else {
            panic!("set LOCALBENCH_LIVE_LOCALPILOT to the solver binary path");
        };
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join(".localpilot.toml"),
            "[provider]\ndefault = \"local\"\n\n[providers.local]\nkind = \"anthropic\"\n\
             base_url = \"http://127.0.0.1:19998/v1\"\napi_key_env = \"LOCALBENCH_LIVE_TOKEN\"\n\
             model = \"live-smoke\"\n",
        )
        .unwrap();
        std::env::set_var("LOCALBENCH_LIVE_TOKEN", "smoke");
        let script_path = dir.path().join("coach.json");
        std::fs::write(
            &script_path,
            r#"{ "schema": 1, "client": "localbench-coach", "rules": [
                { "on": { "event": "warning", "detail_contains": "provider" },
                  "cancel": true, "max_fires": 1 } ] }"#,
        )
        .unwrap();

        let spec = SolveSpec::new("live-smoke", "coached", "t-live", "say hello");
        let json = drive_coached(
            &bin,
            dir.path(),
            &spec,
            &script_path,
            Duration::from_secs(60),
        )
        .unwrap();
        let card = localbench_measure::runner::parse_capability_scorecard(&json).unwrap();
        assert_eq!(card.interventions, 1, "the cancel rule fired once: {json}");
        assert!(
            card.exit_reason == "cancelled" || card.exit_reason == "provider_error",
            "the turn ended by cancel or provider failure: {}",
            card.exit_reason
        );
        assert!(!card.passed, "pass/fail belongs to the grader");
    }
}
