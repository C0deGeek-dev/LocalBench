//! Crash-tolerant, bounded evidence for tuner attempts.

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use localbench_scoring::score::{Overrides, Trial};
use serde::{Deserialize, Serialize};

use crate::trial::{sanitize_excerpt, sanitize_launch_args};

/// Default number of completed run manifests retained beside active runs.
pub const DEFAULT_RETAINED_RUNS: usize = 20;

/// One durable line in a tuner run manifest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunManifestRecord {
    pub run_id: String,
    pub attempt_id: String,
    pub ordinal: usize,
    pub phase: String,
    pub signature: String,
    pub cache_source: String,
    pub requested_overrides: Overrides,
    pub startup_ok: bool,
    pub oom: bool,
    pub measurement_usable: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure_stage: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure_reason: Option<String>,
    /// The typed failure's own detail. Stage and reason classify a failure;
    /// this is the sentence that says what actually happened — for a content
    /// failure, the gate message plus a bounded excerpt of the rejected reply.
    /// Without it a run manifest cannot distinguish a flood from an empty or
    /// merely terse answer, because `diagnostic_excerpt` carries server log
    /// text that is healthy in exactly that case.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure_detail: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub process_status: Option<i32>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub log_path: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub diagnostic_excerpt: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub launch_args: Vec<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub observed_configuration: BTreeMap<String, serde_json::Value>,
    /// Log-derived adjustments are advisory, never claimed as effective.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub advisory_observations: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pp_tps: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tg_tps: Option<f64>,
    pub recorded_at: String,
}

/// Append-only manifest writer. A sibling `.active` marker makes retention
/// fail-safe: interrupted runs remain active and are never pruned.
pub struct RunLedger {
    run_id: String,
    path: PathBuf,
    active_marker: PathBuf,
    log_dir: PathBuf,
    retained_runs: usize,
    next_ordinal: usize,
}

impl RunLedger {
    /// Create a run ledger and apply completed-run retention first.
    pub fn create(log_dir: &Path, run_id: &str, retained_runs: usize) -> std::io::Result<Self> {
        std::fs::create_dir_all(log_dir)?;
        let retained_runs = retained_runs.max(1);
        prune_completed(log_dir, retained_runs)?;
        let safe_id = safe_component(run_id);
        let path = log_dir.join(format!("run-{safe_id}.jsonl"));
        let active_marker = log_dir.join(format!("run-{safe_id}.active"));
        std::fs::write(&active_marker, run_id.as_bytes())?;
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        Ok(Self {
            run_id: run_id.to_string(),
            path,
            active_marker,
            log_dir: log_dir.to_path_buf(),
            retained_runs,
            next_ordinal: 0,
        })
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append and sync one complete JSON line. Earlier lines remain usable if
    /// the process is killed during the final append.
    pub fn record(
        &mut self,
        phase: &str,
        signature: &str,
        cache_source: &str,
        overrides: &Overrides,
        trial: &mut Trial,
        recorded_at: &str,
    ) -> std::io::Result<()> {
        self.next_ordinal += 1;
        let ordinal = self.next_ordinal;
        let measurement_usable = trial.is_measurement_usable();
        let fallback_attempt_id = format!("{}-{ordinal:04}", safe_component(&self.run_id));
        let (attempt_id, log_path) = {
            let diagnostic = trial.diagnostic.get_or_insert_with(Default::default);
            if diagnostic.attempt_id.is_empty() {
                diagnostic.attempt_id = fallback_attempt_id;
            }
            diagnostic.manifest_path = self.path.display().to_string();
            (diagnostic.attempt_id.clone(), diagnostic.log_path.clone())
        };

        let failure_stage = trial.failure.as_ref().map(|failure| {
            serde_json::to_value(failure.stage)
                .ok()
                .and_then(|value| value.as_str().map(str::to_string))
                .unwrap_or_else(|| "unknown".to_string())
        });
        let failure_reason = trial.failure.as_ref().map(|failure| {
            serde_json::to_value(failure.reason)
                .ok()
                .and_then(|value| value.as_str().map(str::to_string))
                .unwrap_or_else(|| "unknown".to_string())
        });
        let failure_detail = trial
            .failure
            .as_ref()
            .map(|failure| sanitize_excerpt(&failure.detail))
            .filter(|detail| !detail.is_empty());
        let record = RunManifestRecord {
            run_id: self.run_id.clone(),
            attempt_id,
            ordinal,
            phase: phase.to_string(),
            signature: signature.to_string(),
            cache_source: cache_source.to_string(),
            requested_overrides: overrides.clone(),
            startup_ok: trial.startup_ok,
            oom: trial.oom,
            measurement_usable,
            failure_stage,
            failure_reason,
            failure_detail,
            process_status: trial.process_status,
            log_path,
            diagnostic_excerpt: sanitize_excerpt(&trial.diagnostic_excerpt),
            launch_args: sanitize_launch_args(&trial.launch_args),
            observed_configuration: sanitize_observed(&trial.observed_configuration),
            advisory_observations: trial
                .advisory_observations
                .iter()
                .map(|line| sanitize_excerpt(line))
                .take(8)
                .collect(),
            pp_tps: trial.pp_tps.is_finite().then_some(trial.pp_tps),
            tg_tps: trial.tg_tps.is_finite().then_some(trial.tg_tps),
            recorded_at: recorded_at.to_string(),
        };
        let encoded = serde_json::to_vec(&record)
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        file.write_all(&encoded)?;
        file.write_all(b"\n")?;
        file.flush()?;
        file.sync_data()
    }

    /// Mark the run complete. Interrupted runs deliberately keep their marker.
    pub fn finish(&self) -> std::io::Result<()> {
        match std::fs::remove_file(&self.active_marker) {
            Ok(()) => prune_completed(&self.log_dir, self.retained_runs),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                prune_completed(&self.log_dir, self.retained_runs)
            }
            Err(error) => Err(error),
        }
    }
}

/// Read every complete manifest line. A truncated final append is ignored; an
/// invalid interior line is reported.
pub fn read_manifest(path: &Path) -> Result<Vec<RunManifestRecord>, String> {
    let raw = std::fs::read_to_string(path).map_err(|error| error.to_string())?;
    let lines: Vec<&str> = raw.lines().collect();
    let mut records = Vec::new();
    for (index, line) in lines.iter().enumerate() {
        match serde_json::from_str(line) {
            Ok(record) => records.push(record),
            Err(_) if index + 1 == lines.len() && !raw.ends_with('\n') => break,
            Err(error) => return Err(format!("manifest line {}: {error}", index + 1)),
        }
    }
    Ok(records)
}

fn prune_completed(log_dir: &Path, retained_runs: usize) -> std::io::Result<()> {
    let mut completed = std::fs::read_dir(log_dir)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("jsonl"))
        .filter(|path| {
            path.file_stem()
                .and_then(|stem| stem.to_str())
                .is_some_and(|stem| !log_dir.join(format!("{stem}.active")).exists())
        })
        .collect::<Vec<_>>();
    completed.sort_by(|left, right| {
        let modified = |path: &Path| {
            std::fs::metadata(path)
                .and_then(|meta| meta.modified())
                .ok()
        };
        modified(left)
            .cmp(&modified(right))
            .then_with(|| left.cmp(right))
    });
    let remove_count = completed.len().saturating_sub(retained_runs);
    for manifest in completed.into_iter().take(remove_count) {
        if let Ok(records) = read_manifest(&manifest) {
            for record in records {
                let log = PathBuf::from(record.log_path);
                if log.parent() == Some(log_dir) {
                    let _ = std::fs::remove_file(log);
                }
            }
        }
        std::fs::remove_file(manifest)?;
    }
    Ok(())
}

fn safe_component(value: &str) -> String {
    let safe: String = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_') {
                ch
            } else {
                '-'
            }
        })
        .take(64)
        .collect();
    let trimmed = safe.trim_matches('-');
    if trimmed.is_empty() {
        "run".to_string()
    } else {
        trimmed.to_string()
    }
}

fn sanitize_observed(
    observed: &BTreeMap<String, serde_json::Value>,
) -> BTreeMap<String, serde_json::Value> {
    observed
        .iter()
        .take(32)
        .map(|(key, value)| {
            let value = value
                .as_str()
                .map(|text| serde_json::Value::from(sanitize_excerpt(text)))
                .unwrap_or_else(|| value.clone());
            (key.chars().take(128).collect(), value)
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use localbench_scoring::score::Telemetry;

    fn measured() -> Trial {
        Trial {
            startup_ok: true,
            measurement_usable: true,
            pp_tps: 100.0,
            tg_tps: 50.0,
            telemetry: Telemetry::default(),
            ..Trial::default()
        }
    }

    /// LocalHub#160, second half. A content failure used to reach the manifest
    /// as a bare stage/reason pair, while `diagnostic_excerpt` carried server
    /// log text that is perfectly healthy for exactly this failure — so the
    /// log could not distinguish a flood from an empty or merely terse reply,
    /// and diagnosing it meant relaunching the server and replaying the
    /// request.
    #[test]
    fn a_content_failure_records_what_the_model_actually_said() {
        let dir = tempfile::tempdir().unwrap();
        let mut ledger = RunLedger::create(dir.path(), "run:content", 2).unwrap();
        let mut trial = measured();
        trial.measurement_usable = false;
        trial.failure = Some(localbench_scoring::score::TrialFailure {
            stage: localbench_scoring::score::TrialFailureStage::Content,
            reason: localbench_scoring::score::TrialFailureReason::DegenerateContent,
            detail: "degenerate response text; visible reply: ////////".to_string(),
        });
        ledger
            .record(
                "baseline",
                "KvK=q8_0;KvV=q8_0",
                "live",
                &Overrides::new(),
                &mut trial,
                "now",
            )
            .unwrap();

        let written = std::fs::read_to_string(ledger.path()).unwrap();
        let record: RunManifestRecord = serde_json::from_str(written.trim()).unwrap();
        let detail = record.failure_detail.unwrap_or_default();
        assert!(
            !detail.is_empty(),
            "a content failure must carry its detail into the manifest"
        );
        assert!(
            detail.contains("degenerate response text"),
            "the gate message was dropped: {detail}"
        );
        assert!(
            detail.contains("////////"),
            "the rejected reply was dropped: {detail}"
        );
    }

    /// A trial that succeeded has nothing to explain, and an empty detail must
    /// not add a null-ish key to every healthy line.
    #[test]
    fn a_healthy_trial_records_no_failure_detail() {
        let dir = tempfile::tempdir().unwrap();
        let mut ledger = RunLedger::create(dir.path(), "run:ok", 2).unwrap();
        ledger
            .record(
                "baseline",
                "KvK=q8_0",
                "live",
                &Overrides::new(),
                &mut measured(),
                "now",
            )
            .unwrap();
        let written = std::fs::read_to_string(ledger.path()).unwrap();
        assert!(
            !written.contains("failure_detail"),
            "a healthy trial should not carry the key at all: {written}"
        );
    }

    #[test]
    fn append_survives_a_truncated_last_line_and_never_contains_a_prompt() {
        let dir = tempfile::tempdir().unwrap();
        let mut ledger = RunLedger::create(dir.path(), "run:one", 2).unwrap();
        let mut trial = measured();
        trial.diagnostic_excerpt = "Authorization: Bearer super-secret".to_string();
        trial.launch_args = vec![
            "--api-key".to_string(),
            "launch-secret-value".to_string(),
            "--chat-template".to_string(),
            "x".repeat(2_000),
        ];
        trial.observed_configuration.insert(
            "build_info".to_string(),
            serde_json::Value::from("y".repeat(4_000)),
        );
        ledger
            .record(
                "baseline",
                "KvK=q8_0",
                "live",
                &Overrides::new(),
                &mut trial,
                "now",
            )
            .unwrap();
        let path = ledger.path().to_path_buf();
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        file.write_all(b"{truncated").unwrap();
        drop(file);

        let records = read_manifest(&path).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].diagnostic_excerpt, "[REDACTED]");
        assert_eq!(records[0].launch_args[1], "[REDACTED]");
        assert_eq!(records[0].launch_args[3].chars().count(), 512);
        assert!(records[0].observed_configuration["build_info"]
            .as_str()
            .is_some_and(|value| value.chars().count() <= 2_048));
        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(!raw.contains("super-secret"));
        assert!(!raw.contains("launch-secret-value"));
        assert!(!raw.contains("stress prompt"));

        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        file.write_all(b"\n").unwrap();
        drop(file);
        assert!(
            read_manifest(&path).is_err(),
            "an invalid complete line is corruption, not a truncated append"
        );
    }

    #[test]
    fn retention_keeps_active_and_only_bounds_completed_runs() {
        let dir = tempfile::tempdir().unwrap();
        let first_log = dir.path().join("trial-first.log");
        std::fs::write(&first_log, "first").unwrap();
        let mut first = RunLedger::create(dir.path(), "001", 1).unwrap();
        let first_path = first.path().to_path_buf();
        let mut first_trial = measured();
        first_trial.diagnostic = Some(localbench_scoring::score::TrialDiagnosticRef {
            log_path: first_log.display().to_string(),
            ..Default::default()
        });
        first
            .record(
                "baseline",
                "first",
                "live",
                &Overrides::new(),
                &mut first_trial,
                "1",
            )
            .unwrap();
        first.finish().unwrap();

        let second_log = dir.path().join("trial-second.log");
        std::fs::write(&second_log, "second").unwrap();
        let mut second = RunLedger::create(dir.path(), "002", 1).unwrap();
        let second_path = second.path().to_path_buf();
        let mut second_trial = measured();
        second_trial.diagnostic = Some(localbench_scoring::score::TrialDiagnosticRef {
            log_path: second_log.display().to_string(),
            ..Default::default()
        });
        second
            .record(
                "baseline",
                "second",
                "live",
                &Overrides::new(),
                &mut second_trial,
                "2",
            )
            .unwrap();
        second.finish().unwrap();
        let active = RunLedger::create(dir.path(), "003", 1).unwrap();

        assert!(!first_path.exists());
        assert!(!first_log.exists());
        assert!(second_path.exists());
        assert!(second_log.exists());
        assert!(active.path().exists());
        assert!(dir.path().join("run-003.active").exists());
        let completed = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| entry.path().extension().and_then(|ext| ext.to_str()) == Some("jsonl"))
            .filter(|entry| {
                let path = entry.path();
                let stem = path.file_stem().and_then(|stem| stem.to_str()).unwrap();
                !dir.path().join(format!("{stem}.active")).exists()
            })
            .count();
        assert_eq!(completed, 1);
    }
}
