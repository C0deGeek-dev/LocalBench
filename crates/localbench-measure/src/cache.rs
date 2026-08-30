//! The fingerprinted trial cache: measured trials keyed by their exact config
//! so a resumed/repeated run never re-pays for a measurement, and invalidated
//! wholesale when anything that shapes a measurement changes (the
//! fingerprint).
//!
//! Stability rules, pinned by test:
//! - The cache key is `driver|signature` — plain strings, no hashing, so an
//!   entry is inspectable and stable forever.
//! - The fingerprint hash is FNV-1a over canonical JSON — **never the std
//!   hasher**, whose output may change across releases and would silently
//!   invalidate every persisted cache.
//! - Soak/guard/recovery/verify phases are never cached: their results depend
//!   on session state (context pressure), not just the config.
//! - Persistence is crash-safe: temp-file write + atomic swap, previous copy
//!   kept as `.bak` and recovered from on load.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use localbench_scoring::score::Overrides;
use localbench_search::overrides::candidate_signature;

/// Which measurement driver produced a trial.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Driver {
    Server,
    Bench,
}

impl Driver {
    fn as_str(self) -> &'static str {
        match self {
            Driver::Server => "server",
            Driver::Bench => "bench",
        }
    }
}

/// The cache key for a measured config: `driver|signature`.
#[must_use]
pub fn trial_cache_key(driver: Driver, overrides: &Overrides) -> String {
    format!("{}|{}", driver.as_str(), candidate_signature(overrides))
}

/// Whether a phase's trials may be cached. Verify/guard/soak/recovery phases
/// are excluded — their outcomes depend on session state, not just the config.
#[must_use]
pub fn cache_eligible_phase(phase: &str) -> bool {
    !(phase == "verify"
        || phase.starts_with("context_guard")
        || phase.starts_with("context_soak")
        || phase.starts_with("stress_recovery"))
}

/// 64-bit FNV-1a. Stable across builds/platforms (unlike the std hasher), so a
/// persisted fingerprint hash never silently changes.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// The stable content hash of any JSON-serializable value: FNV-1a over its
/// canonical (sorted-key, compact) JSON rendering.
#[must_use]
pub fn stable_json_hash(value: &serde_json::Value) -> String {
    format!("{:016x}", fnv1a(canonical_json(value).as_bytes()))
}

/// Render JSON with object keys sorted, so semantically-equal values hash
/// equal regardless of construction order.
fn canonical_json(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Object(map) => {
            let sorted: BTreeMap<&String, &serde_json::Value> = map.iter().collect();
            let parts: Vec<String> = sorted
                .into_iter()
                .map(|(k, v)| {
                    format!(
                        "{}:{}",
                        serde_json::Value::from(k.as_str()),
                        canonical_json(v)
                    )
                })
                .collect();
            format!("{{{}}}", parts.join(","))
        }
        serde_json::Value::Array(items) => {
            let parts: Vec<String> = items.iter().map(canonical_json).collect();
            format!("[{}]", parts.join(","))
        }
        other => other.to_string(),
    }
}

/// Everything that shapes a measurement. Two runs with equal fingerprints may
/// share cached trials; any difference invalidates the whole cache.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Fingerprint {
    pub schema: u32,
    pub tuner_version: String,
    /// Public inference surface used for every sample.
    pub measurement_protocol: String,
    /// Resource-sampling contract used to populate balanced-score telemetry.
    /// The default lets an older fingerprint deserialize and then miss by this
    /// named field instead of failing as an unreadable cache.
    #[serde(default)]
    pub telemetry_protocol: String,
    /// Stable hash of the request envelope and generation controls.
    pub request_shape_hash: String,
    /// Stable hash of the parser/template/thinking inputs used by chat.
    pub chat_template_hash: String,
    /// Stable hash of catalog model fields that can shape emitted argv.
    pub model_definition_hash: String,
    /// Required response content/timing contract.
    pub response_schema: String,
    /// Effective session-shaping launch params after settings/defaults.
    pub session: BTreeMap<String, serde_json::Value>,
    pub key: String,
    pub context_key: String,
    pub context_tokens: u32,
    pub mode: String,
    pub quant: String,
    pub prompt_length: String,
    /// Stable hash of the benchmark prompt + its sizing knobs.
    pub prompt_hash: String,
    pub optimize: String,
    pub profile: String,
    pub search_strategy: String,
    pub beam_width: u32,
    pub runs: u32,
    pub vram_gb: u32,
    pub gpu_names: Vec<String>,
    pub llamacpp_build: String,
    /// GGUF identity: path + size + mtime.
    pub gguf: GgufIdentity,
    pub allowed_kv_types: Vec<String>,
    pub stress_targets: Vec<u32>,
    pub stress_min_free_vram_gb: f64,
    pub skip_mtp: bool,
    pub skip_stress_test: bool,
}

/// The GGUF file identity the fingerprint pins.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GgufIdentity {
    pub path: String,
    pub size_bytes: u64,
    pub last_write_utc: String,
}

impl Fingerprint {
    /// The fingerprint's stable hash.
    ///
    /// # Panics
    /// Never: the type serializes infallibly (all-owned, finite fields).
    #[must_use]
    pub fn hash(&self) -> String {
        let value = serde_json::to_value(self).unwrap_or(serde_json::Value::Null);
        stable_json_hash(&value)
    }
}

/// The fingerprint fields that differ between an expected and a persisted
/// fingerprint, so a mismatch names *what* changed instead of shrugging.
#[must_use]
pub fn fingerprint_diff_keys(
    expected: &serde_json::Value,
    actual: &serde_json::Value,
) -> Vec<String> {
    let (Some(expected), Some(actual)) = (expected.as_object(), actual.as_object()) else {
        return vec!["fingerprint".to_string()];
    };
    let mut keys: Vec<&String> = expected.keys().chain(actual.keys()).collect();
    keys.sort();
    keys.dedup();
    keys.into_iter()
        .filter(|key| {
            match (expected.get(*key), actual.get(*key)) {
                (Some(e), Some(a)) => stable_json_hash(e) != stable_json_hash(a),
                _ => true, // present on one side only
            }
        })
        .cloned()
        .collect()
}

/// One measured trial as handed to [`TrialCache::put`].
#[derive(Debug, Clone)]
pub struct MeasuredTrial {
    pub driver: Driver,
    pub overrides: Overrides,
    pub phase: String,
    pub oom: bool,
    pub startup_ok: bool,
    pub measurement_usable: bool,
    pub measured_at: String,
    /// The full trial payload as measured (schema-agnostic).
    pub trial: serde_json::Value,
}

/// One cached measured trial.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CacheEntry {
    pub cache_key: String,
    pub signature: String,
    pub driver: Driver,
    /// `OK` / `OOM` (startup failures are never cached).
    pub status: String,
    pub measured_at: String,
    /// The full trial payload as measured (schema-agnostic).
    pub trial: serde_json::Value,
}

/// The persisted cache payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CacheFile {
    schema: u32,
    updated_at: String,
    fingerprint_hash: String,
    fingerprint: serde_json::Value,
    entries: Vec<CacheEntry>,
}

/// Why a load produced no reusable entries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadOutcome {
    /// Entries loaded from the live file.
    Loaded(usize),
    /// Entries recovered from the `.bak` left by an interrupted save.
    RecoveredFromBackup(usize),
    /// The persisted fingerprint differs; names the differing fields.
    FingerprintMismatch(Vec<String>),
    /// A file existed but held nothing reusable.
    NoReusableEntries,
    /// No cache file exists yet.
    NotFound,
    /// The file could not be parsed.
    LoadFailed(String),
}

/// The in-memory trial cache for one run.
#[derive(Debug)]
pub struct TrialCache {
    path: PathBuf,
    fingerprint: serde_json::Value,
    fingerprint_hash: String,
    entries: BTreeMap<String, CacheEntry>,
    enabled: bool,
}

impl TrialCache {
    /// A cache rooted at `path` for a run with `fingerprint`. Loads reusable
    /// entries (live file first, then the `.bak` recovery copy) when the
    /// persisted fingerprint hash matches.
    pub fn open(
        path: impl Into<PathBuf>,
        fingerprint: &Fingerprint,
        enabled: bool,
    ) -> (Self, LoadOutcome) {
        let fingerprint_value =
            serde_json::to_value(fingerprint).unwrap_or(serde_json::Value::Null);
        let mut cache = Self {
            path: path.into(),
            fingerprint_hash: fingerprint.hash(),
            fingerprint: fingerprint_value,
            entries: BTreeMap::new(),
            enabled,
        };
        if !enabled {
            return (cache, LoadOutcome::NoReusableEntries);
        }

        let mut any_existed = false;
        let mut last_error: Option<String> = None;
        let mut loaded: Option<(CacheFile, bool)> = None;
        let backup = cache.path.with_extension("json.bak");
        for (candidate, from_backup) in [(cache.path.clone(), false), (backup, true)] {
            if !candidate.is_file() {
                continue;
            }
            any_existed = true;
            let parsed: Result<CacheFile, String> = std::fs::read_to_string(&candidate)
                .map_err(|e| e.to_string())
                .and_then(|raw| serde_json::from_str(&raw).map_err(|e| e.to_string()));
            match parsed {
                Ok(data) => {
                    loaded = Some((data, from_backup));
                    break;
                }
                Err(error) => last_error = Some(error),
            }
        }

        let outcome = match loaded {
            Some((data, from_backup)) => {
                if data.fingerprint_hash != cache.fingerprint_hash {
                    LoadOutcome::FingerprintMismatch(fingerprint_diff_keys(
                        &cache.fingerprint,
                        &data.fingerprint,
                    ))
                } else {
                    for entry in data.entries {
                        cache.entries.insert(entry.cache_key.clone(), entry);
                    }
                    match (cache.entries.len(), from_backup) {
                        (0, _) => LoadOutcome::NoReusableEntries,
                        (count, true) => LoadOutcome::RecoveredFromBackup(count),
                        (count, false) => LoadOutcome::Loaded(count),
                    }
                }
            }
            None => match (any_existed, last_error) {
                (true, Some(error)) => LoadOutcome::LoadFailed(error),
                (true, None) => LoadOutcome::NoReusableEntries,
                (false, _) => LoadOutcome::NotFound,
            },
        };
        (cache, outcome)
    }

    /// The number of reusable entries currently held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache holds no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Look up a cached trial for a config in a phase. The returned trial is
    /// re-tagged for the requesting phase (phase, overrides, `cached` marker,
    /// original measurement time, driver).
    #[must_use]
    pub fn get(
        &self,
        driver: Driver,
        overrides: &Overrides,
        phase: &str,
    ) -> Option<serde_json::Value> {
        if !self.enabled || !cache_eligible_phase(phase) {
            return None;
        }
        let entry = self.entries.get(&trial_cache_key(driver, overrides))?;
        let mut trial = entry.trial.clone();
        if let Some(map) = trial.as_object_mut() {
            map.insert("phase".to_string(), phase.into());
            map.insert(
                "overrides".to_string(),
                serde_json::to_value(overrides).unwrap_or(serde_json::Value::Null),
            );
            map.insert("cached".to_string(), true.into());
            map.insert("cached_from".to_string(), entry.measured_at.clone().into());
            map.insert("driver".to_string(), entry.driver.as_str().into());
        }
        Some(trial)
    }

    /// Record a measured trial. Only decisive results are cached — a healthy
    /// measurement or a definite OOM; a plain startup failure is transient and
    /// is measured fresh next time. Ineligible phases are skipped. Returns
    /// whether the entry was stored.
    pub fn put(&mut self, measured: &MeasuredTrial) -> bool {
        if !self.enabled || !cache_eligible_phase(&measured.phase) {
            return false;
        }
        if !measured.oom && !measured.measurement_usable {
            return false;
        }
        let cache_key = trial_cache_key(measured.driver, &measured.overrides);
        self.entries.insert(
            cache_key.clone(),
            CacheEntry {
                cache_key,
                signature: candidate_signature(&measured.overrides),
                driver: measured.driver,
                status: if measured.oom { "OOM" } else { "OK" }.to_string(),
                measured_at: measured.measured_at.clone(),
                trial: measured.trial.clone(),
            },
        );
        true
    }

    /// Persist the cache crash-safely: serialize to a sibling temp file, then
    /// atomically swap it into place, keeping the previous copy as `.bak`. An
    /// interruption can only truncate the temp file, never the live cache.
    ///
    /// # Errors
    /// Returns the I/O error if neither the atomic swap nor the direct
    /// fallback write succeeds.
    pub fn save(&self, updated_at: &str) -> std::io::Result<()> {
        if !self.enabled {
            return Ok(());
        }
        let payload = CacheFile {
            schema: 1,
            updated_at: updated_at.to_string(),
            fingerprint_hash: self.fingerprint_hash.clone(),
            fingerprint: self.fingerprint.clone(),
            entries: self.entries.values().cloned().collect(),
        };
        let json = serde_json::to_string(&payload)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let tmp = self.path.with_extension("json.tmp");
        let bak = self.path.with_extension("json.bak");
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&tmp, &json)?;
        if self.path.is_file() {
            // Keep the previous good copy, then move the new one into place.
            std::fs::rename(&self.path, &bak)?;
        }
        std::fs::rename(&tmp, &self.path)
    }

    /// The cache file path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use localbench_search::overrides::overrides_of;

    fn fingerprint() -> Fingerprint {
        Fingerprint {
            schema: 1,
            tuner_version: "test".to_string(),
            measurement_protocol: "chat".to_string(),
            telemetry_protocol: "telemetry".to_string(),
            request_shape_hash: "request".to_string(),
            chat_template_hash: "template".to_string(),
            model_definition_hash: "model-definition".to_string(),
            response_schema: "response".to_string(),
            session: BTreeMap::new(),
            key: "model-x".to_string(),
            context_key: "64k".to_string(),
            context_tokens: 65_536,
            mode: "turboquant".to_string(),
            quant: "apex".to_string(),
            prompt_length: "long".to_string(),
            prompt_hash: "abc".to_string(),
            optimize: "coding-agent".to_string(),
            profile: "both".to_string(),
            search_strategy: "beam".to_string(),
            beam_width: 3,
            runs: 3,
            vram_gb: 24,
            gpu_names: vec!["RTX 4090".to_string()],
            llamacpp_build: "b1234".to_string(),
            gguf: GgufIdentity {
                path: "C:/models/x.gguf".to_string(),
                size_bytes: 123,
                last_write_utc: "2026-01-01T00:00:00Z".to_string(),
            },
            allowed_kv_types: vec!["q8_0".to_string()],
            stress_targets: vec![16_384],
            stress_min_free_vram_gb: 0.5,
            skip_mtp: false,
            skip_stress_test: false,
        }
    }

    fn ov() -> Overrides {
        overrides_of(&[("NCpuMoe", 20.into()), ("UbatchSize", 1024.into())])
    }

    fn healthy_trial() -> serde_json::Value {
        serde_json::json!({ "pp_tps": 700.0, "tg_tps": 50.0, "startup_ok": true, "oom": false })
    }

    fn measured(
        driver: Driver,
        overrides: &Overrides,
        phase: &str,
        oom: bool,
        startup_ok: bool,
    ) -> MeasuredTrial {
        MeasuredTrial {
            driver,
            overrides: overrides.clone(),
            phase: phase.to_string(),
            oom,
            startup_ok,
            measurement_usable: startup_ok && !oom,
            measured_at: "2026-07-01T00:00:00Z".to_string(),
            trial: healthy_trial(),
        }
    }

    #[test]
    fn cache_key_is_driver_pipe_signature() {
        assert_eq!(
            trial_cache_key(Driver::Server, &ov()),
            "server|NCpuMoe=20;UbatchSize=1024"
        );
        assert_eq!(
            trial_cache_key(Driver::Bench, &ov()),
            "bench|NCpuMoe=20;UbatchSize=1024"
        );
    }

    #[test]
    fn stable_hash_is_pinned_and_order_independent() {
        // Pinned vector: a changed hash silently invalidates every persisted
        // cache, so this must be a conscious decision.
        assert_eq!(
            stable_json_hash(&serde_json::json!({"a": 1, "b": "x"})),
            "cfcc937b86ef6c1d"
        );
        // Key order does not change the hash (canonical rendering).
        let a: serde_json::Value =
            serde_json::from_str(r#"{"k1": 1, "k2": [true, null]}"#).unwrap();
        let b: serde_json::Value =
            serde_json::from_str(r#"{"k2": [true, null], "k1": 1}"#).unwrap();
        assert_eq!(stable_json_hash(&a), stable_json_hash(&b));
        assert_ne!(
            stable_json_hash(&serde_json::json!({"a": 1})),
            stable_json_hash(&serde_json::json!({"a": 2}))
        );
    }

    #[test]
    fn soak_guard_recovery_and_verify_phases_are_never_cached() {
        for phase in [
            "verify",
            "context_guard_16384",
            "context_soak_32768",
            "stress_recovery_1",
        ] {
            assert!(!cache_eligible_phase(phase), "{phase} must be ineligible");
        }
        for phase in ["seed", "moe", "beam_2", "fine_tune"] {
            assert!(cache_eligible_phase(phase), "{phase} must be eligible");
        }
    }

    #[test]
    fn cache_round_trips_and_retags_the_hit() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trial-cache-x.json");
        let (mut cache, outcome) = TrialCache::open(&path, &fingerprint(), true);
        assert_eq!(outcome, LoadOutcome::NotFound);

        assert!(cache.put(&measured(Driver::Server, &ov(), "seed", false, true)));
        cache.save("2026-07-01T00:00:01Z").unwrap();

        let (reopened, outcome) = TrialCache::open(&path, &fingerprint(), true);
        assert_eq!(outcome, LoadOutcome::Loaded(1));
        let hit = reopened
            .get(Driver::Server, &ov(), "beam_1")
            .expect("cache hit");
        assert_eq!(hit["cached"], serde_json::json!(true));
        assert_eq!(hit["phase"], serde_json::json!("beam_1"));
        assert_eq!(
            hit["cached_from"],
            serde_json::json!("2026-07-01T00:00:00Z")
        );
        assert_eq!(hit["driver"], serde_json::json!("server"));
        assert_eq!(hit["pp_tps"], serde_json::json!(700.0));
        // A different driver misses.
        assert!(reopened.get(Driver::Bench, &ov(), "beam_1").is_none());
        // An ineligible phase never reads the cache.
        assert!(reopened
            .get(Driver::Server, &ov(), "context_soak_1")
            .is_none());
    }

    #[test]
    fn save_creates_a_missing_parent_directory() {
        let dir = tempfile::tempdir().unwrap();
        // A path two levels deep whose parent dir does not exist yet.
        let path = dir.path().join("tuner").join("trial-cache-x.json");
        let (mut cache, outcome) = TrialCache::open(&path, &fingerprint(), true);
        assert_eq!(outcome, LoadOutcome::NotFound);
        assert!(cache.put(&measured(Driver::Server, &ov(), "seed", false, true)));
        cache.save("2026-07-01T00:00:01Z").unwrap();

        // It persisted and reopens cleanly.
        let (reopened, outcome) = TrialCache::open(&path, &fingerprint(), true);
        assert_eq!(outcome, LoadOutcome::Loaded(1));
        assert!(reopened.get(Driver::Server, &ov(), "beam_1").is_some());
    }

    #[test]
    fn only_decisive_results_are_cached() {
        let dir = tempfile::tempdir().unwrap();
        let (mut cache, _) = TrialCache::open(dir.path().join("c.json"), &fingerprint(), true);
        // A definite OOM is decisive and cached.
        assert!(cache.put(&measured(Driver::Server, &ov(), "moe", true, false)));
        // A plain startup failure is transient — never cached.
        let other = overrides_of(&[("NCpuMoe", 25.into())]);
        assert!(!cache.put(&measured(Driver::Server, &other, "moe", false, false)));
        // A ready server with an unusable response is also transient.
        let response_failure = MeasuredTrial {
            measurement_usable: false,
            ..measured(Driver::Server, &other, "moe", false, true)
        };
        assert!(!cache.put(&response_failure));
        // An ineligible phase is skipped.
        assert!(!cache.put(&measured(
            Driver::Server,
            &other,
            "context_soak_1",
            false,
            true
        )));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn a_changed_fingerprint_invalidates_and_names_the_difference() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trial-cache-x.json");
        let (mut cache, _) = TrialCache::open(&path, &fingerprint(), true);
        cache.put(&measured(Driver::Server, &ov(), "seed", false, true));
        cache.save("t").unwrap();

        let mut changed = fingerprint();
        changed.measurement_protocol = "raw-completion-v0".to_string();
        changed.telemetry_protocol = "different-sampler".to_string();
        changed.chat_template_hash = "other-template".to_string();
        changed.session.insert("parallel".to_string(), (-1).into());
        let (reopened, outcome) = TrialCache::open(&path, &changed, true);
        assert!(reopened.is_empty());
        match outcome {
            LoadOutcome::FingerprintMismatch(keys) => {
                assert!(keys.contains(&"measurement_protocol".to_string()));
                assert!(keys.contains(&"telemetry_protocol".to_string()));
                assert!(keys.contains(&"chat_template_hash".to_string()));
                assert!(keys.contains(&"session".to_string()));
                assert!(!keys.contains(&"key".to_string()));
            }
            other => panic!("expected a fingerprint mismatch, got {other:?}"),
        }
    }

    #[test]
    fn a_corrupt_live_file_recovers_from_the_backup() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trial-cache-x.json");
        let (mut cache, _) = TrialCache::open(&path, &fingerprint(), true);
        cache.put(&measured(Driver::Server, &ov(), "seed", false, true));
        cache.save("t1").unwrap();
        // A second save moves the good copy to .bak; then the live file is
        // truncated mid-write.
        cache.save("t2").unwrap();
        std::fs::write(&path, "{ truncated garbage").unwrap();

        let (recovered, outcome) = TrialCache::open(&path, &fingerprint(), true);
        assert_eq!(outcome, LoadOutcome::RecoveredFromBackup(1));
        assert!(recovered.get(Driver::Server, &ov(), "seed").is_some());
    }

    #[test]
    fn a_disabled_cache_never_reads_or_writes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("c.json");
        let (mut cache, _) = TrialCache::open(&path, &fingerprint(), false);
        assert!(!cache.put(&measured(Driver::Server, &ov(), "seed", false, true)));
        assert!(cache.get(Driver::Server, &ov(), "seed").is_none());
        cache.save("t").unwrap();
        assert!(!path.exists(), "disabled cache writes nothing");
    }
}
