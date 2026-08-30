//! Tuner best-config export: merge a winner into the launcher-readable
//! `best-<key>.json` store.
//!
//! One store holds one entry per distinct
//! (quant, context, mode, VRAM, prompt-length, profile, vision) combination —
//! a new winner replaces only its own combination and **never overwrites
//! across the vision/text divide** (an mmproj changes VRAM and behaviour, so a
//! vision profile is a different animal from the text profile of the same
//! model). Every write validates against the shared store schema before it
//! lands, and files are UTF-8 without BOM.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use localx_llama_core::{TunerBestConfig, TunerEntry, TUNER_SCHEMA};

/// An export failure.
#[derive(Debug, thiserror::Error)]
pub enum ExportError {
    /// The existing store could not be read.
    #[error("could not read existing launcher profile at {path}: {reason}")]
    UnreadableStore { path: String, reason: String },
    /// The merged store no longer validates against the shared schema.
    #[error("the merged store does not validate against the tuner schema: {0}")]
    SchemaInvalid(String),
    /// The store could not be written.
    #[error("could not write launcher profile: {0}")]
    Io(#[from] std::io::Error),
}

/// The store/profile path conventions under a `~/.local-llm`-style root.
#[derive(Debug, Clone)]
pub struct StorePaths {
    root: PathBuf,
}

impl StorePaths {
    /// Paths under the given `~/.local-llm`-style root (injected, never
    /// probed, so path logic stays pure and testable).
    #[must_use]
    pub fn new(local_llm_root: impl Into<PathBuf>) -> Self {
        Self {
            root: local_llm_root.into(),
        }
    }

    /// The launcher-readable tuner store: `tuner/best-<key>.json`.
    #[must_use]
    pub fn best_config(&self, key: &str) -> PathBuf {
        self.root.join("tuner").join(format!("best-{key}.json"))
    }

    /// The trial cache: `tuner/trial-cache-<key>[-<contextKey>].json`.
    #[must_use]
    pub fn trial_cache(&self, key: &str, context_key: &str) -> PathBuf {
        let suffix = if context_key.trim().is_empty() {
            String::new()
        } else {
            format!("-{context_key}")
        };
        self.root
            .join("tuner")
            .join(format!("trial-cache-{key}{suffix}.json"))
    }
}

/// Provenance carried on the store and every exported entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExportProvenance {
    /// This product's version string.
    pub localbench_version: String,
    /// The export schema version (the version-triple member).
    pub launcher_export_version: u32,
    /// When the winner was measured (RFC3339 string, injected).
    pub measured_at: String,
}

/// One winner to export: the shared entry payload plus the vision flag that
/// partitions the store.
#[derive(Debug, Clone)]
pub struct ExportWinner {
    /// The shared-schema entry (quant/context/mode/vram/prompt/profile/score/
    /// args/overrides...).
    pub entry: TunerEntry,
    /// Whether the winner ran with a vision projector loaded.
    pub vision: bool,
}

const SCHEMA_VERSION: &str = "localbox-autobest-v1";
const SOURCE: &str = "localbench";

/// Whether two entry values occupy the same store slot: equal quant, context
/// key, mode, VRAM, prompt length, profile, AND vision flag. Absent fields
/// read as their historical defaults (`short` / `pure` / text-only).
fn same_slot(existing: &serde_json::Value, winner: &TunerEntry, winner_vision: bool) -> bool {
    let field = |name: &str| {
        existing
            .get(name)
            .cloned()
            .unwrap_or(serde_json::Value::Null)
    };
    let as_str = |name: &str, default: &str| {
        field(name)
            .as_str()
            .map_or_else(|| default.to_string(), str::to_string)
    };
    let prompt_length = serde_json::to_value(winner.prompt_length)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_default();
    let profile = serde_json::to_value(winner.profile)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_default();
    let mode = serde_json::to_value(winner.mode)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_default();

    field("quant").as_str() == Some(winner.quant.as_str())
        && field("contextKey").as_str() == Some(winner.context_key.as_str())
        && as_str("mode", "") == mode
        && field("vramGB").as_i64() == Some(winner.vram_gb)
        && as_str("prompt_length", "short") == prompt_length
        && as_str("profile", "pure") == profile
        && field("vision").as_bool().unwrap_or(false) == winner_vision
}

/// Merge a winner into `best-<key>.json`: load the existing store (tolerating
/// provenance fields the shared schema does not model), drop only the entry in
/// the same slot, append the winner with its provenance, validate the merged
/// document against the shared [`TunerBestConfig`] schema, and write it
/// UTF-8-no-BOM. Returns the written path.
///
/// # Errors
/// [`ExportError::UnreadableStore`] when an existing store cannot be parsed
/// (never silently clobbered), [`ExportError::SchemaInvalid`] when the merged
/// document would not round-trip the shared schema, or an I/O error.
pub fn export_best_config(
    path: &Path,
    key: &str,
    winner: &ExportWinner,
    provenance: &ExportProvenance,
) -> Result<PathBuf, ExportError> {
    // Load the existing store as raw JSON so unknown provenance fields survive.
    let mut store: serde_json::Map<String, serde_json::Value> = if path.is_file() {
        let raw = std::fs::read_to_string(path)?;
        serde_json::from_str(&raw).map_err(|e| ExportError::UnreadableStore {
            path: path.display().to_string(),
            reason: e.to_string(),
        })?
    } else {
        serde_json::Map::new()
    };

    let mut entries: Vec<serde_json::Value> = store
        .get("entries")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    entries.retain(|existing| !same_slot(existing, &winner.entry, winner.vision));

    let mut entry_value = serde_json::to_value(&winner.entry)
        .map_err(|e| ExportError::SchemaInvalid(e.to_string()))?;
    if let Some(map) = entry_value.as_object_mut() {
        map.insert("source".to_string(), SOURCE.into());
        map.insert("schema_version".to_string(), SCHEMA_VERSION.into());
        map.insert(
            "localbench_version".to_string(),
            provenance.localbench_version.clone().into(),
        );
        map.insert(
            "launcher_export_version".to_string(),
            provenance.launcher_export_version.into(),
        );
        map.insert("vision".to_string(), winner.vision.into());
        map.insert(
            "measured_at".to_string(),
            provenance.measured_at.clone().into(),
        );
    }
    entries.push(entry_value);

    store.insert("key".to_string(), key.into());
    store.insert("schema".to_string(), TUNER_SCHEMA.into());
    store.insert("source".to_string(), SOURCE.into());
    store.insert("schema_version".to_string(), SCHEMA_VERSION.into());
    store.insert(
        "localbench_version".to_string(),
        provenance.localbench_version.clone().into(),
    );
    store.insert(
        "launcher_export_version".to_string(),
        provenance.launcher_export_version.into(),
    );
    store.insert("entries".to_string(), entries.into());

    // The written document must round-trip the shared store schema — the
    // launcher-side contract — before it lands.
    let document = serde_json::Value::Object(store);
    let validated: TunerBestConfig = serde_json::from_value(document.clone())
        .map_err(|e| ExportError::SchemaInvalid(e.to_string()))?;
    if !validated.schema_supported() {
        return Err(ExportError::SchemaInvalid(format!(
            "schema {} is not the supported {TUNER_SCHEMA}",
            validated.schema
        )));
    }

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(&document)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    // UTF-8 without BOM: std::fs::write emits the bytes as-is.
    std::fs::write(path, json)?;
    Ok(path.to_path_buf())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use localx_llama_core::tuner::{Profile, PromptLength, SearchStrategy};
    use localx_llama_core::{Mode, Overrides};

    fn winner(quant: &str, score: f64, vision: bool) -> ExportWinner {
        ExportWinner {
            entry: TunerEntry {
                quant: quant.to_string(),
                context_key: "fast".to_string(),
                context_tokens: Some(65_536),
                mode: Mode::Turboquant,
                vram_gb: 24,
                prompt_length: PromptLength::Short,
                profile: Profile::Pure,
                search_strategy: Some(SearchStrategy::Beam),
                beam_width: Some(3),
                score,
                score_unit: "coding_agent_e2e_tps".to_string(),
                pure_score: Some(score),
                args: vec!["--ctx-size".to_string(), "65536".to_string()],
                overrides: Overrides::default(),
                measured_at: String::new(),
                tuner_version: localx_llama_core::CURRENT_TUNER_VERSION,
                trial_count: Some(30),
                gpu_names: None,
                llamacpp_build: None,
            },
            vision,
        }
    }

    fn provenance() -> ExportProvenance {
        ExportProvenance {
            localbench_version: "1.2.1".to_string(),
            launcher_export_version: 3,
            measured_at: "2026-07-01T00:00:00Z".to_string(),
        }
    }

    #[test]
    fn a_fresh_store_validates_against_the_shared_schema() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tuner").join("best-q36.json");
        export_best_config(&path, "q36", &winner("apex", 396.67, false), &provenance()).unwrap();

        let raw = std::fs::read(&path).unwrap();
        assert_ne!(&raw[..3], [0xEF, 0xBB, 0xBF], "no BOM");
        let store: TunerBestConfig = serde_json::from_slice(&raw).unwrap();
        assert!(store.schema_supported());
        assert_eq!(store.key, "q36");
        assert_eq!(store.entries.len(), 1);
        assert_eq!(store.entries[0].score, 396.67);
        assert_eq!(store.entries[0].search_strategy, Some(SearchStrategy::Beam));
        assert_eq!(store.entries[0].beam_width, Some(3));
        // Provenance fields survive as raw JSON alongside the shared shape.
        let value: serde_json::Value = serde_json::from_slice(&raw).unwrap();
        assert_eq!(value["schema_version"], "localbox-autobest-v1");
        assert_eq!(value["entries"][0]["launcher_export_version"], 3);
        assert_eq!(value["entries"][0]["vision"], false);
    }

    #[test]
    fn a_current_retune_replaces_its_version_four_slot_but_preserves_other_slots() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("best-q36.json");
        export_best_config(&path, "q36", &winner("apex", 300.0, false), &provenance()).unwrap();
        // A different quant lands in its own slot.
        export_best_config(
            &path,
            "q36",
            &winner("compact", 250.0, false),
            &provenance(),
        )
        .unwrap();
        let mut stale: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        for entry in stale["entries"].as_array_mut().unwrap() {
            entry["tuner_version"] = 4.into();
        }
        std::fs::write(&path, serde_json::to_string_pretty(&stale).unwrap()).unwrap();

        // Re-tuning the first slot replaces its stale entry, not appends, while
        // an unrelated stale slot remains available for explicit migration.
        export_best_config(&path, "q36", &winner("apex", 310.0, false), &provenance()).unwrap();

        let store: TunerBestConfig =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(store.entries.len(), 2);
        let apex = store.entries.iter().find(|e| e.quant == "apex").unwrap();
        assert_eq!(apex.score, 310.0);
        assert_eq!(apex.tuner_version, localx_llama_core::CURRENT_TUNER_VERSION);
        let compact = store.entries.iter().find(|e| e.quant == "compact").unwrap();
        assert_eq!(compact.tuner_version, 4);
    }

    #[test]
    fn vision_and_text_profiles_never_overwrite_each_other() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("best-q36.json");
        export_best_config(&path, "q36", &winner("apex", 300.0, false), &provenance()).unwrap();
        export_best_config(&path, "q36", &winner("apex", 250.0, true), &provenance()).unwrap();

        let value: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let entries = value["entries"].as_array().unwrap();
        assert_eq!(entries.len(), 2, "vision + text coexist for one quant");
        assert!(entries.iter().any(|e| e["vision"] == true));
        assert!(entries.iter().any(|e| e["vision"] == false));
    }

    #[test]
    fn an_unreadable_existing_store_is_never_clobbered() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("best-q36.json");
        std::fs::write(&path, "{ corrupt").unwrap();
        let err = export_best_config(&path, "q36", &winner("apex", 300.0, false), &provenance())
            .unwrap_err();
        assert!(matches!(err, ExportError::UnreadableStore { .. }));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "{ corrupt");
    }

    #[test]
    fn store_paths_follow_the_on_disk_conventions() {
        let paths = StorePaths::new("/home/x/.local-llm");
        assert!(paths
            .best_config("q36")
            .ends_with(PathBuf::from("tuner").join("best-q36.json")));
        assert!(paths
            .trial_cache("q36", "fast")
            .ends_with(PathBuf::from("tuner").join("trial-cache-q36-fast.json")));
        assert!(paths
            .trial_cache("q36", "")
            .ends_with(PathBuf::from("tuner").join("trial-cache-q36.json")));
    }
}
