//! The launcher-contract consumer side: gate on the version triple and the
//! product-version floor before trusting a launcher, and expose this
//! product's own version envelope for the reverse gate.

use std::io::Write;
use std::path::PathBuf;

use localbox_launcher::fetch::{human_bytes, DownloadKind, DownloadProgress};
use localbox_launcher::launcher::LlamaLauncher;
use localx_llama_core::{
    assert_compatible, LauncherError, LauncherVersion, RUNTIME_LLAMACPP, TARGET_LOCALBOX,
};

/// The artifacts a tuning run needs on disk: the GGUF and nothing else. A
/// trial never loads a vision projector or a draft model, so a catalog that
/// configures them must not make `findbest` pull them.
pub const TUNING_DOWNLOADS: &[DownloadKind] = &[DownloadKind::Gguf];

/// The fetch step `ensure_gguf_on_disk` delegates to: given the kinds to
/// fetch and a progress sink, put them on disk and return their paths.
/// Injected so the pre-flight's selection, output, and failure handling can be
/// pinned without a network.
pub type FetchFiles<'a> = dyn FnMut(&[DownloadKind], &mut dyn FnMut(&DownloadProgress<'_>)) -> Result<Vec<PathBuf>, String>
    + 'a;

/// Make sure the model's GGUF for `quant` is on disk before the first trial,
/// downloading it through the launcher's own resumable fetch when it is not —
/// the same download a LocalBox launch performs, so a model that has never
/// been launched can still be tuned. Returns the GGUF path.
///
/// # Errors
/// A plain-language string when the catalog cannot name the file or the
/// download fails.
pub fn ensure_gguf_on_disk(
    launcher: &LlamaLauncher,
    key: &str,
    quant: Option<&str>,
    out: &mut dyn Write,
) -> Result<PathBuf, String> {
    let mut fetch = |kinds: &[DownloadKind], report: &mut dyn FnMut(&DownloadProgress<'_>)| {
        let runtime = tokio::runtime::Runtime::new().map_err(|e| e.to_string())?;
        runtime
            .block_on(launcher.fetch_model_files(key, quant, kinds, report))
            .map_err(|e| e.to_string())
    };
    ensure_gguf_on_disk_with(launcher, key, quant, out, &mut fetch)
}

/// [`ensure_gguf_on_disk`] over an injected fetch step.
///
/// # Errors
/// As [`ensure_gguf_on_disk`].
pub fn ensure_gguf_on_disk_with(
    launcher: &LlamaLauncher,
    key: &str,
    quant: Option<&str>,
    out: &mut dyn Write,
    fetch: &mut FetchFiles<'_>,
) -> Result<PathBuf, String> {
    let targets = launcher
        .model_download_targets(key, quant)
        .map_err(|e| e.to_string())?;
    let gguf = targets
        .iter()
        .find(|target| target.kind == DownloadKind::Gguf)
        .ok_or_else(|| format!("{key} names no GGUF file to tune"))?;
    if gguf.present {
        return Ok(gguf.path.clone());
    }
    writeln!(
        out,
        "GGUF not on disk — downloading {} …",
        gguf.path.display()
    )
    .map_err(|e| e.to_string())?;
    let mut last_step: Option<u64> = None;
    let mut report = |progress: &DownloadProgress<'_>| {
        // Coarse, log-friendly progress: one line per 5% (or per 512 MiB when
        // the total is unknown), never a spinner.
        let step = match progress.total {
            Some(total) if total > 0 => progress.received.saturating_mul(20) / total,
            _ => progress.received / (512 * 1024 * 1024),
        };
        if Some(step) != last_step {
            last_step = Some(step);
            match progress.total {
                Some(total) => eprintln!(
                    "  {} / {} ({}%)",
                    human_bytes(progress.received),
                    human_bytes(total),
                    progress.received.saturating_mul(100) / total.max(1)
                ),
                None => eprintln!("  {}", human_bytes(progress.received)),
            }
        }
    };
    let fetched = fetch(TUNING_DOWNLOADS, &mut report)?;
    writeln!(out, "Download complete.").map_err(|e| e.to_string())?;
    fetched
        .into_iter()
        .next()
        .ok_or_else(|| format!("{key}: the download reported success but produced no GGUF"))
}

/// This product's contract API version.
pub const API_VERSION: u32 = 1;
/// This product's launcher-export schema version.
pub const LAUNCHER_EXPORT_VERSION: u32 = 3;

/// The version envelope this product presents to a launcher's reverse gate.
#[must_use]
pub fn own_version(product_version: &str) -> LauncherVersion {
    LauncherVersion {
        version: product_version.to_string(),
        api_version: API_VERSION,
        launcher_export_version: LAUNCHER_EXPORT_VERSION,
        supported_targets: vec![TARGET_LOCALBOX.to_string(), "LocalLLMLauncher".to_string()],
        supported_runtimes: vec![RUNTIME_LLAMACPP.to_string()],
    }
}

/// Compare two dotted numeric versions (missing segments read as 0).
fn version_at_least(version: &str, minimum: &str) -> bool {
    let parse = |v: &str| -> Vec<u64> {
        v.split(['.', '-', '+'])
            .map_while(|part| part.parse::<u64>().ok())
            .collect()
    };
    let (v, m) = (parse(version), parse(minimum));
    for i in 0..v.len().max(m.len()) {
        let a = v.get(i).copied().unwrap_or(0);
        let b = m.get(i).copied().unwrap_or(0);
        if a != b {
            return a > b;
        }
    }
    true
}

/// Gate a launcher before the tuner trusts it: the product version must reach
/// `min_version`, the version triple must satisfy the contract floor
/// (`api_version >= 1`, `launcher_export_version >= 1`), and the
/// LocalBox/llamacpp pairing must be declared supported.
///
/// # Errors
/// Returns [`LauncherError::Incompatible`] naming exactly what failed.
pub fn assert_launcher_usable(
    version: &LauncherVersion,
    min_version: &str,
) -> Result<(), LauncherError> {
    if !version_at_least(&version.version, min_version) {
        return Err(LauncherError::Incompatible(format!(
            "launcher {} is below required {min_version}",
            version.version
        )));
    }
    assert_compatible(version, TARGET_LOCALBOX, RUNTIME_LLAMACPP)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn launcher_with_modules(dir: &std::path::Path) -> LlamaLauncher {
        use serde_json::{Map, Value};
        let catalog: Map<String, Value> = serde_json::from_str(
            r#"{
            "Models": {
                "vis": {
                    "Root": "vis",
                    "Repo": "owner/vis-GGUF",
                    "File": "vis.gguf",
                    "VisionModule": "mmproj-Q8_0.gguf",
                    "DraftModule": "vis-draft.gguf",
                    "Contexts": { "": 8192 }
                }
            }
        }"#,
        )
        .unwrap();
        let settings: Map<String, Value> = serde_json::from_str(&format!(
            r#"{{ "LlamaCppGgufRoot": {} }}"#,
            Value::from(dir.to_str().unwrap())
        ))
        .unwrap();
        let catalog =
            localbox_launcher::catalog::Catalog::from_layers(&Map::new(), &catalog, &settings)
                .unwrap();
        LlamaLauncher::new(catalog, "3.1.0", dir.join("home"), 24)
    }

    #[test]
    fn a_present_gguf_is_returned_without_touching_the_configured_modules() {
        // The tuner asks for the GGUF only: with it on disk, nothing is fetched
        // — not the projector, not the drafter the catalog also configures.
        let dir = tempfile::tempdir().unwrap();
        let launcher = launcher_with_modules(dir.path());
        let folder = dir.path().join("vis");
        std::fs::create_dir_all(&folder).unwrap();
        let gguf = folder.join("vis.gguf");
        std::fs::write(&gguf, b"weights").unwrap();

        let mut out = Vec::new();
        let path = ensure_gguf_on_disk(&launcher, "vis", None, &mut out).unwrap();

        assert_eq!(path, gguf);
        assert!(
            out.is_empty(),
            "no download announced: {}",
            String::from_utf8_lossy(&out)
        );
        assert!(!folder.join("mmproj-Q8_0.gguf").exists());
        assert!(!folder.join("vis-draft.gguf").exists());
        assert_eq!(TUNING_DOWNLOADS, &[DownloadKind::Gguf]);
    }

    #[test]
    fn a_missing_gguf_is_fetched_as_the_only_kind_and_its_path_returned() {
        // The branch that closes the gap: no GGUF on disk, a catalog that also
        // configures a projector and a drafter. The pre-flight asks the fetch
        // step for exactly `[Gguf]`, announces the download, reports progress,
        // and returns the path the fetch produced; the modules stay untouched.
        let dir = tempfile::tempdir().unwrap();
        let launcher = launcher_with_modules(dir.path());
        let folder = dir.path().join("vis");
        let gguf = folder.join("vis.gguf");
        assert!(!gguf.exists());

        let mut requested: Vec<Vec<DownloadKind>> = Vec::new();
        let mut fetch = |kinds: &[DownloadKind], report: &mut dyn FnMut(&DownloadProgress<'_>)| {
            requested.push(kinds.to_vec());
            std::fs::create_dir_all(&folder).unwrap();
            std::fs::write(&gguf, b"weights").unwrap();
            report(&DownloadProgress {
                kind: DownloadKind::Gguf,
                path: &gguf,
                received: 7,
                total: Some(7),
            });
            Ok(vec![gguf.clone()])
        };
        let mut out = Vec::new();
        let path = ensure_gguf_on_disk_with(&launcher, "vis", None, &mut out, &mut fetch).unwrap();

        assert_eq!(path, gguf);
        assert_eq!(requested, vec![vec![DownloadKind::Gguf]]);
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("GGUF not on disk"), "{text}");
        assert!(text.contains("Download complete."), "{text}");
        assert!(!folder.join("mmproj-Q8_0.gguf").exists());
        assert!(!folder.join("vis-draft.gguf").exists());
    }

    #[test]
    fn a_failed_download_propagates_and_never_claims_completion() {
        let dir = tempfile::tempdir().unwrap();
        let launcher = launcher_with_modules(dir.path());
        let mut fetch = |_: &[DownloadKind], _: &mut dyn FnMut(&DownloadProgress<'_>)| {
            Err("download failed: 503".to_string())
        };
        let mut out = Vec::new();
        let error =
            ensure_gguf_on_disk_with(&launcher, "vis", None, &mut out, &mut fetch).unwrap_err();
        assert!(error.contains("503"), "{error}");
        let text = String::from_utf8(out).unwrap();
        assert!(!text.contains("Download complete."), "{text}");
    }

    #[test]
    fn an_unknown_model_fails_before_any_download() {
        let dir = tempfile::tempdir().unwrap();
        let launcher = launcher_with_modules(dir.path());
        let mut out = Vec::new();
        let error = ensure_gguf_on_disk(&launcher, "nope", None, &mut out).unwrap_err();
        assert!(error.contains("nope"), "{error}");
        assert!(out.is_empty());
    }
    use std::path::PathBuf;

    use localx_llama_core::{BackendSession, KvTypes, Launcher, Mode, ModelDef};

    #[test]
    fn our_own_envelope_passes_the_reverse_gate() {
        let envelope = own_version("1.2.1");
        assert_compatible(&envelope, TARGET_LOCALBOX, RUNTIME_LLAMACPP).expect("self-consistent");
        assert_eq!(envelope.api_version, 1);
        assert!(envelope.launcher_export_version >= 1);
    }

    #[test]
    fn the_usability_gate_checks_version_floor_then_triple() {
        let mut version = own_version("1.2.1");
        assert!(assert_launcher_usable(&version, "1.0.0").is_ok());
        assert!(assert_launcher_usable(&version, "1.2.1").is_ok());
        let err = assert_launcher_usable(&version, "1.3.0").unwrap_err();
        assert!(err.to_string().contains("below required 1.3.0"));
        version.api_version = 0;
        assert!(assert_launcher_usable(&version, "1.0.0").is_err());
    }

    #[test]
    fn dotted_version_comparison_reads_numerically() {
        assert!(version_at_least("1.10.0", "1.9.9"));
        assert!(!version_at_least("1.9.9", "1.10.0"));
        assert!(version_at_least("2.0", "2.0.0"));
        assert!(version_at_least("1.2.1-beta.3", "1.2.1"));
    }

    /// A minimal mock proving the contract seam is implementable and
    /// consumable without a real launcher — the compile-time analogue of the
    /// function-existence + parameter-obligation checks.
    struct MockLauncher;

    impl Launcher for MockLauncher {
        fn version(&self) -> LauncherVersion {
            LauncherVersion {
                version: "1.2.1".to_string(),
                api_version: 1,
                launcher_export_version: 1,
                supported_targets: vec![TARGET_LOCALBOX.to_string()],
                supported_runtimes: vec![RUNTIME_LLAMACPP.to_string()],
            }
        }
        fn model_def(&self, key: &str) -> Result<ModelDef, LauncherError> {
            Err(LauncherError::UnknownModel(key.to_string()))
        }
        fn gguf_path(
            &self,
            _def: &ModelDef,
            _quant: Option<&str>,
        ) -> Result<PathBuf, LauncherError> {
            Ok(PathBuf::from("model.gguf"))
        }
        fn context_value(&self, _def: &ModelDef, _key: &str) -> Result<u32, LauncherError> {
            Ok(65_536)
        }
        fn resolve_context_key(&self, _def: &ModelDef, key: &str) -> Result<String, LauncherError> {
            Ok(key.to_string())
        }
        fn vision_module_path(&self, _key: &str, _def: &ModelDef) -> Option<PathBuf> {
            None
        }
        fn resolve_quant_key(&self, _def: &ModelDef, quant: &str) -> Result<String, LauncherError> {
            Ok(quant.to_string())
        }
        fn vram_gb(&self) -> u32 {
            24
        }
        fn server_binary(&self, _mode: Mode, _ni: bool) -> Result<PathBuf, LauncherError> {
            Ok(PathBuf::from("llama-server"))
        }
        fn bench_binary(&self, _ni: bool) -> Option<PathBuf> {
            None
        }
        fn perplexity_binary(&self, _ni: bool, _mode: Mode) -> Option<PathBuf> {
            None
        }
        fn install_root(&self, _mode: Mode) -> PathBuf {
            PathBuf::from(".")
        }
        fn kv_types(&self, _def: &ModelDef) -> KvTypes {
            KvTypes {
                k: "q8_0".to_string(),
                v: "q8_0".to_string(),
            }
        }
        fn kv_type_supported(&self, kv_type: &str, mode: Mode) -> bool {
            !kv_type.starts_with("turbo") || mode != Mode::Native
        }
        fn free_port(&self, start: u16) -> Result<u16, LauncherError> {
            Ok(start)
        }
        fn wait_server(&self, _port: u16, _timeout: u32) -> Result<(), LauncherError> {
            Ok(())
        }
        fn stop_server(&self, _quiet: bool) {}
        fn set_backend_session(&self, _session: &BackendSession) {}
        fn expand_path(&self, path: &str) -> PathBuf {
            PathBuf::from(path)
        }
    }

    #[test]
    fn a_mock_launcher_satisfies_the_consumer_seam() {
        // Trait objects work (the tuner holds `&dyn Launcher`), and the
        // version gate accepts the mock.
        let launcher: &dyn Launcher = &MockLauncher;
        assert_launcher_usable(&launcher.version(), "1.0.0").expect("usable");
        assert_eq!(launcher.vram_gb(), 24);
        assert!(launcher.kv_type_supported("turbo3", Mode::Turboquant));
        assert!(!launcher.kv_type_supported("turbo3", Mode::Native));
        assert!(matches!(
            launcher.model_def("nope"),
            Err(LauncherError::UnknownModel(_))
        ));
    }
}
