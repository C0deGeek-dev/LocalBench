//! Offline grade support: the vendored cargo cache for the `--network=none`
//! Rust grade, and the language-aware test counting (shared table in
//! `localx_eval_core::grade`; a generic fallback for unlisted languages).
//!
//! The offline rust grade runs with `--network=none` + `CARGO_NET_OFFLINE`, so
//! any dependency an exercise declares (or an agentic arm adds) must be
//! pre-vendored into a warmed shared cargo registry. A miss still fails loud
//! in the grade tail (a cargo offline error, detected below), never silently —
//! an under-vendored cache must read as an infrastructure gap, not a solve
//! failure.

use std::path::Path;

use serde::{Deserialize, Serialize};

pub use localx_eval_core::grade::{count_tests, count_tests_generic, grade, GradeOutcome, Lang};

/// Parse a grade language label onto the shared table; `None` means unlisted
/// (grade with [`count_tests_generic`], fail closed).
#[must_use]
pub fn lang_of(label: &str) -> Option<Lang> {
    match label {
        "rust" => Some(Lang::Rust),
        "python" => Some(Lang::Python),
        "cpp" => Some(Lang::Cpp),
        "javascript" => Some(Lang::Javascript),
        "go" => Some(Lang::Go),
        "java" => Some(Lang::Java),
        _ => None,
    }
}

/// Count tests for a language label: the shared table when known, the generic
/// passed-sum fallback otherwise.
#[must_use]
pub fn test_count(label: &str, output: &str) -> u32 {
    match lang_of(label) {
        Some(lang) => count_tests(lang, output),
        None => count_tests_generic(output),
    }
}

/// Common crates an agentic arm reaches for on the exercise corpus but that no
/// exercise *declares* — pre-vendored so an agent that adds one still builds
/// offline. Pinned to the major/minor the corpus era resolves. Declared deps
/// take precedence over these on a name collision.
pub const CURATED_RUST_DEPS: &[(&str, &str)] = &[
    ("rayon", "1"),           // data parallelism
    ("itertools", "0.13"),    // iterator combinators, very commonly reached
    ("regex", "1"),           // pattern/parsing exercises
    ("num-bigint", "0.4"),    // big-integer exercises
    ("num-traits", "0.2"),    // numeric generics
    ("num-integer", "0.1"),   // gcd/lcm helpers
    ("once_cell", "1"),       // lazy statics (current idiom)
    ("lazy_static", "1"),     // lazy statics (older idiom)
    ("thiserror", "1"),       // derive-based error types
    ("permutohedron", "0.2"), // permutations
    ("counter", "0.5"),       // multiset counting
];

/// Where a cache dependency came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DepSource {
    /// Declared by an exercise's `Cargo.toml` (authoritative on collision).
    Declared,
    /// From the curated common-model-dep list.
    Curated,
}

/// One crate to vendor into the offline registry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheDep {
    pub name: String,
    pub version: String,
    pub source: DepSource,
}

/// A grade-support failure.
#[derive(Debug, thiserror::Error)]
pub enum GradeSupportError {
    /// The corpus layout is not what the scanner expects.
    #[error("rust practice dir not found: {0}")]
    PracticeDirMissing(String),
    /// The corpus could not be read.
    #[error("could not scan the corpus: {0}")]
    Io(#[from] std::io::Error),
}

/// Parse the `[dependencies]` / `[dev-dependencies]` / `[build-dependencies]`
/// crates from one `Cargo.toml`'s text. Handles both the simple
/// `name = "1.0"` form and the table form `name = { version = "1.0", ... }`.
/// A dep without a version string (git/path) is skipped — it can't be
/// vendored offline and would fail loud in the grade either way.
#[must_use]
pub fn cargo_toml_dependencies(toml_text: &str) -> Vec<(String, String)> {
    const DEP_SECTIONS: &[&str] = &["dependencies", "dev-dependencies", "build-dependencies"];
    let mut in_deps = false;
    let mut result = Vec::new();
    for raw in toml_text.lines() {
        let line = raw.trim();
        if line.starts_with('[') && line.ends_with(']') {
            let section = line[1..line.len() - 1].trim();
            in_deps = DEP_SECTIONS
                .iter()
                .any(|s| section == *s || section.ends_with(&format!(".{s}")));
            continue;
        }
        if !in_deps || line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((name, rhs)) = line.split_once('=') else {
            continue;
        };
        let name = name.trim();
        if name.is_empty()
            || !name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        {
            continue;
        }
        let rhs = rhs.trim();
        let version = if let Some(stripped) = rhs.strip_prefix('"') {
            stripped.split('"').next().map(str::to_string)
        } else {
            rhs.split_once("version")
                .and_then(|(_, tail)| tail.split('"').nth(1))
                .map(str::to_string)
        };
        if let Some(version) = version {
            result.push((name.to_string(), version));
        }
    }
    result
}

/// The union of every rust exercise's declared deps (scanned from
/// `<corpus>/rust/exercises/practice/*/Cargo.toml`) and the curated
/// common-model-dep list, deduped by crate name with the *declared* version
/// winning. Deterministic order (declared first, then curated), so the warm
/// step is loggable and reproducible.
///
/// # Errors
/// Returns [`GradeSupportError::PracticeDirMissing`] when the corpus layout is
/// wrong — a mis-pointed corpus must fail loud, not warm an empty cache.
pub fn rust_cargo_cache_deps(
    corpus_root: &Path,
    include_curated: bool,
) -> Result<Vec<CacheDep>, GradeSupportError> {
    let practice = corpus_root.join("rust").join("exercises").join("practice");
    if !practice.is_dir() {
        return Err(GradeSupportError::PracticeDirMissing(
            practice.display().to_string(),
        ));
    }

    let mut declared: Vec<(String, String)> = Vec::new();
    let mut dirs: Vec<_> = std::fs::read_dir(&practice)?
        .flatten()
        .filter(|e| e.path().is_dir())
        .map(|e| e.path())
        .collect();
    dirs.sort();
    for dir in dirs {
        let toml = dir.join("Cargo.toml");
        let Ok(text) = std::fs::read_to_string(&toml) else {
            continue;
        };
        for (name, version) in cargo_toml_dependencies(&text) {
            // First declaration wins a tie across exercises; record once.
            if !declared.iter().any(|(n, _)| *n == name) {
                declared.push((name, version));
            }
        }
    }

    let mut out: Vec<CacheDep> = declared
        .iter()
        .map(|(name, version)| CacheDep {
            name: name.clone(),
            version: version.clone(),
            source: DepSource::Declared,
        })
        .collect();
    if include_curated {
        for (name, version) in CURATED_RUST_DEPS {
            if declared.iter().any(|(n, _)| n == name) {
                continue; // the declared version wins
            }
            out.push(CacheDep {
                name: (*name).to_string(),
                version: (*version).to_string(),
                source: DepSource::Curated,
            });
        }
    }
    Ok(out)
}

/// The throwaway crate manifest whose `[dependencies]` lists every crate to
/// vendor; `cargo fetch` against it populates the shared registry. Edition
/// 2021 + a fixed name so the warm is reproducible.
#[must_use]
pub fn cargo_warm_manifest(deps: &[CacheDep]) -> String {
    let mut sorted: Vec<&CacheDep> = deps.iter().collect();
    sorted.sort_by(|a, b| a.name.cmp(&b.name));
    let mut manifest = String::from(
        "[package]\nname = \"aider-cargo-warm\"\nversion = \"0.0.0\"\nedition = \"2021\"\n\n[dependencies]\n",
    );
    for dep in sorted {
        manifest.push_str(&format!("{} = \"{}\"\n", dep.name, dep.version));
    }
    manifest
}

/// Whether graded output shows cargo refusing/failing a network fetch under
/// the offline grade — the loud signal of an under-vendored cache. Such a cell
/// is an infrastructure gap (re-warm the cache), never a solve failure to
/// score quietly.
#[must_use]
pub fn is_offline_fetch_error(output: &str) -> bool {
    const MARKERS: &[&str] = &[
        "--offline was passed",
        "can't make an http request in offline mode",
        "failed to download",
        "no matching package named",
        "unable to get packages from source",
    ];
    let lower = output.to_lowercase();
    MARKERS.iter().any(|marker| lower.contains(marker))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn parses_simple_and_table_dependency_forms() {
        let toml = r#"
[package]
name = "acronym"
version = "1.0.0"

[dependencies]
itertools = "0.13"
serde = { version = "1.0", features = ["derive"] }
local-helper = { path = "../helper" }

[dev-dependencies]
proptest = "1.4"
"#;
        let deps = cargo_toml_dependencies(toml);
        assert_eq!(
            deps,
            vec![
                ("itertools".to_string(), "0.13".to_string()),
                ("serde".to_string(), "1.0".to_string()),
                ("proptest".to_string(), "1.4".to_string()),
            ],
            "path deps are skipped; package fields never parse as deps"
        );
    }

    #[test]
    fn cache_deps_union_declared_and_curated_with_declared_winning() {
        let dir = tempfile::tempdir().unwrap();
        let practice = dir.path().join("rust").join("exercises").join("practice");
        std::fs::create_dir_all(practice.join("alphametics")).unwrap();
        std::fs::write(
            practice.join("alphametics").join("Cargo.toml"),
            "[dependencies]\nitertools = \"0.10\"\n",
        )
        .unwrap();

        let deps = rust_cargo_cache_deps(dir.path(), true).expect("scan");
        let itertools = deps.iter().find(|d| d.name == "itertools").unwrap();
        // The exercise declares 0.10; the curated 0.13 must NOT override it.
        assert_eq!(itertools.version, "0.10");
        assert_eq!(itertools.source, DepSource::Declared);
        // Curated crates the corpus does not declare are appended.
        let rayon = deps.iter().find(|d| d.name == "rayon").unwrap();
        assert_eq!(rayon.source, DepSource::Curated);
    }

    #[test]
    fn a_mispointed_corpus_fails_loud_not_an_empty_warm() {
        let dir = tempfile::tempdir().unwrap();
        assert!(matches!(
            rust_cargo_cache_deps(dir.path(), true),
            Err(GradeSupportError::PracticeDirMissing(_))
        ));
    }

    #[test]
    fn warm_manifest_is_sorted_and_reproducible() {
        let deps = vec![
            CacheDep {
                name: "zzz".to_string(),
                version: "1".to_string(),
                source: DepSource::Curated,
            },
            CacheDep {
                name: "aaa".to_string(),
                version: "2".to_string(),
                source: DepSource::Declared,
            },
        ];
        let manifest = cargo_warm_manifest(&deps);
        assert!(manifest.starts_with("[package]\nname = \"aider-cargo-warm\""));
        let deps_idx = manifest.find("aaa = \"2\"").unwrap();
        assert!(deps_idx < manifest.find("zzz = \"1\"").unwrap());
        assert_eq!(manifest, cargo_warm_manifest(&deps), "deterministic");
    }

    #[test]
    fn under_vendored_cache_is_detected_loud() {
        assert!(is_offline_fetch_error(
            "error: no matching package named `leftpad` found"
        ));
        assert!(is_offline_fetch_error(
            "error: attempting to make an HTTP request, but --offline was passed"
        ));
        assert!(is_offline_fetch_error("error: failed to download `serde`"));
        assert!(!is_offline_fetch_error(
            "error[E0308]: mismatched types\n --> src/lib.rs:4:5"
        ));
    }

    #[test]
    fn zero_test_exit_zero_scores_zero_through_the_label_table() {
        // The shared gate through the label front door: compile-only output.
        assert_eq!(test_count("rust", "Compiling foo\nFinished"), 0);
        assert!(!grade(Lang::Rust, 0, "Compiling foo\nFinished").passed);
        // Unlisted language falls back to the generic passed-sum.
        assert_eq!(test_count("zig", "All 9 passed."), 9);
        assert_eq!(test_count("zig", "gibberish"), 0);
    }
}
