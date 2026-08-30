//! Cross-repo launcher-contract conformance: gate the real LocalBox version
//! envelope with this product's consumer gate.
//!
//! Point `LOCALBOX_ROOT` at a LocalBox checkout (CI checks one out next to
//! this repo) and the test loads the committed envelope wire fixture LocalBox
//! pins against its live implementation, then runs the same usability gate
//! the tuner applies before trusting a launcher. Without `LOCALBOX_ROOT` the
//! test skips, so plain workspace runs stay hermetic.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fs;
use std::path::PathBuf;

use localbench::consumer::assert_launcher_usable;
use localx_llama_core::LauncherVersion;

/// The oldest launcher release this consumer supports; suffix-free numeric
/// floor, distinct from any human-facing release string.
const MIN_LAUNCHER_VERSION: &str = "1.0.0";

#[test]
fn the_real_localbox_envelope_passes_the_consumer_gate() {
    let Ok(root) = std::env::var("LOCALBOX_ROOT") else {
        eprintln!("LOCALBOX_ROOT not set; skipping the cross-repo conformance run");
        return;
    };
    let fixture =
        PathBuf::from(root).join("crates/localbox-launcher/tests/fixtures/launcher-envelope.json");
    let raw = fs::read_to_string(&fixture)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", fixture.display()));
    let envelope: LauncherVersion =
        serde_json::from_str(&raw).expect("the LocalBox envelope parses as the shared wire shape");

    assert_launcher_usable(&envelope, MIN_LAUNCHER_VERSION)
        .expect("the LocalBox launcher envelope satisfies the consumer gate");
}
