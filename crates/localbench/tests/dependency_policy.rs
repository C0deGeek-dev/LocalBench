//! An audit exception for an unbuilt transport must not mask an active one.

use std::{path::Path, process::Command};

#[test]
fn optional_quic_stays_out_of_all_supported_workspace_feature_graphs(
) -> Result<(), Box<dyn std::error::Error>> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("workspace root is missing")?;
    let output = Command::new(env!("CARGO"))
        .env("CARGO_TERM_COLOR", "never")
        .args([
            "tree",
            "--locked",
            "--workspace",
            "--all-features",
            "--target",
            "all",
            "--edges",
            "normal,build,dev",
            "--prefix",
            "none",
            "--format",
            "{p}",
        ])
        .current_dir(root)
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let graph = String::from_utf8(output.stdout)?;
    let active: Vec<_> = graph
        .lines()
        .filter(|line| line.starts_with("quinn"))
        .collect();
    assert!(
        active.is_empty(),
        "QUIC is now active: remove its audit exceptions and adopt patched versions before shipping: {active:?}"
    );
    Ok(())
}
