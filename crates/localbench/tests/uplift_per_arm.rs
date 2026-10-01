//! The per-arm uplift surface end to end through the real binary: emit an
//! arm's configuration, stage it, run each arm alone against a stand-in
//! `localpilot`, combine the two arm files, and render the receipt.
//!
//! The stand-in answers from what is staged, the way the real solver would:
//! with learning on it answers correctly and records the lesson it used; with
//! learning off it does not know and records nothing. No model runs.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const TASK_SET: &str = r#"{
  "schema": 1,
  "name": "headroom-fixture",
  "tasks": [
    {
      "id": "migrate",
      "prompt": "How do I migrate the foo database?",
      "expect": { "mode": "substring", "value": "foo db sync" },
      "lesson_ids": ["lesson-migrate"]
    }
  ],
  "lessons": [
    { "id": "lesson-migrate", "body": "Use foo db sync to migrate the foo database." }
  ]
}"#;

fn localbench(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_localbench"))
        .args(args)
        .output()
        .expect("run localbench")
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

/// A stand-in `localpilot`: answers from the staged configuration and leaves
/// the session log the audit is read from. `used` is the memory id a
/// learning-on turn records.
fn stand_in(dir: &Path, workspace: &Path, used: &str) -> PathBuf {
    std::fs::write(
        workspace.join("lesson-log.jsonl"),
        format!(
            "{{\"kind\":{{\"type\":\"memories_used\",\"memories\":[{{\"id\":\"{used}\",\"score\":5,\"layer\":\"memory\"}}]}}}}\n"
        ),
    )
    .unwrap();
    std::fs::write(
        workspace.join("baseline-log.jsonl"),
        "{\"kind\":{\"type\":\"turn_done\"}}\n",
    )
    .unwrap();
    std::fs::create_dir_all(workspace.join(".localpilot").join("sessions")).unwrap();
    if cfg!(windows) {
        let path = dir.join("standin.cmd");
        std::fs::write(
            &path,
            "@echo off\r\n\
             findstr /c:\"enabled = true\" .localmind.toml >nul\r\n\
             if errorlevel 1 goto off\r\n\
             copy /y lesson-log.jsonl .localpilot\\sessions\\turn.jsonl >nul\r\n\
             echo Run foo db sync.\r\n\
             exit /b 0\r\n\
             :off\r\n\
             copy /y baseline-log.jsonl .localpilot\\sessions\\turn.jsonl >nul\r\n\
             echo I do not know.\r\n",
        )
        .unwrap();
        path
    } else {
        let path = dir.join("standin.sh");
        std::fs::write(
            &path,
            "#!/bin/sh\n\
             if grep -q 'enabled = true' .localmind.toml; then\n\
               cp lesson-log.jsonl .localpilot/sessions/turn.jsonl\n\
               echo 'Run foo db sync.'\n\
             else\n\
               cp baseline-log.jsonl .localpilot/sessions/turn.jsonl\n\
               echo 'I do not know.'\n\
             fi\n",
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        path
    }
}

struct Fixture {
    dir: tempfile::TempDir,
    workspace: PathBuf,
    task_set: PathBuf,
    solver: PathBuf,
}

fn fixture(used: &str) -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let workspace = dir.path().join("workspace");
    std::fs::create_dir_all(&workspace).unwrap();
    let task_set = dir.path().join("tasks.json");
    std::fs::write(&task_set, TASK_SET).unwrap();
    let solver = stand_in(dir.path(), &workspace, used);
    Fixture {
        dir,
        workspace,
        task_set,
        solver,
    }
}

impl Fixture {
    /// Stage the arm's configuration exactly as the CLI emits it.
    fn stage(&self, arm: &str) {
        let config = localbench(&["uplift", "--emit-arm-config", arm]);
        assert!(config.status.success(), "{}", stderr(&config));
        std::fs::write(self.workspace.join(".localmind.toml"), &config.stdout).unwrap();
    }

    fn run_arm(&self, arm: &str, binding: &str) -> (Output, PathBuf) {
        let out = self.dir.path().join(format!("{arm}.json"));
        let output = localbench(&[
            "uplift",
            "--task-set",
            self.task_set.to_str().unwrap(),
            "--arm",
            arm,
            "--workspace",
            self.workspace.to_str().unwrap(),
            "--model",
            "fixture-model",
            "--trials",
            "2",
            "--timeout",
            "60",
            "--localpilot",
            self.solver.to_str().unwrap(),
            "--intended",
            "mem-1",
            "--binding",
            binding,
            "--out",
            out.to_str().unwrap(),
        ]);
        (output, out)
    }

    fn pair(&self, binding: &str) -> (PathBuf, PathBuf) {
        self.stage("baseline");
        let (output, baseline) = self.run_arm("baseline", binding);
        assert!(output.status.success(), "{}", stderr(&output));
        self.stage("lessons");
        let (output, lessons) = self.run_arm("lessons", binding);
        assert!(output.status.success(), "{}", stderr(&output));
        (baseline, lessons)
    }
}

fn combine(first: &Path, second: &Path, out: &Path) -> Output {
    localbench(&[
        "uplift",
        "--combine",
        first.to_str().unwrap(),
        "--with",
        second.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
    ])
}

#[test]
fn staged_arms_run_alone_and_combine_into_a_bound_receipt() {
    let fixture = fixture("mem-1");
    let (baseline, lessons) = fixture.pair("bind-42");
    let out = fixture.dir.path().join("receipt.json");

    let combined = combine(&lessons, &baseline, &out);
    assert!(combined.status.success(), "{}", stderr(&combined));

    let receipt: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
    assert_eq!(receipt["schema"], "localbench-uplift-v2");
    assert_eq!(receipt["identity"]["binding"], "bind-42");
    assert_eq!(receipt["identity"]["task_set"]["task_count"], 1);
    assert!(receipt["identity"]["task_set"]["digest"]
        .as_str()
        .unwrap()
        .starts_with("sha256:"));
    assert_eq!(
        receipt["identity"]["lessons"]["injection"]["intended"],
        serde_json::json!(["mem-1"])
    );
    assert_eq!(
        receipt["identity"]["lessons"]["injection"]["mode"],
        "retrieved"
    );
    assert_eq!(receipt["identity"]["baseline"]["trials"], 2);
    assert_eq!(receipt["uplift"]["verdict"], "uplift");
    assert_eq!(receipt["void"], serde_json::Value::Null);
    assert_eq!(
        receipt["arms"][1]["injection"]["injected"],
        serde_json::json!(["mem-1"])
    );

    let rendered = localbench(&["uplift", "--report", out.to_str().unwrap()]);
    let text = stdout(&rendered);
    assert!(text.contains("binding bind-42"), "{text}");
    assert!(text.contains("**uplift**"), "{text}");
}

#[test]
fn an_arm_in_a_workspace_staged_for_the_other_is_refused() {
    let fixture = fixture("mem-1");
    fixture.stage("lessons");
    let (output, out) = fixture.run_arm("baseline", "bind-1");
    assert!(!output.status.success());
    assert!(
        stderr(&output).contains("mis-staged baseline arm"),
        "{}",
        stderr(&output)
    );
    assert!(!out.exists(), "a refused arm writes no arm file");
}

#[test]
fn a_lesson_arm_that_used_another_memory_makes_the_receipt_void() {
    let fixture = fixture("mem-something-else");
    let (baseline, lessons) = fixture.pair("bind-7");
    let out = fixture.dir.path().join("receipt.json");

    let combined = combine(&baseline, &lessons, &out);

    assert_eq!(combined.status.code(), Some(3), "{}", stderr(&combined));
    assert!(stderr(&combined).contains("VOID"));
    let receipt: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
    assert_eq!(receipt["uplift"], serde_json::Value::Null);
    assert!(receipt["void"].as_str().unwrap().contains("lessons"));
    let rendered = localbench(&["uplift", "--report", out.to_str().unwrap()]);
    assert!(stdout(&rendered).contains("**VOID**"));
}

#[test]
fn arm_files_of_two_requests_do_not_combine() {
    let fixture = fixture("mem-1");
    let (baseline, _) = fixture.pair("bind-a");
    let lessons_a = fixture.dir.path().join("lessons-a.json");
    std::fs::rename(fixture.dir.path().join("lessons.json"), &lessons_a).unwrap();
    let (_, lessons_b) = fixture.pair("bind-b");
    let out = fixture.dir.path().join("receipt.json");

    // `pair` rewrote baseline.json for bind-b; the kept lesson file is bind-a's.
    let combined = combine(&baseline, &lessons_a, &out);
    assert!(!combined.status.success());
    assert!(
        stderr(&combined).contains("binding"),
        "{}",
        stderr(&combined)
    );
    assert!(!out.exists());

    // Half a pair is not a result either.
    let half = combine(&lessons_b, &lessons_b, &out);
    assert!(!half.status.success());
    assert!(stderr(&half).contains("not one baseline arm and one lesson arm"));
}

#[test]
fn the_seed_pack_and_arm_configurations_are_emitted_without_a_model() {
    let fixture = fixture("mem-1");
    let pack = localbench(&[
        "uplift",
        "--task-set",
        fixture.task_set.to_str().unwrap(),
        "--emit-seed-pack",
    ]);
    assert!(pack.status.success());
    let value: serde_json::Value = serde_json::from_str(&stdout(&pack)).unwrap();
    assert_eq!(
        value["lessons"][0]["body"],
        "Use foo db sync to migrate the foo database."
    );
    assert_eq!(
        stdout(&localbench(&["uplift", "--emit-arm-config", "baseline"])),
        "[learning]\nenabled = false\n"
    );
    let unknown = localbench(&["uplift", "--emit-arm-config", "warm"]);
    assert!(!unknown.status.success());
}
