//! The live lesson-uplift A/B: load a headroom task set, drive each arm N
//! trials through a driver (live = `localpilot print` per task, reading the
//! turn's memories-used audit from the session event log), then aggregate,
//! assert the injection contract, and emit the uplift report.
//!
//! The statistics and the injection-void contract live in
//! `localbench_scoring::uplift`; this module owns the task-set format, the
//! session-log audit parse, the live driver, and the report shape.

use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};

use localbench_measure::arms::{assert_arm_isolation, RawArmConfig};
use localbench_scoring::uplift::{
    aggregate, assert_injection, grade_answer, significance, Aggregate, ArmResult, Expect,
    InjectionSummary, MemoryUsed, Significance, TaskResult, SIGNIFICANCE_FLOOR,
};

use crate::solver::run_bounded;

/// One headroom task: a prompt the base model fails unguided, graded
/// deterministically.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UpliftTask {
    pub id: String,
    pub prompt: String,
    /// The deterministic expectation (`mode`/`value`).
    pub expect: Expect,
    #[serde(default)]
    pub case_sensitive: bool,
    /// The lessons that supply this task's answer (the injection assertion
    /// verifies the arm injected at least one of them).
    #[serde(default)]
    pub lesson_ids: Vec<String>,
}

/// One seed lesson in the task set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SeedLesson {
    pub id: String,
    pub body: String,
    #[serde(default = "default_category")]
    pub category: String,
    #[serde(default = "default_confidence")]
    pub confidence: f64,
    #[serde(default)]
    pub tags: Vec<String>,
}

fn default_category() -> String {
    "ProjectConvention".to_string()
}

fn default_confidence() -> f64 {
    0.9
}

/// A headroom task-set file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskSet {
    pub schema: u32,
    pub name: String,
    pub tasks: Vec<UpliftTask>,
    /// The seed pack the lesson arm seeds before its trials.
    #[serde(default)]
    pub lessons: Vec<SeedLesson>,
}

/// Load and validate a task set, failing loud on anything unusable.
///
/// # Errors
/// A plain-language message naming the defect.
pub fn load_task_set(path: &Path) -> Result<TaskSet, String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| format!("uplift task set not found: {}: {e}", path.display()))?;
    let set: TaskSet = serde_json::from_str(&raw)
        .map_err(|e| format!("{} does not parse: {e}", path.display()))?;
    if set.schema != 1 {
        return Err(format!(
            "unsupported task-set schema {} (expected 1)",
            set.schema
        ));
    }
    if set.tasks.is_empty() {
        return Err(format!("task set '{}' has no tasks", set.name));
    }
    for task in &set.tasks {
        if task.id.trim().is_empty() || task.prompt.trim().is_empty() {
            return Err(format!(
                "task set '{}' has a task without an id or prompt",
                set.name
            ));
        }
    }
    Ok(set)
}

/// Project a task set's lessons into the seed-pack JSON shape the lesson arm
/// seeds (`{ lessons: [{ body, category, confidence, tags }] }`).
#[must_use]
pub fn seed_pack(set: &TaskSet) -> serde_json::Value {
    serde_json::json!({
        "lessons": set
            .lessons
            .iter()
            .map(|lesson| {
                serde_json::json!({
                    "body": lesson.body,
                    "category": lesson.category,
                    "confidence": lesson.confidence,
                    "tags": lesson.tags,
                })
            })
            .collect::<Vec<_>>(),
    })
}

/// Parse the LAST turn's memories-used audit from a session event log (JSONL;
/// each line a tag-typed event). Pure and fixture-testable: no model run.
#[must_use]
pub fn memories_from_session_log(log_text: &str) -> Vec<MemoryUsed> {
    let mut last: Option<Vec<MemoryUsed>> = None;
    for line in log_text.lines() {
        let Ok(entry) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        let kind = &entry["kind"];
        if kind["type"] == "memories_used" {
            let memories = kind["memories"]
                .as_array()
                .map(|list| {
                    list.iter()
                        .filter_map(|m| m["id"].as_str())
                        .map(|id| MemoryUsed { id: id.to_string() })
                        .collect()
                })
                .unwrap_or_default();
            last = Some(memories);
        }
    }
    last.unwrap_or_default()
}

/// The newest session event log under a workspace's `.localpilot` store.
#[must_use]
pub fn latest_session_log(workspace: &Path) -> Option<PathBuf> {
    let dir = workspace.join(".localpilot").join("sessions");
    let entries = std::fs::read_dir(dir).ok()?;
    entries
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "jsonl"))
        .filter_map(|e| {
            let modified = e.metadata().ok()?.modified().ok()?;
            Some((modified, e.path()))
        })
        .max_by_key(|(modified, _)| *modified)
        .map(|(_, path)| path)
}

/// One trial's turn: the model's answer plus the recorded injection audit.
#[derive(Debug, Clone, PartialEq)]
pub struct Turn {
    pub answer: String,
    pub memories_used: Vec<MemoryUsed>,
}

/// Produces one turn per (task, trial) — live via `localpilot print`, or a
/// mock in tests.
pub trait UpliftDriver {
    /// Run one trial of a task.
    ///
    /// # Errors
    /// A plain-language message; the run stops (an uplift number from a
    /// partially-run arm would be meaningless).
    fn turn(&mut self, task: &UpliftTask, trial: u32) -> Result<Turn, String>;
}

/// The live driver: `localpilot print "<prompt>" --model <m>` in the
/// workspace, then the turn's memories-used from the newest session log.
pub struct PrintDriver {
    pub bin: String,
    pub workspace: PathBuf,
    pub model: String,
    pub timeout: Duration,
}

impl UpliftDriver for PrintDriver {
    fn turn(&mut self, task: &UpliftTask, _trial: u32) -> Result<Turn, String> {
        let args = vec![
            "print".to_string(),
            task.prompt.clone(),
            "--model".to_string(),
            self.model.clone(),
        ];
        let run = run_bounded(&self.bin, &args, Some(&self.workspace), self.timeout)?;
        if run.timed_out {
            return Err(format!(
                "'{} print' timed out after {}s (task '{}')",
                self.bin,
                self.timeout.as_secs(),
                task.id
            ));
        }
        if !run.exit_ok {
            return Err(format!(
                "'{} print' failed (task '{}'): {}",
                self.bin,
                task.id,
                run.stderr.trim()
            ));
        }
        let memories = latest_session_log(&self.workspace)
            .and_then(|log| std::fs::read_to_string(log).ok())
            .map(|text| memories_from_session_log(&text))
            .unwrap_or_default();
        Ok(Turn {
            answer: run.stdout,
            memories_used: memories,
        })
    }
}

/// Prove the deterministic grader still catches a known pass + fail pair
/// before a run spends a model on it.
///
/// # Errors
/// A refusal message when either reference misbehaves.
pub fn assert_grader_selftest() -> Result<(), String> {
    let expect = Expect::Substring("foo db sync".to_string());
    if !grade_answer("Run `foo db sync` to migrate.", &expect, false) {
        return Err(
            "uplift grader self-test failed: a known-correct answer did not match. \
             Refusing to run."
                .to_string(),
        );
    }
    if grade_answer("Run `foo migrate` to migrate.", &expect, false) {
        return Err(
            "uplift grader self-test failed: a known-wrong answer matched. Refusing to run."
                .to_string(),
        );
    }
    Ok(())
}

/// Run one arm of the A/B over the task set, `trials` times through the
/// driver. A contaminated baseline config is refused before spending.
///
/// # Errors
/// The isolation refusal, or the driver's failure.
pub fn run_uplift_arm(
    arm: &str,
    is_lesson_arm: bool,
    set: &TaskSet,
    driver: &mut dyn UpliftDriver,
    trials: u32,
    config: &RawArmConfig,
) -> Result<ArmResult, String> {
    if trials < 1 {
        return Err(format!("uplift arm '{arm}' needs at least one trial"));
    }
    assert_arm_isolation(arm, config).map_err(|e| e.to_string())?;

    let mut tasks = Vec::new();
    for task in &set.tasks {
        let mut passes = Vec::new();
        let mut memories_used = Vec::new();
        for trial in 0..trials {
            let turn = driver.turn(task, trial)?;
            passes.push(grade_answer(
                &turn.answer,
                &task.expect,
                task.case_sensitive,
            ));
            memories_used.push(turn.memories_used);
        }
        tasks.push(TaskResult {
            passes,
            memories_used,
        });
    }
    Ok(ArmResult {
        arm: arm.to_string(),
        is_lesson_arm,
        trials,
        tasks,
    })
}

/// One arm's row in the uplift report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UpliftArmRow {
    #[serde(flatten)]
    pub aggregate: Aggregate,
    pub injection: InjectionSummary,
}

/// The uplift report (`localbench-uplift-v1`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UpliftReport {
    pub schema: u32,
    pub task_set: String,
    pub model: String,
    pub trials: u32,
    pub arms: Vec<UpliftArmRow>,
    pub uplift: Significance,
}

/// Run the full lesson-on/off A/B: the grader self-test gate, both arms,
/// the injection contract (a result is VOID unless each arm injected as
/// configured), then aggregate + significance. Arm setup (seed lessons +
/// memory enable for the lesson arm; nothing for the baseline) is the
/// caller's job before building the drivers.
///
/// # Errors
/// Any gate refusal, driver failure, or injection void.
pub fn run_uplift(
    set: &TaskSet,
    baseline_driver: &mut dyn UpliftDriver,
    lesson_driver: &mut dyn UpliftDriver,
    intended_lesson_ids: &[String],
    trials: u32,
    model: &str,
) -> Result<UpliftReport, String> {
    assert_grader_selftest()?;

    let baseline_config = RawArmConfig {
        is_baseline: Some(true),
        ..RawArmConfig::default()
    };
    let lesson_config = RawArmConfig {
        is_baseline: Some(false),
        retrieval: true,
        ..RawArmConfig::default()
    };
    let baseline = run_uplift_arm(
        "baseline",
        false,
        set,
        baseline_driver,
        trials,
        &baseline_config,
    )?;
    let lessons = run_uplift_arm("lessons", true, set, lesson_driver, trials, &lesson_config)?;

    let baseline_injection = assert_injection(&baseline, &[]).map_err(|e| e.to_string())?;
    let lesson_injection =
        assert_injection(&lessons, intended_lesson_ids).map_err(|e| e.to_string())?;

    let baseline_agg = aggregate(&baseline).map_err(|e| e.to_string())?;
    let lesson_agg = aggregate(&lessons).map_err(|e| e.to_string())?;
    let uplift = significance(&baseline_agg, &lesson_agg, SIGNIFICANCE_FLOOR);

    Ok(UpliftReport {
        schema: 1,
        task_set: set.name.clone(),
        model: model.to_string(),
        trials,
        arms: vec![
            UpliftArmRow {
                aggregate: baseline_agg,
                injection: baseline_injection,
            },
            UpliftArmRow {
                aggregate: lesson_agg,
                injection: lesson_injection,
            },
        ],
        uplift,
    })
}

/// Render an uplift report as Markdown: per-arm mean ± stddev with the
/// injection audit, and the significance verdict — never a bare delta.
#[must_use]
pub fn render_uplift_report(report: &UpliftReport) -> String {
    let mut lines = vec![
        format!(
            "# Lesson uplift — {} (model: {}, trials: {})",
            report.task_set, report.model, report.trials
        ),
        String::new(),
        "| arm | mean | stddev | per-trial | injection |".to_string(),
        "|---|---|---|---|---|".to_string(),
    ];
    for arm in &report.arms {
        let injection = if arm.aggregate.is_lesson_arm {
            format!(
                "injected {}/{} intended",
                arm.injection.injected.len(),
                arm.injection.intended.len()
            )
        } else {
            "none (baseline)".to_string()
        };
        let per_trial = arm
            .aggregate
            .per_trial_success_rate
            .iter()
            .map(|r| format!("{:.0}%", r * 100.0))
            .collect::<Vec<_>>()
            .join(" ");
        lines.push(format!(
            "| {} | {:.0}% | {:.3} | {} | {} |",
            arm.aggregate.arm,
            arm.aggregate.mean * 100.0,
            arm.aggregate.stddev,
            per_trial,
            injection
        ));
    }
    let effect = report
        .uplift
        .effect_size
        .map_or_else(|| "n/a (zero variance)".to_string(), |e| format!("{e:.2}"));
    lines.push(String::new());
    lines.push(format!(
        "**Uplift:** delta={:.0}% (band ±{:.0}%, effect size {effect}) -> **{}**",
        report.uplift.delta * 100.0,
        report.uplift.band * 100.0,
        match report.uplift.verdict {
            localbench_scoring::uplift::Verdict::Uplift => "uplift",
            localbench_scoring::uplift::Verdict::Regression => "regression",
            localbench_scoring::uplift::Verdict::NoEffect => "no-effect (within noise)",
        }
    ));
    lines.join("\n")
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    const TASK_SET: &str = r#"{
        "schema": 1,
        "name": "headroom-v1",
        "tasks": [
            {
                "id": "migrate",
                "prompt": "How do I migrate the foo database?",
                "expect": { "mode": "substring", "value": "foo db sync" },
                "lesson_ids": ["lesson-migrate"]
            },
            {
                "id": "port",
                "prompt": "Which port does the bar daemon use?",
                "expect": { "mode": "regex", "value": "\\b7443\\b" },
                "lesson_ids": ["lesson-port"]
            }
        ],
        "lessons": [
            { "id": "lesson-migrate", "body": "Use foo db sync.", "tags": ["foo"] },
            { "id": "lesson-port", "body": "bar listens on 7443.", "category": "Environment", "confidence": 0.8 }
        ]
    }"#;

    fn task_set() -> TaskSet {
        serde_json::from_str(TASK_SET).unwrap()
    }

    struct ScriptedDriver {
        /// answer per task id, plus the memory ids each turn records.
        answers: Vec<(&'static str, &'static str, Vec<&'static str>)>,
    }

    impl UpliftDriver for ScriptedDriver {
        fn turn(&mut self, task: &UpliftTask, _trial: u32) -> Result<Turn, String> {
            let (_, answer, memories) = self
                .answers
                .iter()
                .find(|(id, _, _)| *id == task.id)
                .ok_or("unscripted task")?;
            Ok(Turn {
                answer: (*answer).to_string(),
                memories_used: memories
                    .iter()
                    .map(|id| MemoryUsed {
                        id: (*id).to_string(),
                    })
                    .collect(),
            })
        }
    }

    #[test]
    fn the_shipping_headroom_task_set_parses() {
        let raw = include_str!("../../../data/uplift/headroom-tasks-v1.json");
        let set: TaskSet = serde_json::from_str(raw).unwrap();
        assert_eq!(set.name, "headroom-project-conventions-v1");
        assert!(set.tasks.len() >= 5);
        // Every task's lesson_ids resolve to a declared seed lesson, so the
        // injection assertion always has something to verify against.
        for task in &set.tasks {
            assert!(
                !task.lesson_ids.is_empty(),
                "task {} has no lessons",
                task.id
            );
            for id in &task.lesson_ids {
                assert!(
                    set.lessons.iter().any(|lesson| &lesson.id == id),
                    "task {} names an undeclared lesson {id}",
                    task.id
                );
            }
        }
    }

    #[test]
    fn task_sets_load_and_fail_loud() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("set.json");
        std::fs::write(&path, TASK_SET).unwrap();
        let set = load_task_set(&path).unwrap();
        assert_eq!(set.tasks.len(), 2);
        assert_eq!(set.lessons[0].category, "ProjectConvention");
        assert!((set.lessons[0].confidence - 0.9).abs() < 1e-9);
        assert_eq!(set.lessons[1].category, "Environment");

        std::fs::write(&path, r#"{"schema":1,"name":"empty","tasks":[]}"#).unwrap();
        assert!(load_task_set(&path).unwrap_err().contains("no tasks"));
    }

    #[test]
    fn the_seed_pack_projects_the_localpilot_shape() {
        let pack = seed_pack(&task_set());
        let lessons = pack["lessons"].as_array().unwrap();
        assert_eq!(lessons.len(), 2);
        assert_eq!(lessons[0]["body"], "Use foo db sync.");
        assert_eq!(lessons[0]["category"], "ProjectConvention");
        assert_eq!(lessons[1]["confidence"], 0.8);
        // The engine's id is never recomputed here — ids stay out of the pack.
        assert!(lessons[0].get("id").is_none());
    }

    #[test]
    fn the_session_log_audit_reads_the_last_memories_used_event() {
        let log = r#"{"kind":{"type":"turn_started"}}
{"kind":{"type":"memories_used","memories":[{"id":"old-1","score":3,"layer":"project"}]}}
not json at all
{"kind":{"type":"memories_used","memories":[{"id":"m-1","score":5,"layer":"global"},{"id":"m-2","score":2,"layer":"project"}]}}
{"kind":{"type":"turn_done"}}"#;
        let memories = memories_from_session_log(log);
        assert_eq!(
            memories.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(),
            ["m-1", "m-2"],
            "the LAST audit event wins"
        );
        assert!(memories_from_session_log("").is_empty());
    }

    #[test]
    fn the_newest_session_log_wins() {
        let dir = tempfile::tempdir().unwrap();
        let sessions = dir.path().join(".localpilot").join("sessions");
        std::fs::create_dir_all(&sessions).unwrap();
        std::fs::write(sessions.join("old.jsonl"), "{}").unwrap();
        let newer = sessions.join("new.jsonl");
        std::fs::write(&newer, "{}").unwrap();
        let old_time = std::time::SystemTime::now() - Duration::from_secs(3600);
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(sessions.join("old.jsonl"))
            .unwrap();
        file.set_times(std::fs::FileTimes::new().set_modified(old_time))
            .unwrap();
        drop(file);
        assert_eq!(latest_session_log(dir.path()), Some(newer));
        assert_eq!(latest_session_log(&dir.path().join("missing")), None);
    }

    #[test]
    fn a_full_ab_run_produces_the_report_and_verdict() {
        let set = task_set();
        // Baseline fails both tasks and injects nothing.
        let mut baseline = ScriptedDriver {
            answers: vec![
                ("migrate", "Try foo migrate maybe?", vec![]),
                ("port", "No idea.", vec![]),
            ],
        };
        // The lesson arm answers correctly and records its intended lessons.
        let mut lessons = ScriptedDriver {
            answers: vec![
                ("migrate", "Run foo db sync.", vec!["lesson-migrate"]),
                ("port", "It uses 7443.", vec!["lesson-port"]),
            ],
        };
        let intended = vec!["lesson-migrate".to_string(), "lesson-port".to_string()];
        let report = run_uplift(&set, &mut baseline, &mut lessons, &intended, 3, "apex").unwrap();
        assert_eq!(report.schema, 1);
        assert_eq!(report.arms[0].aggregate.mean, 0.0);
        assert_eq!(report.arms[1].aggregate.mean, 1.0);
        assert_eq!(
            report.uplift.verdict,
            localbench_scoring::uplift::Verdict::Uplift
        );
        let rendered = render_uplift_report(&report);
        assert!(rendered.contains("**uplift**"));
        assert!(rendered.contains("none (baseline)"));
        assert!(rendered.contains("injected 2/2 intended"));
    }

    #[test]
    fn a_baseline_that_injects_voids_the_result() {
        let set = task_set();
        let mut baseline = ScriptedDriver {
            answers: vec![
                ("migrate", "Run foo db sync.", vec!["lesson-migrate"]),
                ("port", "It uses 7443.", vec![]),
            ],
        };
        let mut lessons = ScriptedDriver {
            answers: vec![
                ("migrate", "Run foo db sync.", vec!["lesson-migrate"]),
                ("port", "It uses 7443.", vec!["lesson-port"]),
            ],
        };
        let intended = vec!["lesson-migrate".to_string()];
        let err = run_uplift(&set, &mut baseline, &mut lessons, &intended, 2, "apex").unwrap_err();
        assert!(err.contains("VOID"));
        assert!(err.contains("baseline"));
    }

    #[test]
    fn a_lesson_arm_that_never_injects_voids_the_result() {
        let set = task_set();
        let mut baseline = ScriptedDriver {
            answers: vec![("migrate", "?", vec![]), ("port", "?", vec![])],
        };
        let mut lessons = ScriptedDriver {
            answers: vec![
                ("migrate", "Run foo db sync.", vec![]),
                ("port", "It uses 7443.", vec![]),
            ],
        };
        let intended = vec!["lesson-migrate".to_string()];
        let err = run_uplift(&set, &mut baseline, &mut lessons, &intended, 2, "apex").unwrap_err();
        assert!(err.contains("VOID"));
        assert!(err.contains("lessons"));
    }
}
