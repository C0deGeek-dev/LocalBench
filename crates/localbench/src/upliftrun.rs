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

use localx_eval_core::uplift::{
    text_digest, ArmIdentity, ArmRunIdentity, InjectionIdentity, InjectionMode, TaskSetIdentity,
    UpliftIdentity, UPLIFT_ARM_SCHEMA, UPLIFT_RECEIPT_SCHEMA,
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

/// The stop reasons with which a `localpilot print` turn ends without having
/// answered: the provider failed or was marked degraded, the turn was
/// cancelled, timed out, or shut down. A turn that answered (`Done`) or that
/// the model itself ran into the ground (`BudgetExceeded`, `NoProgress`) is
/// graded like any other answer.
pub const NON_ANSWER_STOPS: &[&str] = &[
    "ProviderError",
    "Degraded",
    "Cancelled",
    "TimedOut",
    "Quiesced",
];

/// The stop reason from the last `handoff:` line `localpilot print` writes to
/// stderr, when it is one of [`NON_ANSWER_STOPS`]. `None` for an answered turn,
/// and for output with no readable handoff (an older solver).
#[must_use]
pub fn turn_stopped_without_answer(stderr: &str) -> Option<String> {
    let handoff = stderr
        .lines()
        .rev()
        .find_map(|line| line.trim().strip_prefix("handoff:"))?;
    let value: serde_json::Value = serde_json::from_str(handoff.trim()).ok()?;
    let stop = value["stop"].as_str()?;
    NON_ANSWER_STOPS.contains(&stop).then(|| stop.to_string())
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
    pub answer_only: bool,
    pub bin: String,
    pub workspace: PathBuf,
    pub model: String,
    pub timeout: Duration,
}

impl UpliftDriver for PrintDriver {
    fn turn(&mut self, task: &UpliftTask, _trial: u32) -> Result<Turn, String> {
        let mut args = vec![
            "print".to_string(),
            task.prompt.clone(),
            "--model".to_string(),
            self.model.clone(),
        ];
        if self.answer_only {
            args.push("--answer-only".to_string());
        }
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
        // A turn that stopped without an answer exits 0 and leaves partial text
        // on stdout. Grading that as a miss would count an infrastructure
        // failure as evidence about the lesson, so it fails the arm instead.
        if let Some(stop) = turn_stopped_without_answer(&run.stderr) {
            return Err(format!(
                "'{} print' stopped without an answer (task '{}'): the turn ended with {stop}, \
                 which says nothing about the lesson",
                self.bin, task.id
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
    /// Whether turns answered from adjacent context without tools.
    #[serde(default)]
    pub answer_only: bool,
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
        answer_only: false,
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
        format!(
            "Solver mode: {}",
            if report.answer_only {
                "answer-only, project context beside the question, no tools"
            } else {
                "coding-agent, system context, tools available"
            }
        ),
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

// --- Per-arm runs and the combined, identity-bound receipt ------------------
//
// `run_uplift` runs both arms back to back, which leaves no moment to change
// the workspace's memory between them. A caller that stages memory per arm
// runs each arm by itself (`run_arm_file`), then joins the two arm files
// (`combine`). The statistics and the injection contract are the same ones
// `run_uplift` uses.

/// The memory configuration an arm's workspace must be staged with.
#[must_use]
pub fn arm_config(lesson_arm: bool) -> String {
    if lesson_arm {
        localbench_measure::arms::localmind_lesson_arm_config()
    } else {
        localbench_measure::arms::localmind_measurement_config()
    }
}

/// Bind answer-only presentation and tool availability into configuration
/// identity. Legacy coding-agent digests remain byte-for-byte unchanged.
#[must_use]
pub fn solver_config_digest(memory_digest: &str, answer_only: bool) -> String {
    if answer_only {
        text_digest(format!("answer-only-context-v1:{memory_digest}").as_bytes())
    } else {
        memory_digest.to_string()
    }
}

/// The seed pack as the exact text `--emit-seed-pack` prints, so its digest is
/// the same for whoever stages it and whoever attests it.
///
/// # Errors
/// A serialization failure.
pub fn seed_pack_text(set: &TaskSet) -> Result<String, String> {
    serde_json::to_string_pretty(&seed_pack(set)).map_err(|e| e.to_string())
}

/// A task set by the bytes of its file: whoever wrote the file can compute the
/// same digest without knowing the format.
#[must_use]
pub fn task_set_identity(set: &TaskSet, file_bytes: &[u8]) -> TaskSetIdentity {
    TaskSetIdentity {
        name: set.name.clone(),
        digest: text_digest(file_bytes),
        task_count: set.tasks.len(),
    }
}

/// Check the workspace is staged for this arm: its `.localmind.toml` must be
/// exactly the arm's configuration. Returns the configuration's digest.
///
/// # Errors
/// A mis-staging refusal naming what was found.
pub fn assert_staged(workspace: &Path, lesson_arm: bool) -> Result<String, String> {
    let expected = arm_config(lesson_arm);
    let path = workspace.join(".localmind.toml");
    let found = std::fs::read_to_string(&path).map_err(|e| {
        format!(
            "mis-staged {} arm: {} cannot be read ({e}). Stage the arm's memory \
             configuration first (uplift --emit-arm-config).",
            arm_name(lesson_arm),
            path.display()
        )
    })?;
    if found.replace("\r\n", "\n") != expected {
        return Err(format!(
            "mis-staged {} arm: {} is not the arm's configuration. Refusing to run: \
             the arms may differ only in the seeded lesson.",
            arm_name(lesson_arm),
            path.display()
        ));
    }
    Ok(text_digest(expected.as_bytes()))
}

fn arm_name(lesson_arm: bool) -> &'static str {
    if lesson_arm {
        "lessons"
    } else {
        "baseline"
    }
}

/// What one arm run is asked to do.
#[derive(Debug, Clone)]
pub struct ArmRequest<'a> {
    pub answer_only: bool,
    pub set: &'a TaskSet,
    pub task_set: TaskSetIdentity,
    pub lesson_arm: bool,
    /// The requester's binding for the whole run, carried into the receipt.
    pub binding: String,
    pub model: String,
    pub trials: u32,
    pub timeout_secs: u64,
    /// The memory ids the lesson arm must show it used. Ignored for a baseline.
    pub intended: Vec<String>,
}

/// One arm's result, with the identity it ran under.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArmFile {
    #[serde(default)]
    pub answer_only: bool,
    pub schema: String,
    pub identity: ArmRunIdentity,
    pub result: ArmResult,
}

/// Run one arm in a workspace already staged for it.
///
/// # Errors
/// The grader self-test, a mis-staged workspace, a contaminated baseline
/// configuration, or the driver's failure.
pub fn run_arm_file(
    request: &ArmRequest<'_>,
    workspace: &Path,
    driver: &mut dyn UpliftDriver,
) -> Result<ArmFile, String> {
    assert_grader_selftest()?;
    let config_digest = solver_config_digest(
        &assert_staged(workspace, request.lesson_arm)?,
        request.answer_only,
    );
    let config = RawArmConfig {
        is_baseline: Some(!request.lesson_arm),
        retrieval: request.lesson_arm,
        ..RawArmConfig::default()
    };
    let arm = arm_name(request.lesson_arm);
    let injection = if request.lesson_arm {
        InjectionIdentity::lessons(
            InjectionMode::Retrieved,
            request.intended.clone(),
            text_digest(seed_pack_text(request.set)?.as_bytes()),
        )
    } else {
        InjectionIdentity::none()
    };
    let result = run_uplift_arm(
        arm,
        request.lesson_arm,
        request.set,
        driver,
        request.trials,
        &config,
    )?;
    Ok(ArmFile {
        answer_only: request.answer_only,
        schema: UPLIFT_ARM_SCHEMA.to_string(),
        identity: ArmRunIdentity {
            binding: request.binding.clone(),
            task_set: request.task_set.clone(),
            arm: ArmIdentity {
                arm: arm.to_string(),
                is_lesson_arm: request.lesson_arm,
                config_digest,
                model: request.model.clone(),
                trials: request.trials,
                timeout_secs: request.timeout_secs,
                injection,
            },
        },
        result,
    })
}

/// One arm's row in the combined receipt. `injection` is absent for the arm
/// that broke the injection contract.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReceiptArm {
    #[serde(flatten)]
    pub aggregate: Aggregate,
    pub injection: Option<InjectionSummary>,
}

/// The combined receipt (`localbench-uplift-v2`): the v1 report's numbers,
/// bound to the content identity of the run that produced them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UpliftReceipt {
    #[serde(default)]
    pub answer_only: bool,
    pub schema: String,
    /// Digest of `identity`.
    pub run_id: String,
    pub identity: UpliftIdentity,
    pub arms: Vec<ReceiptArm>,
    /// The significance signal. Absent when the run is void.
    pub uplift: Option<Significance>,
    /// Why the run is void: an arm did not inject as configured. A void run
    /// reports no uplift number — it is not "no effect".
    pub void: Option<String>,
}

/// Join two arm files into the receipt: the injection contract, aggregation
/// and significance, over arms proven to be one pair.
///
/// # Errors
/// The files are not a pair of the same request, carry the wrong schema, or
/// an arm has no tasks. A broken injection contract is not an error: it is a
/// void receipt.
pub fn combine(first: &ArmFile, second: &ArmFile) -> Result<UpliftReceipt, String> {
    if first.answer_only != second.answer_only {
        return Err("the arm files use different solver modes".to_string());
    }
    for file in [first, second] {
        let memory_digest = text_digest(arm_config(file.identity.arm.is_lesson_arm).as_bytes());
        if file.identity.arm.config_digest != solver_config_digest(&memory_digest, file.answer_only)
        {
            return Err(
                "the arm file's configuration digest does not match its solver mode".to_string(),
            );
        }
        if file.schema != UPLIFT_ARM_SCHEMA {
            return Err(format!(
                "unsupported arm file schema '{}' (expected {UPLIFT_ARM_SCHEMA})",
                file.schema
            ));
        }
        if file.result.is_lesson_arm != file.identity.arm.is_lesson_arm
            || file.result.trials != file.identity.arm.trials
            || file.result.tasks.len() != file.identity.task_set.task_count
        {
            return Err(format!(
                "arm file '{}' does not match its own identity",
                file.identity.arm.arm
            ));
        }
    }
    let identity = UpliftIdentity::pair(&first.identity, &second.identity).map_err(|problems| {
        format!(
            "the arm files are not one pair: {}",
            problems
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("; ")
        )
    })?;
    let (baseline, lessons) = if first.result.is_lesson_arm {
        (&second.result, &first.result)
    } else {
        (&first.result, &second.result)
    };

    let baseline_agg = aggregate(baseline).map_err(|e| e.to_string())?;
    let lesson_agg = aggregate(lessons).map_err(|e| e.to_string())?;
    let baseline_injection = assert_injection(baseline, &[]);
    let lesson_injection = assert_injection(lessons, &identity.lessons.injection.intended);
    let void = [&baseline_injection, &lesson_injection]
        .iter()
        .filter_map(|result| result.as_ref().err().map(ToString::to_string))
        .collect::<Vec<_>>();
    let uplift = void
        .is_empty()
        .then(|| significance(&baseline_agg, &lesson_agg, SIGNIFICANCE_FLOOR));
    Ok(UpliftReceipt {
        answer_only: first.answer_only,
        schema: UPLIFT_RECEIPT_SCHEMA.to_string(),
        run_id: identity.run_id(),
        identity,
        arms: vec![
            ReceiptArm {
                aggregate: baseline_agg,
                injection: baseline_injection.ok(),
            },
            ReceiptArm {
                aggregate: lesson_agg,
                injection: lesson_injection.ok(),
            },
        ],
        uplift,
        void: (!void.is_empty()).then(|| void.join(" ")),
    })
}

/// Render a combined receipt as Markdown: what it is bound to, then either
/// the v1 report's table and verdict or, for a void run, why no number exists.
#[must_use]
pub fn render_uplift_receipt(receipt: &UpliftReceipt) -> String {
    let identity = &receipt.identity;
    let mut lines = vec![
        format!("Run {} (binding {})", receipt.run_id, identity.binding),
        format!(
            "Task set {} — {} ({} tasks)",
            identity.task_set.name, identity.task_set.digest, identity.task_set.task_count
        ),
        String::new(),
    ];
    match (&receipt.uplift, &receipt.void) {
        (Some(uplift), None) => {
            let arms = receipt
                .arms
                .iter()
                .filter_map(|arm| {
                    Some(UpliftArmRow {
                        aggregate: arm.aggregate.clone(),
                        injection: arm.injection.clone()?,
                    })
                })
                .collect();
            lines.push(render_uplift_report(&UpliftReport {
                answer_only: receipt.answer_only,
                schema: 1,
                task_set: identity.task_set.name.clone(),
                model: identity.lessons.model.clone(),
                trials: identity.lessons.trials,
                arms,
                uplift: uplift.clone(),
            }));
        }
        (_, void) => lines.push(format!(
            "**VOID** — no uplift number is reported. {}",
            void.as_deref().unwrap_or("The receipt carries no result.")
        )),
    }
    lines.join("\n")
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    #[test]
    fn a_turn_that_ended_without_an_answer_is_recognised_from_its_handoff() {
        let handoff = |stop: &str| {
            format!(
                "warning: something\nhandoff: {{\"files_changed\":[],\"stop\":\"{stop}\",\"tool_calls\":0}}\n"
            )
        };
        for stop in NON_ANSWER_STOPS {
            assert_eq!(
                turn_stopped_without_answer(&handoff(stop)).as_deref(),
                Some(*stop)
            );
        }
        for answered in ["Done", "BudgetExceeded", "NoProgress"] {
            assert_eq!(
                turn_stopped_without_answer(&handoff(answered)),
                None,
                "{answered}"
            );
        }
        // No handoff, or one that does not parse: graded as before.
        assert_eq!(turn_stopped_without_answer("just some stderr\n"), None);
        assert_eq!(turn_stopped_without_answer("handoff: not json\n"), None);
        // The last handoff wins.
        let both = format!("{}{}", handoff("ProviderError"), handoff("Done"));
        assert_eq!(turn_stopped_without_answer(&both), None);
    }

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

    // --- per-arm runs and the combined receipt ------------------------------

    fn staged(lesson_arm: bool) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(".localmind.toml"), arm_config(lesson_arm)).unwrap();
        dir
    }

    fn request(set: &TaskSet, lesson_arm: bool) -> ArmRequest<'_> {
        ArmRequest {
            answer_only: false,
            set,
            task_set: task_set_identity(set, TASK_SET.as_bytes()),
            lesson_arm,
            binding: "bind-1".to_string(),
            model: "apex".to_string(),
            trials: 3,
            timeout_secs: 600,
            intended: vec!["mem-migrate".to_string(), "mem-port".to_string()],
        }
    }

    fn failing_baseline() -> ScriptedDriver {
        ScriptedDriver {
            answers: vec![
                ("migrate", "Try foo migrate maybe?", vec![]),
                ("port", "No idea.", vec![]),
            ],
        }
    }

    fn passing_lessons() -> ScriptedDriver {
        ScriptedDriver {
            answers: vec![
                ("migrate", "Run foo db sync.", vec!["mem-migrate"]),
                ("port", "It uses 7443.", vec!["mem-port"]),
            ],
        }
    }

    fn arm(set: &TaskSet, lesson_arm: bool, driver: &mut ScriptedDriver) -> ArmFile {
        let workspace = staged(lesson_arm);
        run_arm_file(&request(set, lesson_arm), workspace.path(), driver).unwrap()
    }

    #[test]
    fn each_arm_runs_alone_and_the_pair_combines_into_a_bound_receipt() {
        let set = task_set();
        let baseline = arm(&set, false, &mut failing_baseline());
        let lessons = arm(&set, true, &mut passing_lessons());
        assert_eq!(baseline.schema, UPLIFT_ARM_SCHEMA);
        assert_eq!(
            baseline.identity.arm.config_digest,
            text_digest(arm_config(false).as_bytes())
        );
        assert_eq!(
            lessons.identity.arm.injection.seed_pack_digest,
            Some(text_digest(seed_pack_text(&set).unwrap().as_bytes()))
        );

        // An arm file survives the disk, and the pair combines in either order.
        let reread: ArmFile =
            serde_json::from_str(&serde_json::to_string(&lessons).unwrap()).unwrap();
        let receipt = combine(&reread, &baseline).unwrap();
        assert_eq!(receipt, combine(&baseline, &lessons).unwrap());

        assert_eq!(receipt.schema, UPLIFT_RECEIPT_SCHEMA);
        assert_eq!(receipt.run_id, receipt.identity.run_id());
        assert_eq!(receipt.identity.binding, "bind-1");
        assert_eq!(
            receipt.identity.task_set.digest,
            text_digest(TASK_SET.as_bytes())
        );
        assert_eq!(receipt.void, None);
        assert_eq!(
            receipt.uplift.as_ref().unwrap().verdict,
            localbench_scoring::uplift::Verdict::Uplift
        );
        assert_eq!(
            receipt.arms[1].injection.as_ref().unwrap().injected,
            ["mem-migrate", "mem-port"]
        );
    }

    #[test]
    fn the_four_outcomes_stay_distinct() {
        let set = task_set();
        let verdict = |baseline: &mut ScriptedDriver, lessons: &mut ScriptedDriver| {
            let receipt = combine(&arm(&set, false, baseline), &arm(&set, true, lessons)).unwrap();
            assert_eq!(receipt.void, None);
            receipt.uplift.unwrap().verdict
        };
        let passing_baseline = || ScriptedDriver {
            answers: vec![
                ("migrate", "Run foo db sync.", vec![]),
                ("port", "It uses 7443.", vec![]),
            ],
        };
        let failing_lessons = || ScriptedDriver {
            answers: vec![
                ("migrate", "Try foo migrate.", vec!["mem-migrate"]),
                ("port", "No idea.", vec!["mem-port"]),
            ],
        };
        use localbench_scoring::uplift::Verdict;
        // Control fails, treatment passes.
        assert_eq!(
            verdict(&mut failing_baseline(), &mut passing_lessons()),
            Verdict::Uplift
        );
        // Both pass, and both fail: no demonstrated effect — not void.
        assert_eq!(
            verdict(&mut passing_baseline(), &mut passing_lessons()),
            Verdict::NoEffect
        );
        assert_eq!(
            verdict(&mut failing_baseline(), &mut failing_lessons()),
            Verdict::NoEffect
        );
        // The lesson made it worse.
        assert_eq!(
            verdict(&mut passing_baseline(), &mut failing_lessons()),
            Verdict::Regression
        );
    }

    #[test]
    fn a_broken_injection_contract_is_a_void_receipt_with_no_number() {
        let set = task_set();
        // The control saw the lesson.
        let mut contaminated = ScriptedDriver {
            answers: vec![
                ("migrate", "Run foo db sync.", vec!["mem-migrate"]),
                ("port", "It uses 7443.", vec![]),
            ],
        };
        let receipt = combine(
            &arm(&set, false, &mut contaminated),
            &arm(&set, true, &mut passing_lessons()),
        )
        .unwrap();
        assert_eq!(receipt.uplift, None, "a void run reports no number");
        assert!(receipt.void.as_ref().unwrap().contains("baseline"));
        assert_eq!(receipt.arms[0].injection, None);
        assert!(receipt.arms[1].injection.is_some());

        // The treatment never got the intended lesson — it got another one.
        let mut wrong_lesson = ScriptedDriver {
            answers: vec![
                ("migrate", "Run foo db sync.", vec!["mem-other"]),
                ("port", "It uses 7443.", vec!["mem-other"]),
            ],
        };
        let receipt = combine(
            &arm(&set, false, &mut failing_baseline()),
            &arm(&set, true, &mut wrong_lesson),
        )
        .unwrap();
        assert_eq!(receipt.uplift, None);
        assert!(receipt.void.as_ref().unwrap().contains("lessons"));

        // Or got nothing at all.
        let mut nothing = ScriptedDriver {
            answers: vec![
                ("migrate", "Run foo db sync.", vec![]),
                ("port", "It uses 7443.", vec![]),
            ],
        };
        let receipt = combine(
            &arm(&set, false, &mut failing_baseline()),
            &arm(&set, true, &mut nothing),
        )
        .unwrap();
        assert!(receipt.void.is_some() && receipt.uplift.is_none());
    }

    #[test]
    fn a_mis_staged_workspace_is_refused_before_any_turn() {
        let set = task_set();
        struct Unreachable;
        impl UpliftDriver for Unreachable {
            fn turn(&mut self, _task: &UpliftTask, _trial: u32) -> Result<Turn, String> {
                panic!("a mis-staged arm must not spend a turn")
            }
        }
        // The baseline's workspace still has the lesson arm's configuration.
        let wrong = staged(true);
        let err = run_arm_file(&request(&set, false), wrong.path(), &mut Unreachable).unwrap_err();
        assert!(err.contains("mis-staged baseline arm"), "{err}");
        // No configuration at all.
        let empty = tempfile::tempdir().unwrap();
        let err = run_arm_file(&request(&set, true), empty.path(), &mut Unreachable).unwrap_err();
        assert!(err.contains("mis-staged lessons arm"), "{err}");
        // CRLF line endings are the same configuration.
        let crlf = tempfile::tempdir().unwrap();
        std::fs::write(
            crlf.path().join(".localmind.toml"),
            arm_config(false).replace('\n', "\r\n"),
        )
        .unwrap();
        assert!(assert_staged(crlf.path(), false).is_ok());
    }

    #[test]
    fn only_a_true_pair_combines() {
        let set = task_set();
        let baseline = arm(&set, false, &mut failing_baseline());
        let lessons = arm(&set, true, &mut passing_lessons());

        // Half a pair, twice.
        let err = combine(&baseline, &baseline).unwrap_err();
        assert!(
            err.contains("not one baseline arm and one lesson arm"),
            "{err}"
        );

        // Arms of different requests, task sets or settings.
        let mut other = lessons.clone();
        other.identity.binding = "bind-2".to_string();
        assert!(combine(&baseline, &other).unwrap_err().contains("binding"));
        let mut other = lessons.clone();
        other.identity.task_set.digest = "sha256:other".to_string();
        assert!(combine(&baseline, &other).unwrap_err().contains("task set"));
        let mut other = lessons.clone();
        other.identity.arm.model = "another".to_string();
        assert!(combine(&baseline, &other).unwrap_err().contains("model"));

        // A result swapped under another arm's identity.
        let mut forged = lessons.clone();
        forged.result = baseline.result.clone();
        assert!(combine(&baseline, &forged)
            .unwrap_err()
            .contains("does not match its own identity"));

        // An unknown arm-file schema.
        let mut old = lessons;
        old.schema = "localbench-uplift-arm-v0".to_string();
        assert!(combine(&baseline, &old)
            .unwrap_err()
            .contains("unsupported"));
    }

    #[test]
    fn the_arm_configurations_differ_only_in_learning() {
        assert_eq!(arm_config(false), "[learning]\nenabled = false\n");
        let lessons = arm_config(true);
        assert!(lessons.contains("enabled = true"));
        assert!(lessons.contains("allowed_scopes = [\"project\"]"));
        assert!(
            !lessons.contains("global") && !lessons.contains("inference"),
            "no machine-wide memory and no model-backed extraction: {lessons}"
        );
    }
}
