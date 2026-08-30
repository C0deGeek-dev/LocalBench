//! The live arm-matrix: drive the solver-under-test across every harness arm ×
//! task cell, grade each cell in a network-isolated container, persist cells,
//! and assemble the comparative capability report.
//!
//! Guarantees carried from the measurement contract: the instrument self-test
//! gate runs before any spend; a contaminated baseline is refused at setup; a
//! failing or hung cell is isolated (recorded unsolved, never aborts the
//! matrix); a wall-clock timeout stays a caveat, not a capability result; and
//! a wedged Docker engine trips the circuit breaker so the run yields with its
//! cell ledger intact instead of burning hours.

use std::collections::BTreeSet;
use std::path::Path;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use localbench_measure::arms::{assert_arm_isolation, RawArmConfig};
use localbench_measure::container::{
    cargo_warm_plan, container_grade_plan, docker_cleanup_command, docker_healthcheck_plan,
    DockerWedgeBreaker, GradeTask, GraderPlan,
};
use localbench_measure::grade::{
    cargo_warm_manifest, is_offline_fetch_error, rust_cargo_cache_deps, test_count,
};
use localbench_measure::runner::{
    annotate_cell_note, assert_external_corpus_path, assert_instruments_ready, capability_report,
    infra_gap_caveat, instrument_checks, parse_capability_scorecard, save_cell, timeout_caveat,
    ArmCards, CapabilityReport, Cell, ParsedCard,
};

use crate::output::RunEvent;
use crate::solver::{run_bounded, SolveSpec, Solver};

/// One arm of the matrix.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArmSpec {
    pub arm: String,
    /// The arm's effective config, when the runner tracks isolation (a
    /// baseline arm should always carry one so its isolation is checked).
    #[serde(default)]
    pub config: Option<RawArmConfig>,
    /// Enable the verify-before-done gate for this arm's runs.
    #[serde(default)]
    pub verify: bool,
    /// Close each run out into review-gated memory (the warm/teaching arm).
    #[serde(default)]
    pub learn: bool,
    /// Path to a coach script: the arm's runs are driven through the
    /// solver-under-test's MCP serve surface by a deterministic scripted
    /// coach instead of a headless eval, and the cell records the
    /// interventions the script made.
    #[serde(default)]
    pub coach: Option<String>,
}

/// One materialized task.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskSpec {
    pub id: String,
    /// The task workspace on the host (outside any LocalPilot checkout).
    pub workspace: String,
    /// The problem statement handed to the solver.
    pub problem: String,
    /// The benchmark's own test command, run inside the grade container.
    pub test_command: String,
    /// Container image override (defaults to the per-task convention).
    #[serde(default)]
    pub image: Option<String>,
    /// Grade language label for test counting (unlisted labels use the
    /// generic fail-closed counter).
    #[serde(default)]
    pub grade_label: Option<String>,
    /// Path to the shared cargo registry that LocalBench warms before a `rust`
    /// grade, then mounts read-only for the `--network=none` graded window.
    /// Absent for other languages.
    #[serde(default)]
    pub cargo_cache: Option<String>,
}

/// A full matrix run specification (the `arms --spec` file).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunSpec {
    pub schema: u32,
    pub model: String,
    /// `first-party` or `external` (default).
    #[serde(default = "default_corpus")]
    pub corpus: String,
    pub arms: Vec<ArmSpec>,
    pub tasks: Vec<TaskSpec>,
}

fn default_corpus() -> String {
    "external".to_string()
}

/// Load and validate a run spec, failing loud on anything unusable.
///
/// # Errors
/// A plain-language message naming the defect.
pub fn load_run_spec(path: &Path) -> Result<RunSpec, String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| format!("could not read {}: {e}", path.display()))?;
    let spec: RunSpec = serde_json::from_str(&raw)
        .map_err(|e| format!("{} does not parse: {e}", path.display()))?;
    if spec.schema != 1 {
        return Err(format!(
            "unsupported run-spec schema {} (expected 1)",
            spec.schema
        ));
    }
    if spec.model.trim().is_empty() {
        return Err("the run spec names no model".to_string());
    }
    if spec.corpus != "external" && spec.corpus != "first-party" {
        return Err(format!(
            "unknown corpus '{}' (use first-party|external)",
            spec.corpus
        ));
    }
    if spec.arms.is_empty() {
        return Err("at least one arm is required".to_string());
    }
    if spec.tasks.is_empty() {
        return Err("at least one task is required".to_string());
    }
    for arm in &spec.arms {
        if arm.arm.trim().is_empty() {
            return Err("each arm needs a non-empty name".to_string());
        }
    }
    for task in &spec.tasks {
        if task.id.trim().is_empty() || task.workspace.trim().is_empty() {
            return Err("each task needs an id and a workspace".to_string());
        }
        if task.grade_label.as_deref() == Some("rust")
            && !matches!(task.cargo_cache.as_deref(), Some(path) if !path.trim().is_empty())
        {
            return Err(format!(
                "rust task '{}' needs cargo_cache so dependencies can be warmed before grading",
                task.id
            ));
        }
    }
    Ok(spec)
}

/// One graded verdict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GradeVerdict {
    /// The benchmark's own pass/fail; `None` when grading was skipped.
    pub passed: Option<bool>,
    /// Whether the grade hit its wall-clock bound (a caveat, never a result).
    pub timed_out: bool,
    /// Set when the grade could not actually assess the solve because the grade
    /// ENVIRONMENT failed — currently an under-vendored offline cargo cache (a
    /// `--network=none` + `CARGO_NET_OFFLINE` rust grade whose deps were not
    /// pre-vendored). Such a cell is an infrastructure gap (re-warm the cache
    /// and rescore), never the solver's fault, so it counts unsolved as a floor
    /// with a caveat rather than being scored a solve failure that would
    /// silently deflate the arm's rate. Carries the caveat reason.
    pub infra_gap: Option<String>,
    /// A secondary failure while compensating for the exact container after a
    /// timeout. The primary timeout/grade outcome remains authoritative.
    pub cleanup_diagnostic: Option<String>,
    pub note: String,
}

impl GradeVerdict {
    /// The no-grading verdict: the solver's self-reported result stands.
    #[must_use]
    pub fn skipped() -> Self {
        Self {
            passed: None,
            timed_out: false,
            infra_gap: None,
            cleanup_diagnostic: None,
            note: String::new(),
        }
    }
}

/// Grades one task after a solve.
pub trait Grader {
    /// Grade a task; a timeout is reported, not scored.
    ///
    /// # Errors
    /// A message when grading infrastructure is unusable (wedged engine);
    /// the matrix then yields with its ledger intact.
    fn grade(&mut self, task: &TaskSpec) -> Result<GradeVerdict, String>;
}

/// A no-op grader: every verdict is [`GradeVerdict::skipped`].
pub struct NoGrader;

impl Grader for NoGrader {
    fn grade(&mut self, _task: &TaskSpec) -> Result<GradeVerdict, String> {
        Ok(GradeVerdict::skipped())
    }
}

/// The live container grader: pre-flight engine health, then per task run the
/// network-isolated read-only grade plan; the wedge breaker trips after
/// consecutive timeouts. The command executor is a seam so the whole flow is
/// testable without Docker.
pub struct ContainerGrader<E>
where
    E: FnMut(&[String], Duration) -> Result<(bool, String, bool), String>,
{
    /// Runs `(command, timeout)` → `(exit_ok, combined output, timed_out)`.
    pub exec: E,
    pub timeout: Duration,
    pub breaker: DockerWedgeBreaker,
    health_checked: bool,
    warmed_cargo_manifests: BTreeSet<(String, String)>,
}

struct ContainerExecution {
    exit_ok: bool,
    output: String,
    timed_out: bool,
    cleanup_diagnostic: Option<String>,
}

impl<E> ContainerGrader<E>
where
    E: FnMut(&[String], Duration) -> Result<(bool, String, bool), String>,
{
    /// A grader over an executor seam.
    pub fn new(exec: E, timeout: Duration) -> Self {
        Self {
            exec,
            timeout,
            breaker: DockerWedgeBreaker::default(),
            health_checked: false,
            warmed_cargo_manifests: BTreeSet::new(),
        }
    }

    fn cleanup_container(&mut self, container_name: &str) -> Result<(), String> {
        let command = docker_cleanup_command(container_name);
        let cleanup_timeout = self.timeout.min(Duration::from_secs(10));
        match (self.exec)(&command, cleanup_timeout) {
            Ok((true, _, false)) => Ok(()),
            // Docker uses a non-zero exit when the named container disappeared
            // between timeout detection and compensation. Absence is the
            // required postcondition, so that race is successful cleanup.
            Ok((false, output, false))
                if output.to_ascii_lowercase().contains("no such container") =>
            {
                Ok(())
            }
            Ok((_, output, true)) => Err(format!(
                "exact Docker cleanup for container '{container_name}' timed out after {}s{}",
                cleanup_timeout.as_secs(),
                output_suffix(&output)
            )),
            Ok((false, output, false)) => Err(format!(
                "exact Docker cleanup for container '{container_name}' failed{}",
                output_suffix(&output)
            )),
            Err(error) => Err(format!(
                "exact Docker cleanup for container '{container_name}' could not run: {error}"
            )),
        }
    }

    fn execute_plan(&mut self, plan: &GraderPlan) -> Result<ContainerExecution, String> {
        match (self.exec)(&plan.command, self.timeout) {
            Ok((exit_ok, output, timed_out)) => {
                let cleanup_diagnostic = if timed_out {
                    self.cleanup_container(&plan.container_name).err()
                } else {
                    None
                };
                Ok(ContainerExecution {
                    exit_ok,
                    output,
                    timed_out,
                    cleanup_diagnostic,
                })
            }
            Err(primary) => {
                // A spawn/wait error is ambiguous: Docker may have accepted
                // the create request even though the CLI did not report a
                // result. Compensate by the plan's exact, unique name.
                let cleanup = match self.cleanup_container(&plan.container_name) {
                    Ok(()) => format!(
                        "exact Docker cleanup completed for container '{}'",
                        plan.container_name
                    ),
                    Err(diagnostic) => diagnostic,
                };
                Err(format!("{primary}; {cleanup}"))
            }
        }
    }

    fn warm_cargo_cache(&mut self, task: &TaskSpec, cargo_cache: &str) -> Result<(), String> {
        let workspace = Path::new(&task.workspace);
        let corpus_root = workspace
            .ancestors()
            .find(|candidate| {
                candidate
                    .join("rust")
                    .join("exercises")
                    .join("practice")
                    .is_dir()
            })
            .ok_or_else(|| {
                format!(
                    "task '{}' is not inside a Rust corpus containing rust/exercises/practice; \
                     cannot prepare its offline Cargo cache",
                    task.id
                )
            })?;
        let deps = rust_cargo_cache_deps(corpus_root, true)
            .map_err(|error| format!("could not prepare cargo cache for '{}': {error}", task.id))?;
        let manifest = cargo_warm_manifest(&deps);
        let cache_key = (cargo_cache.to_string(), manifest.clone());
        if self.warmed_cargo_manifests.contains(&cache_key) {
            return Ok(());
        }

        let warm_dir = Path::new(cargo_cache).join(".localbench-warm");
        std::fs::create_dir_all(&warm_dir).map_err(|error| {
            format!(
                "could not create Cargo warm directory {}: {error}",
                warm_dir.display()
            )
        })?;
        // `cargo fetch` parses the manifest before it resolves anything, and a
        // package with no discoverable target is a parse error ("no targets
        // specified in the manifest"). The warm crate therefore needs a target
        // file on disk, not just dependencies — without it every Rust grade
        // fails in preparation instead of merely paying a cold fetch.
        let warm_src = warm_dir.join("src");
        std::fs::create_dir_all(&warm_src).map_err(|error| {
            format!(
                "could not create Cargo warm source directory {}: {error}",
                warm_src.display()
            )
        })?;
        let lib_path = warm_src.join("lib.rs");
        if !lib_path.is_file() {
            std::fs::write(&lib_path, "").map_err(|error| {
                format!(
                    "could not write Cargo warm crate target {}: {error}",
                    lib_path.display()
                )
            })?;
        }

        let manifest_path = warm_dir.join("Cargo.toml");
        let current = std::fs::read_to_string(&manifest_path).ok();
        if current.as_deref() != Some(manifest.as_str()) {
            std::fs::write(&manifest_path, &manifest).map_err(|error| {
                format!(
                    "could not write Cargo warm manifest {}: {error}",
                    manifest_path.display()
                )
            })?;
        }

        let grade_task = GradeTask {
            id: task.id.clone(),
            workspace: task.workspace.clone(),
            test_command: task.test_command.clone(),
        };
        let plan = cargo_warm_plan(&grade_task, task.image.as_deref(), cargo_cache);
        let execution = self.execute_plan(&plan)?;
        if !execution.exit_ok || execution.timed_out {
            let detail = if execution.timed_out {
                format!(
                    "timed out after {}s{}",
                    self.timeout.as_secs(),
                    execution
                        .cleanup_diagnostic
                        .as_deref()
                        .map_or_else(String::new, |diagnostic| format!("; {diagnostic}"))
                )
            } else {
                execution.output.trim().to_string()
            };
            return Err(format!(
                "Cargo cache preparation failed for task '{}': {detail}",
                task.id
            ));
        }
        self.warmed_cargo_manifests.insert(cache_key);
        Ok(())
    }
}

impl<E> Grader for ContainerGrader<E>
where
    E: FnMut(&[String], Duration) -> Result<(bool, String, bool), String>,
{
    fn grade(&mut self, task: &TaskSpec) -> Result<GradeVerdict, String> {
        if self.breaker.tripped() {
            return Err(format!(
                "docker looks wedged ({} consecutive grade timeouts) — yielding with the \
                 cell ledger intact; restart the engine and rescore",
                self.breaker.strikes()
            ));
        }
        // Prove the engine actually RUNS a container before spending — `docker
        // info` can answer while `docker run` hangs.
        if !self.health_checked {
            let health = docker_healthcheck_plan();
            let execution = self.execute_plan(&health)?;
            if !execution.exit_ok || execution.timed_out {
                let detail = if execution.timed_out {
                    format!(
                        "timed out after {}s{}",
                        self.timeout.as_secs(),
                        execution
                            .cleanup_diagnostic
                            .as_deref()
                            .map_or_else(String::new, |diagnostic| format!("; {diagnostic}"))
                    )
                } else {
                    execution.output.trim().to_string()
                };
                return Err(format!(
                    "docker failed its pre-flight health check (a container must run to \
                     completion): {}",
                    detail
                ));
            }
            self.health_checked = true;
        }

        // Fetch the complete corpus dependency union before the isolated grade
        // starts. One successful manifest/cache pair is remembered for the
        // matrix run, so grading the same task in another arm cannot re-fetch.
        let cargo_cache = task.cargo_cache.as_deref();
        if task.grade_label.as_deref() == Some("rust") {
            if let Some(cache) = cargo_cache {
                self.warm_cargo_cache(task, cache)?;
            }
        }
        let plan = container_grade_plan(
            &GradeTask {
                id: task.id.clone(),
                workspace: task.workspace.clone(),
                test_command: task.test_command.clone(),
            },
            task.image.as_deref(),
            cargo_cache,
        );
        let execution = self.execute_plan(&plan)?;
        let _ = self.breaker.record(execution.timed_out);
        if execution.timed_out {
            return Ok(GradeVerdict {
                passed: None,
                timed_out: true,
                infra_gap: None,
                cleanup_diagnostic: execution.cleanup_diagnostic,
                note: format!("grade timed out after {}s", self.timeout.as_secs()),
            });
        }
        // A failed grade whose tail is a cargo offline-fetch error is an
        // under-vendored cache, not a real test failure: the offline rust grade
        // (`--network=none` + `CARGO_NET_OFFLINE`) could not resolve a dependency
        // the exercise (or the solver) reached for. Per `grade.rs` this must read
        // as an infrastructure gap — a caveat/floor — never a silent solve
        // failure that deflates the arm's rate.
        if !execution.exit_ok && is_offline_fetch_error(&execution.output) {
            return Ok(GradeVerdict {
                passed: None,
                timed_out: false,
                infra_gap: Some(format!(
                    "task '{}': the offline cargo cache is under-vendored (a dependency could \
                     not be fetched under --network=none) — re-warm the cache and rescore",
                    task.id
                )),
                cleanup_diagnostic: None,
                note: "offline cargo fetch failed (infrastructure gap, not a solve failure)"
                    .to_string(),
            });
        }
        let label = task.grade_label.as_deref().unwrap_or("");
        let tests = test_count(label, &execution.output);
        // Grade fidelity: passed requires exit 0 AND tests that actually ran.
        Ok(GradeVerdict {
            passed: Some(execution.exit_ok && tests > 0),
            timed_out: false,
            infra_gap: None,
            cleanup_diagnostic: None,
            note: annotate_cell_note(tests, false, ""),
        })
    }
}

fn output_suffix(output: &str) -> String {
    let output = output.trim();
    if output.is_empty() {
        String::new()
    } else {
        format!(": {output}")
    }
}

/// Run a command via the OS for the container grader (the production seam).
///
/// # Errors
/// A message when the process cannot be spawned.
pub fn os_exec(command: &[String], timeout: Duration) -> Result<(bool, String, bool), String> {
    let (program, args) = command
        .split_first()
        .ok_or("empty grade command".to_string())?;
    let run = run_bounded(program, args, None, timeout)?;
    let mut output = format!("{}{}", run.stdout, run.stderr);
    if let Some(diagnostic) = run.cleanup_diagnostic {
        output.push_str(&format!("\nprocess cleanup: {diagnostic}"));
    }
    Ok((run.exit_ok, output, run.timed_out))
}

/// The matrix outcome: the comparative report plus run bookkeeping.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatrixOutcome {
    pub report: CapabilityReport,
    /// Cells persisted for offline rescore.
    pub cells_saved: usize,
    /// Fairness caveats (wall-clock-budget cells count unsolved — a floor).
    pub caveats: Vec<String>,
    /// Why the run yielded early, when it did (ledger intact).
    pub aborted: Option<String>,
}

/// Patch a raw scorecard's graded verdict so the persisted cell rescoreS to
/// the same report the live run printed.
fn patch_scorecard_verdict(raw: &str, passed: bool) -> String {
    let Ok(mut value) = serde_json::from_str::<serde_json::Value>(raw) else {
        return raw.to_string();
    };
    if let Some(results) = value.get_mut("results") {
        results["passed"] = serde_json::Value::Bool(passed);
    }
    serde_json::to_string(&value).unwrap_or_else(|_| raw.to_string())
}

/// A synthetic scorecard JSON for a failed/hung cell — every metric zeroed and
/// `passed = false` — so a persisted failed cell round-trips through
/// `parse_capability_scorecard` to the same unsolved card `failed_card` yields.
/// This is what makes `rescore` reproduce the live report: a failed cell is on
/// disk, not silently dropped.
fn synthetic_unsolved_scorecard(task: &TaskSpec, arm: &str, model: &str, reason: &str) -> String {
    use localx_eval_core::{
        ProcessBlock, QualityBlock, ResultsBlock, Scorecard, SpeedBlock, SCORECARD_SCHEMA,
    };
    let card = Scorecard {
        schema: SCORECARD_SCHEMA,
        task: task.id.clone(),
        arm: arm.to_string(),
        model: model.to_string(),
        results: ResultsBlock {
            passed: false,
            regression_safe: false,
            partial_credit: 0.0,
            tests_total: 0,
            tests_passed: 0,
        },
        quality: QualityBlock {
            diff_added: 0,
            diff_removed: 0,
            diff_files: 0,
            vs_gold_ratio: None,
            format_clean: false,
            lint_clean: false,
            typecheck_clean: false,
            complexity_delta: None,
            tests_added: false,
        },
        process: ProcessBlock {
            tool_calls: 0,
            redundant_calls: 0,
            reproduce_before_fix: false,
            test_before_done: false,
            retrieval_used: false,
            retrieval_count: 0,
            exit_reason: format!("cell-failed: {reason}"),
            recovered_after_failure: false,
            interventions: 0,
            discipline: None,
        },
        speed: SpeedBlock {
            wall_ms: 0,
            input_tokens: 0,
            output_tokens: 0,
        },
        judge: None,
    };
    card.to_json().unwrap_or_default()
}

/// Persist a synthetic unsolved cell so a failed/hung solve is on disk for
/// `rescore`, exactly as the live report already counts it.
fn save_failed_cell(
    cells_dir: &Path,
    task: &TaskSpec,
    arm: &ArmSpec,
    model: &str,
    reason: &str,
) -> Result<Cell, String> {
    let cell = Cell {
        task: task.id.clone(),
        arm: arm.arm.clone(),
        model: model.to_string(),
        arm_config: arm.config.clone(),
        scorecard: synthetic_unsolved_scorecard(task, &arm.arm, model, reason),
    };
    save_cell(cells_dir, &cell)
        .map(|_| cell)
        .map_err(|e| e.to_string())
}

/// A synthetic unsolved card for an isolated cell failure.
fn failed_card(task: &TaskSpec, arm: &str, model: &str, reason: &str) -> ParsedCard {
    ParsedCard {
        task: task.id.clone(),
        arm: arm.to_string(),
        model: model.to_string(),
        passed: false,
        partial: 0.0,
        regression_safe: false,
        diff_added: 0,
        diff_removed: 0,
        vs_gold: None,
        tool_calls: 0,
        redundant: 0,
        retrieval: false,
        exit_reason: format!("cell-failed: {reason}"),
        wall_ms: 0,
        judge_overall: None,
        interventions: 0,
    }
}

/// Run the full arm × task matrix. Every cell that yields a scorecard is
/// persisted under `cells_dir` (graded verdict included) so the report can be
/// recomputed offline; a failed cell is isolated as an unsolved card.
///
/// # Errors
/// Setup failures only (broken instruments, contaminated baseline, corpus
/// path inside a LocalPilot checkout, unwritable cells) — a mid-run grader
/// failure yields a partial result instead of an error.
pub fn run_matrix(
    solver: &mut dyn Solver,
    grader: &mut dyn Grader,
    spec: &RunSpec,
    cells_dir: &Path,
    log: &mut dyn FnMut(&str),
    on_event: &mut dyn FnMut(&RunEvent),
) -> Result<MatrixOutcome, String> {
    // Prove the deterministic instruments still work before spending.
    assert_instruments_ready(&instrument_checks()).map_err(|e| e.to_string())?;
    for task in &spec.tasks {
        assert_external_corpus_path(&task.workspace).map_err(|e| e.to_string())?;
    }
    // A contaminated baseline is refused at setup, before any solver run. Two
    // checks close the "isolation is opt-in" gap, so a baseline can never anchor
    // a delta unguarded:
    //   1. A baseline arm MUST carry a `config` block. Without one its isolation
    //      cannot be checked, yet it still anchors every harness-vs-baseline
    //      delta — a missing config would let an unchecked baseline through.
    //   2. The declared clean config is cross-checked against the arm's ACTUAL
    //      solver invocation: a baseline that also sets a harness flag
    //      (`verify`/`learn`) is contradictory, because those flags reach
    //      `localpilot eval` as `--verify`/`--learn` (harness behaviour), so the
    //      "harness off" the config asserts is not what actually runs.
    for arm in &spec.arms {
        match &arm.config {
            Some(config) => {
                let effective =
                    assert_arm_isolation(&arm.arm, config).map_err(|e| e.to_string())?;
                if effective.is_baseline {
                    let mut harness_flags = Vec::new();
                    if arm.verify {
                        harness_flags.push("--verify");
                    }
                    if arm.learn {
                        harness_flags.push("--learn");
                    }
                    if !harness_flags.is_empty() {
                        return Err(format!(
                            "baseline arm '{}' declares the harness off but its run sets {} — the \
                             declared clean config contradicts the actual solver invocation \
                             (localpilot eval would receive that harness flag). Drop the flag or \
                             mark the arm non-baseline.",
                            arm.arm,
                            harness_flags.join(" and ")
                        ));
                    }
                }
            }
            None => {
                if arm.arm.eq_ignore_ascii_case("baseline") {
                    return Err(format!(
                        "baseline arm '{}' carries no `config` block, so its isolation cannot be \
                         checked. A baseline anchors every harness-vs-baseline delta and must \
                         declare its effective config — add `\"config\": {{ \"is_baseline\": true \
                         }}` (plus any env/config_file/plugins/system_prompt/retrieval it really \
                         uses).",
                        arm.arm
                    ));
                }
            }
        }
    }

    // The JSONL protocol: `started` fires only after every setup gate has
    // passed (a refused run emits nothing), each persisted cell streams a
    // `result`, and the run always terminates the stream — `completed` on a
    // full matrix, `error` on an abort. No dangling `started`.
    let run_name = format!("arms-{}", spec.model);
    let emit_cell = |cell: &Cell, on_event: &mut dyn FnMut(&RunEvent)| {
        let payload = serde_json::to_value(cell).unwrap_or(serde_json::Value::Null);
        on_event(&RunEvent::Result {
            run: run_name.clone(),
            payload,
        });
    };
    on_event(&RunEvent::Started {
        run: run_name.clone(),
        total: Some((spec.arms.len() * spec.tasks.len()) as u64),
    });

    let mut results: Vec<ArmCards> = Vec::new();
    let mut cells_saved = 0_usize;
    let mut caveats = Vec::new();
    let mut aborted = None;

    'arms: for arm in &spec.arms {
        let mut cards = Vec::new();
        let mut timed_out_cells = 0_usize;
        let mut infra_gap_cells = 0_usize;
        let mut cleanup_issue_cells = 0_usize;
        for task in &spec.tasks {
            log(&format!("arm '{}' task '{}': solving", arm.arm, task.id));
            let mut solve_spec = SolveSpec::new(&spec.model, &arm.arm, &task.id, &task.problem);
            solve_spec.verify = arm.verify;
            solve_spec.learn = arm.learn;
            solve_spec.coach = arm.coach.clone();
            let raw = match solver.solve(Path::new(&task.workspace), &solve_spec) {
                Ok(raw) => raw,
                Err(reason) => {
                    // A failing or hung cell is isolated: record it unsolved so
                    // the arm still aggregates and the matrix completes — and
                    // persist it so `rescore` sees the same cell the live report
                    // counts.
                    log(&format!(
                        "arm '{}' task '{}': {reason} (recorded unsolved)",
                        arm.arm, task.id
                    ));
                    cards.push(failed_card(task, &arm.arm, &spec.model, &reason));
                    let cell = save_failed_cell(cells_dir, task, arm, &spec.model, &reason)?;
                    emit_cell(&cell, on_event);
                    cells_saved += 1;
                    continue;
                }
            };
            let mut card = match parse_capability_scorecard(&raw) {
                Ok(card) => card,
                Err(e) => {
                    log(&format!(
                        "arm '{}' task '{}': {e} (recorded unsolved)",
                        arm.arm, task.id
                    ));
                    cards.push(failed_card(task, &arm.arm, &spec.model, &e.to_string()));
                    let cell = save_failed_cell(cells_dir, task, arm, &spec.model, &e.to_string())?;
                    emit_cell(&cell, on_event);
                    cells_saved += 1;
                    continue;
                }
            };

            let mut persisted = raw;
            match grader.grade(task) {
                Ok(verdict) => {
                    if let Some(diagnostic) = &verdict.cleanup_diagnostic {
                        cleanup_issue_cells += 1;
                        log(&format!(
                            "arm '{}' task '{}': secondary cleanup issue: {diagnostic}",
                            arm.arm, task.id
                        ));
                    }
                    if let Some(reason) = &verdict.infra_gap {
                        // The grade environment failed (under-vendored offline
                        // cache): the cell counts unsolved (a floor) and the run
                        // carries an infra caveat, never a silent solve failure.
                        infra_gap_cells += 1;
                        card.passed = false;
                        persisted = patch_scorecard_verdict(&persisted, false);
                        log(&format!("arm '{}' task '{}': {reason}", arm.arm, task.id));
                    } else if verdict.timed_out {
                        timed_out_cells += 1;
                        card.passed = false;
                        persisted = patch_scorecard_verdict(&persisted, false);
                    } else if let Some(passed) = verdict.passed {
                        card.passed = passed;
                        persisted = patch_scorecard_verdict(&persisted, passed);
                    }
                    if !verdict.note.is_empty() {
                        log(&format!(
                            "arm '{}' task '{}': {}",
                            arm.arm, task.id, verdict.note
                        ));
                    }
                }
                Err(reason) => {
                    // Grading infrastructure died (wedged engine): yield with
                    // the ledger intact rather than burning the rest. The grader
                    // never returned a verdict, so the solver's self-claim is
                    // untrusted — mark the in-flight cell unsolved (a floor),
                    // mirroring a timeout, so the yielded ledger and rescore agree.
                    log(&reason);
                    card.passed = false;
                    persisted = patch_scorecard_verdict(&persisted, false);
                    cards.push(card);
                    let cell = Cell {
                        task: task.id.clone(),
                        arm: arm.arm.clone(),
                        model: spec.model.clone(),
                        arm_config: arm.config.clone(),
                        scorecard: persisted,
                    };
                    save_cell(cells_dir, &cell).map_err(|e| e.to_string())?;
                    emit_cell(&cell, on_event);
                    cells_saved += 1;
                    results.push(ArmCards {
                        arm: arm.arm.clone(),
                        model: spec.model.clone(),
                        cards: std::mem::take(&mut cards),
                        arm_config: arm.config.clone(),
                    });
                    aborted = Some(reason);
                    break 'arms;
                }
            }

            let cell = Cell {
                task: task.id.clone(),
                arm: arm.arm.clone(),
                model: spec.model.clone(),
                arm_config: arm.config.clone(),
                scorecard: persisted,
            };
            save_cell(cells_dir, &cell).map_err(|e| e.to_string())?;
            emit_cell(&cell, on_event);
            cells_saved += 1;
            cards.push(card);
        }
        if let Some(caveat) = timeout_caveat(&arm.arm, timed_out_cells) {
            caveats.push(caveat);
        }
        if let Some(caveat) = infra_gap_caveat(&arm.arm, infra_gap_cells) {
            caveats.push(caveat);
        }
        if cleanup_issue_cells > 0 {
            caveats.push(format!(
                "caveat: {cleanup_issue_cells} {} cell(s) reported a secondary exact-container \
                 cleanup failure; primary grade and timeout outcomes were preserved",
                arm.arm
            ));
        }
        results.push(ArmCards {
            arm: arm.arm.clone(),
            model: spec.model.clone(),
            cards,
            arm_config: arm.config.clone(),
        });
    }

    let outcome = MatrixOutcome {
        report: capability_report(&results, &spec.corpus),
        cells_saved,
        caveats,
        aborted,
    };
    match &outcome.aborted {
        Some(reason) => on_event(&RunEvent::Error {
            run: run_name.clone(),
            message: reason.clone(),
        }),
        None => on_event(&RunEvent::Completed {
            run: run_name.clone(),
            payload: serde_json::to_value(&outcome).unwrap_or(serde_json::Value::Null),
        }),
    }
    Ok(outcome)
}

/// Render a comparative capability report as Markdown, with the contamination
/// caveat printed for public corpora so absolute numbers read as deltas.
#[must_use]
pub fn render_capability_report(report: &CapabilityReport) -> String {
    let mut lines = vec![
        format!(
            "# Harness capability — comparative ({} arm(s), corpus: {})",
            report.arms.len(),
            report.corpus
        ),
        String::new(),
    ];
    if report.contamination_suspect {
        lines
            .push("> Contamination caveat: this is a public benchmark; treat absolute".to_string());
        lines
            .push("> numbers as suspect and read them as deltas between harness arms.".to_string());
        lines.push(String::new());
    }
    lines.push(
        "| arm | model | solved | solve rate | avg tool calls | avg redundant | avg interventions | judge | isolation |"
            .to_string(),
    );
    lines.push("|---|---|---|---|---|---|---|---|---|".to_string());
    for arm in &report.arms {
        let judge = arm
            .judge_overall
            .map_or_else(|| "n/a".to_string(), |j| format!("{j:.2}"));
        let isolation = arm.isolation.clone().unwrap_or_else(|| "n/a".to_string());
        lines.push(format!(
            "| {} | {} | {}/{} | {:.0}% | {:.2} | {:.2} | {:.2} | {} | {} |",
            arm.arm,
            arm.model,
            arm.solved,
            arm.tasks,
            arm.solve_rate * 100.0,
            arm.avg_tool_calls,
            arm.avg_redundant,
            arm.avg_interventions,
            judge,
            isolation
        ));
    }
    lines.join("\n")
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use localbench_measure::runner::rescore;

    fn card_json(task: &str, arm: &str, passed: bool) -> String {
        format!(
            r#"{{"schema":1,"task":"{task}","arm":"{arm}","model":"apex",
                "results":{{"passed":{passed},"regression_safe":true,"partial_credit":1.0,"tests_total":1,"tests_passed":1}},
                "quality":{{"diff_added":3,"diff_removed":0,"diff_files":1,"vs_gold_ratio":null,"format_clean":true,"lint_clean":true,"typecheck_clean":true,"complexity_delta":null,"tests_added":false}},
                "process":{{"tool_calls":2,"redundant_calls":0,"reproduce_before_fix":true,"test_before_done":true,"retrieval_used":false,"retrieval_count":0,"exit_reason":"Done","recovered_after_failure":false,"discipline":null}},
                "speed":{{"wall_ms":100,"input_tokens":1,"output_tokens":1}},"judge":null}}"#
        )
    }

    struct MockSolver {
        fail_task: Option<String>,
    }

    impl Solver for MockSolver {
        fn solve(&mut self, _workspace: &Path, spec: &SolveSpec) -> Result<String, String> {
            if self.fail_task.as_deref() == Some(spec.task.as_str()) {
                return Err("solver hung (mock)".to_string());
            }
            Ok(card_json(&spec.task, &spec.arm, true))
        }
    }

    fn spec_with(arms: Vec<ArmSpec>, tasks: Vec<TaskSpec>) -> RunSpec {
        RunSpec {
            schema: 1,
            model: "apex".to_string(),
            corpus: "external".to_string(),
            arms,
            tasks,
        }
    }

    fn task(id: &str) -> TaskSpec {
        TaskSpec {
            id: id.to_string(),
            workspace: format!("C:/bench/work/{id}"),
            problem: "fix it".to_string(),
            test_command: "cargo test".to_string(),
            image: None,
            grade_label: Some("rust".to_string()),
            cargo_cache: None,
        }
    }

    fn arm(name: &str) -> ArmSpec {
        let config = (name == "baseline").then(RawArmConfig::default);
        ArmSpec {
            arm: name.to_string(),
            config,
            verify: false,
            learn: false,
            coach: None,
        }
    }

    #[test]
    fn run_specs_fail_loud_on_defects() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("spec.json");
        std::fs::write(&path, r#"{"schema":2,"model":"m","arms":[],"tasks":[]}"#).unwrap();
        assert!(load_run_spec(&path).unwrap_err().contains("schema 2"));
        std::fs::write(
            &path,
            r#"{"schema":1,"model":"m","arms":[{"arm":"full"}],"tasks":[]}"#,
        )
        .unwrap();
        assert!(load_run_spec(&path)
            .unwrap_err()
            .contains("at least one task"));
        std::fs::write(
            &path,
            r#"{"schema":1,"model":"m","corpus":"external",
                "arms":[{"arm":"full","verify":true}],
                "tasks":[{"id":"t","workspace":"C:/w/t","problem":"p","test_command":"c"}]}"#,
        )
        .unwrap();
        let spec = load_run_spec(&path).unwrap();
        assert!(spec.arms[0].verify);

        std::fs::write(
            &path,
            r#"{"schema":1,"model":"m","corpus":"external",
                "arms":[{"arm":"full"}],
                "tasks":[{"id":"rust-task","workspace":"C:/corpus/rust/exercises/practice/x",
                "problem":"p","test_command":"cargo test","grade_label":"rust"}]}"#,
        )
        .unwrap();
        let error = load_run_spec(&path).unwrap_err();
        assert!(error.contains("rust task 'rust-task' needs cargo_cache"));
    }

    #[test]
    fn a_failed_cell_is_isolated_and_the_matrix_completes() {
        let dir = tempfile::tempdir().unwrap();
        let mut solver = MockSolver {
            fail_task: Some("t2".to_string()),
        };
        let spec = spec_with(
            vec![arm("baseline"), arm("full")],
            vec![task("t1"), task("t2")],
        );
        let outcome = run_matrix(
            &mut solver,
            &mut NoGrader,
            &spec,
            dir.path(),
            &mut |_| {},
            &mut |_| {},
        )
        .unwrap();
        assert!(outcome.aborted.is_none());
        assert_eq!(outcome.report.arms.len(), 2);
        for row in &outcome.report.arms {
            assert_eq!(row.tasks, 2);
            assert_eq!(row.solved, 1, "the hung cell counts unsolved");
        }
        // Every cell is persisted — the passing one AND the hung one as a
        // synthetic unsolved cell — so `rescore` sees the same 2 tasks/arm the
        // live report counted (2 arms × 2 tasks = 4).
        assert_eq!(outcome.cells_saved, 4);
        // rescore ≡ live: recomputing from disk reproduces the live report,
        // including the injected failed cell. Before the synthetic-cell fix this
        // diverged (rescore saw only the passing cells → a higher solve rate).
        let rescored = rescore(dir.path(), "external").unwrap();
        assert_eq!(rescored, outcome.report);
        // The tracked baseline reads clean; the harness arm is n/a.
        let baseline = &outcome.report.arms[0];
        assert_eq!(baseline.isolation.as_deref(), Some("clean"));
    }

    #[test]
    fn graded_verdicts_are_persisted_so_rescore_matches_the_live_report() {
        let dir = tempfile::tempdir().unwrap();
        let mut solver = MockSolver { fail_task: None };
        struct FailAll;
        impl Grader for FailAll {
            fn grade(&mut self, _task: &TaskSpec) -> Result<GradeVerdict, String> {
                Ok(GradeVerdict {
                    passed: Some(false),
                    timed_out: false,
                    infra_gap: None,
                    cleanup_diagnostic: None,
                    note: String::new(),
                })
            }
        }
        let spec = spec_with(vec![arm("full")], vec![task("t1")]);
        let outcome = run_matrix(
            &mut solver,
            &mut FailAll,
            &spec,
            dir.path(),
            &mut |_| {},
            &mut |_| {},
        )
        .unwrap();
        // The solver self-reported passed; the grader's verdict wins.
        assert_eq!(outcome.report.arms[0].solved, 0);
        let rescored = rescore(dir.path(), "external").unwrap();
        assert_eq!(rescored, outcome.report);
    }

    #[test]
    fn a_grade_timeout_is_a_caveat_not_a_capability_result() {
        let dir = tempfile::tempdir().unwrap();
        let mut solver = MockSolver { fail_task: None };
        struct TimeoutGrader;
        impl Grader for TimeoutGrader {
            fn grade(&mut self, _task: &TaskSpec) -> Result<GradeVerdict, String> {
                Ok(GradeVerdict {
                    passed: None,
                    timed_out: true,
                    infra_gap: None,
                    cleanup_diagnostic: None,
                    note: "grade timed out after 5s".to_string(),
                })
            }
        }
        let spec = spec_with(vec![arm("full")], vec![task("t1")]);
        let outcome = run_matrix(
            &mut solver,
            &mut TimeoutGrader,
            &spec,
            dir.path(),
            &mut |_| {},
            &mut |_| {},
        )
        .unwrap();
        assert_eq!(
            outcome.report.arms[0].solved, 0,
            "timed out counts unsolved"
        );
        assert_eq!(outcome.caveats.len(), 1);
        assert!(outcome.caveats[0].contains("floor"));
    }

    #[test]
    fn a_contaminated_baseline_is_refused_before_any_solve() {
        let dir = tempfile::tempdir().unwrap();
        let mut solver = MockSolver { fail_task: None };
        let mut baseline = arm("baseline");
        baseline.config = Some(RawArmConfig {
            retrieval: true,
            ..RawArmConfig::default()
        });
        let spec = spec_with(vec![baseline], vec![task("t1")]);
        let err = run_matrix(
            &mut solver,
            &mut NoGrader,
            &spec,
            dir.path(),
            &mut |_| {},
            &mut |_| {},
        )
        .unwrap_err();
        assert!(err.contains("retrieval"));
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    }

    #[test]
    fn a_baseline_arm_without_a_config_is_refused_before_any_solve() {
        // A baseline with NO config block used to silently skip the isolation
        // check and run its delta unguarded; it must now be refused at setup.
        let dir = tempfile::tempdir().unwrap();
        let mut solver = MockSolver { fail_task: None };
        let baseline = ArmSpec {
            arm: "baseline".to_string(),
            config: None,
            verify: false,
            learn: false,
            coach: None,
        };
        let spec = spec_with(vec![baseline], vec![task("t1")]);
        let err = run_matrix(
            &mut solver,
            &mut NoGrader,
            &spec,
            dir.path(),
            &mut |_| {},
            &mut |_| {},
        )
        .unwrap_err();
        assert!(err.contains("no `config` block"));
        assert!(err.contains("baseline"));
        // Refused before any cell was written.
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    }

    #[test]
    fn a_baseline_that_also_sets_a_harness_flag_is_refused() {
        // The declared clean config is cross-checked against the ACTUAL solver
        // invocation: a baseline that also flips a harness flag contradicts its
        // "harness off" claim, because the flag reaches `localpilot eval`.
        let dir = tempfile::tempdir().unwrap();
        let mut solver = MockSolver { fail_task: None };
        let mut baseline = arm("baseline"); // carries a clean default config
        baseline.verify = true;
        let spec = spec_with(vec![baseline], vec![task("t1")]);
        let err = run_matrix(
            &mut solver,
            &mut NoGrader,
            &spec,
            dir.path(),
            &mut |_| {},
            &mut |_| {},
        )
        .unwrap_err();
        assert!(err.contains("--verify"));
        assert!(err.contains("contradicts the actual solver invocation"));
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    }

    #[test]
    fn a_corpus_path_inside_localpilot_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let mut solver = MockSolver { fail_task: None };
        let mut bad = task("t1");
        bad.workspace = "D:/repos/LocalX/LocalPilot/corpus/t1".to_string();
        let spec = spec_with(vec![arm("full")], vec![bad]);
        let err = run_matrix(
            &mut solver,
            &mut NoGrader,
            &spec,
            dir.path(),
            &mut |_| {},
            &mut |_| {},
        )
        .unwrap_err();
        assert!(err.contains("LocalPilot"));
    }

    #[test]
    fn the_container_grader_enforces_grade_fidelity_and_health() {
        // Health check runs once and must pass before any grade.
        let mut calls: Vec<Vec<String>> = Vec::new();
        let exec = |command: &[String], _t: Duration| {
            calls.push(command.to_vec());
            if command.contains(&"busybox".to_string()) {
                return Ok((true, String::new(), false));
            }
            // Exit 0 but no tests ran — grade fidelity demands a fail.
            Ok((true, "Compiling... Finished.".to_string(), false))
        };
        let verdict = {
            let mut grader = ContainerGrader::new(exec, Duration::from_secs(5));
            grader.grade(&task("t1")).unwrap()
        };
        assert_eq!(
            verdict.passed,
            Some(false),
            "exit 0 with 0 tests is not a pass"
        );
        assert!(verdict.note.contains("0 tests ran"));
        assert!(calls[0].contains(&"busybox".to_string()));
        assert!(calls[1].contains(&"--network=none".to_string()));

        // A passing grade needs exit 0 AND counted tests.
        let exec = |command: &[String], _t: Duration| {
            if command.contains(&"busybox".to_string()) {
                return Ok((true, String::new(), false));
            }
            Ok((
                true,
                "test result: ok. 5 passed; 0 failed; 0 ignored".to_string(),
                false,
            ))
        };
        let mut grader = ContainerGrader::new(exec, Duration::from_secs(5));
        assert_eq!(grader.grade(&task("t1")).unwrap().passed, Some(true));
    }

    #[test]
    fn a_health_timeout_removes_only_its_exact_named_container() {
        let mut calls: Vec<Vec<String>> = Vec::new();
        let exec = |command: &[String], _t: Duration| {
            calls.push(command.to_vec());
            if command.get(1).map(String::as_str) == Some("rm") {
                return Ok((
                    false,
                    "Error: No such container: already-absent".to_string(),
                    false,
                ));
            }
            Ok((false, "health still running".to_string(), true))
        };
        let error = {
            let mut grader = ContainerGrader::new(exec, Duration::from_secs(5));
            grader.grade(&task("t1")).unwrap_err()
        };
        assert!(error.contains("pre-flight health check"));
        assert!(error.contains("timed out after 5s"));
        assert_eq!(calls.len(), 2);
        let name = calls[0]
            .windows(2)
            .find(|args| args[0] == "--name")
            .map(|args| args[1].clone())
            .expect("health plan has an exact name");
        assert_eq!(
            calls[1],
            vec![
                "docker".to_string(),
                "rm".to_string(),
                "-f".to_string(),
                name
            ]
        );
    }

    #[test]
    fn an_ambiguous_docker_spawn_error_still_runs_exact_compensation() {
        let mut calls: Vec<Vec<String>> = Vec::new();
        let exec = |command: &[String], _t: Duration| {
            calls.push(command.to_vec());
            if command.get(1).map(String::as_str) == Some("rm") {
                Ok((true, String::new(), false))
            } else {
                Err("docker CLI connection broke after create".to_string())
            }
        };
        let error = {
            let mut grader = ContainerGrader::new(exec, Duration::from_secs(5));
            grader.grade(&task("t1")).unwrap_err()
        };
        assert!(error.contains("connection broke after create"));
        assert!(error.contains("exact Docker cleanup completed"));
        assert_eq!(calls.len(), 2);
        let expected_name = calls[0]
            .windows(2)
            .find(|args| args[0] == "--name")
            .map(|args| args[1].clone())
            .expect("health plan has an exact name");
        assert_eq!(calls[1][3], expected_name);
    }

    #[test]
    fn a_grade_timeout_preserves_primary_semantics_when_exact_cleanup_fails() {
        let mut calls: Vec<Vec<String>> = Vec::new();
        let exec = |command: &[String], _t: Duration| {
            calls.push(command.to_vec());
            if command.contains(&"busybox".to_string()) {
                return Ok((true, String::new(), false));
            }
            if command.get(1).map(String::as_str) == Some("rm") {
                return Ok((false, "daemon unavailable".to_string(), false));
            }
            Ok((false, "grade still running".to_string(), true))
        };
        let verdict = {
            let mut grader = ContainerGrader::new(exec, Duration::from_secs(5));
            grader.grade(&task("t1")).unwrap()
        };
        assert!(verdict.timed_out);
        assert_eq!(verdict.passed, None);
        assert!(verdict.note.contains("grade timed out"));
        assert!(verdict
            .cleanup_diagnostic
            .as_deref()
            .is_some_and(|diagnostic| diagnostic.contains("daemon unavailable")));
        assert_eq!(calls.len(), 3);
        let grade_name = calls[1]
            .windows(2)
            .find(|args| args[0] == "--name")
            .map(|args| args[1].clone())
            .expect("grade plan has an exact name");
        assert_eq!(calls[2][3], grade_name);
    }

    #[test]
    fn rust_cache_warms_before_grading_and_the_same_cell_does_not_refetch() {
        let dir = tempfile::tempdir().unwrap();
        let exercise = dir
            .path()
            .join("rust")
            .join("exercises")
            .join("practice")
            .join("clock");
        std::fs::create_dir_all(&exercise).unwrap();
        std::fs::write(
            exercise.join("Cargo.toml"),
            "[package]\nname = \"clock\"\nversion = \"0.1.0\"\n\n[dependencies]\nserde = \"1\"\n",
        )
        .unwrap();
        let cache = dir.path().join("cargo-registry");
        let mut rust_task = task("clock");
        rust_task.workspace = exercise.to_string_lossy().into_owned();
        rust_task.cargo_cache = Some(cache.to_string_lossy().into_owned());

        let mut calls: Vec<Vec<String>> = Vec::new();
        let exec = |command: &[String], _t: Duration| {
            calls.push(command.to_vec());
            if command.contains(&"busybox".to_string())
                || command.windows(2).any(|w| w == ["cargo", "fetch"])
            {
                return Ok((true, String::new(), false));
            }
            Ok((
                true,
                "test result: ok. 1 passed; 0 failed; 0 ignored".to_string(),
                false,
            ))
        };
        {
            let mut grader = ContainerGrader::new(exec, Duration::from_secs(5));
            assert_eq!(grader.grade(&rust_task).unwrap().passed, Some(true));
            assert_eq!(grader.grade(&rust_task).unwrap().passed, Some(true));
        }

        assert!(calls[0].contains(&"busybox".to_string()));
        assert!(calls[1].windows(2).any(|w| w == ["cargo", "fetch"]));
        assert!(!calls[1].contains(&"--network=none".to_string()));
        assert!(calls[2].contains(&"--network=none".to_string()));
        assert!(calls[3].contains(&"--network=none".to_string()));
        assert_eq!(
            calls
                .iter()
                .filter(|command| command.windows(2).any(|w| w == ["cargo", "fetch"]))
                .count(),
            1,
            "the same cache/manifest pair is fetched only once"
        );
        let warm_dir = cache.join(".localbench-warm");
        let manifest = std::fs::read_to_string(warm_dir.join("Cargo.toml")).unwrap();
        assert!(manifest.contains("serde = \"1\""));
        assert!(manifest.contains("rayon = \"1\""));

        // A mocked exec cannot prove the generated crate is one cargo accepts,
        // and that is exactly what was wrong: a manifest carrying dependencies
        // and no target, which `cargo fetch` rejects while parsing. `cargo
        // metadata --no-deps` runs the same manifest load without resolving or
        // downloading anything, so it catches the defect offline.
        let metadata = std::process::Command::new(env!("CARGO"))
            .arg("metadata")
            .arg("--no-deps")
            .arg("--format-version")
            .arg("1")
            .arg("--offline")
            .arg("--manifest-path")
            .arg(warm_dir.join("Cargo.toml"))
            .output()
            .unwrap();
        assert!(
            metadata.status.success(),
            "cargo must accept the generated warm crate: {}",
            String::from_utf8_lossy(&metadata.stderr)
        );
    }

    #[test]
    fn an_offline_fetch_failure_grades_as_an_infra_gap_not_a_solve_failure() {
        // A failed grade whose tail is a cargo offline-fetch error is an
        // under-vendored cache, not a real test failure. Health check passes,
        // then the grade fails with the offline marker.
        let exec = |command: &[String], _t: Duration| {
            if command.contains(&"busybox".to_string()) {
                return Ok((true, String::new(), false));
            }
            Ok((
                false,
                "error: no matching package named `leftpad` found\n\
                 (while attempting to fetch under --offline)"
                    .to_string(),
                false,
            ))
        };
        let mut grader = ContainerGrader::new(exec, Duration::from_secs(5));
        let verdict = grader.grade(&task("t1")).unwrap();
        // Not scored a solve failure: no pass/fail verdict, an infra-gap reason.
        assert_eq!(verdict.passed, None, "an infra gap is not a graded solve");
        assert!(verdict.infra_gap.is_some());
        assert!(verdict.infra_gap.unwrap().contains("under-vendored"));
        // A genuine compile error (not a fetch error) still scores unsolved.
        let exec = |command: &[String], _t: Duration| {
            if command.contains(&"busybox".to_string()) {
                return Ok((true, String::new(), false));
            }
            Ok((
                false,
                "error[E0308]: mismatched types\n --> src/lib.rs:4:5".to_string(),
                false,
            ))
        };
        let mut grader = ContainerGrader::new(exec, Duration::from_secs(5));
        let verdict = grader.grade(&task("t1")).unwrap();
        assert_eq!(verdict.passed, Some(false));
        assert!(verdict.infra_gap.is_none());
    }

    #[test]
    fn an_infra_gap_cell_is_a_run_caveat_and_counts_unsolved() {
        let dir = tempfile::tempdir().unwrap();
        let mut solver = MockSolver { fail_task: None };
        struct InfraGapGrader;
        impl Grader for InfraGapGrader {
            fn grade(&mut self, _task: &TaskSpec) -> Result<GradeVerdict, String> {
                Ok(GradeVerdict {
                    passed: None,
                    timed_out: false,
                    infra_gap: Some("offline cargo cache under-vendored".to_string()),
                    cleanup_diagnostic: None,
                    note: String::new(),
                })
            }
        }
        let spec = spec_with(vec![arm("full")], vec![task("t1")]);
        let outcome = run_matrix(
            &mut solver,
            &mut InfraGapGrader,
            &spec,
            dir.path(),
            &mut |_| {},
            &mut |_| {},
        )
        .unwrap();
        // The solver self-reported passed; an infra gap floors the cell unsolved
        // rather than scoring it a solve failure or trusting the self-claim.
        assert_eq!(outcome.report.arms[0].solved, 0);
        assert_eq!(outcome.report.arms[0].tasks, 1);
        assert_eq!(outcome.caveats.len(), 1);
        assert!(outcome.caveats[0].contains("infrastructure gap"));
        // rescore ≡ live: the unsolved cell is on disk, so a re-score agrees.
        let rescored = rescore(dir.path(), "external").unwrap();
        assert_eq!(rescored, outcome.report);
    }

    #[test]
    fn a_cleanup_failure_is_secondary_and_does_not_change_rescore() {
        let dir = tempfile::tempdir().unwrap();
        let mut solver = MockSolver { fail_task: None };
        struct TimeoutWithCleanupFailure;
        impl Grader for TimeoutWithCleanupFailure {
            fn grade(&mut self, _task: &TaskSpec) -> Result<GradeVerdict, String> {
                Ok(GradeVerdict {
                    passed: None,
                    timed_out: true,
                    infra_gap: None,
                    cleanup_diagnostic: Some(
                        "exact Docker cleanup for container 'named' failed".to_string(),
                    ),
                    note: "grade timed out after 5s".to_string(),
                })
            }
        }
        let spec = spec_with(vec![arm("full")], vec![task("t1")]);
        let mut logs = Vec::new();
        let outcome = run_matrix(
            &mut solver,
            &mut TimeoutWithCleanupFailure,
            &spec,
            dir.path(),
            &mut |line| logs.push(line.to_string()),
            &mut |_| {},
        )
        .unwrap();
        assert_eq!(outcome.report.arms[0].solved, 0);
        assert_eq!(outcome.caveats.len(), 2);
        assert!(outcome
            .caveats
            .iter()
            .any(|caveat| caveat.contains("wall-clock")));
        assert!(outcome
            .caveats
            .iter()
            .any(|caveat| caveat.contains("secondary exact-container cleanup failure")));
        assert!(logs
            .iter()
            .any(|line| line.contains("secondary cleanup issue")));
        assert_eq!(rescore(dir.path(), "external").unwrap(), outcome.report);
    }

    #[test]
    fn three_grade_timeouts_trip_the_wedge_breaker_and_yield() {
        let exec = |command: &[String], _t: Duration| {
            if command.contains(&"busybox".to_string()) {
                return Ok((true, String::new(), false));
            }
            Ok((false, String::new(), true))
        };
        let mut grader = ContainerGrader::new(exec, Duration::from_secs(1));
        for _ in 0..3 {
            let verdict = grader.grade(&task("t1")).unwrap();
            assert!(verdict.timed_out);
        }
        let err = grader.grade(&task("t1")).unwrap_err();
        assert!(err.contains("wedged"));
        assert!(err.contains("ledger intact"));
    }

    #[test]
    fn a_wedged_engine_mid_matrix_yields_with_the_ledger_intact() {
        let dir = tempfile::tempdir().unwrap();
        let mut solver = MockSolver { fail_task: None };
        struct WedgeAfter(usize);
        impl Grader for WedgeAfter {
            fn grade(&mut self, _task: &TaskSpec) -> Result<GradeVerdict, String> {
                if self.0 == 0 {
                    return Err(
                        "docker looks wedged — yielding with the cell ledger intact".to_string()
                    );
                }
                self.0 -= 1;
                Ok(GradeVerdict::skipped())
            }
        }
        let spec = spec_with(vec![arm("full")], vec![task("t1"), task("t2"), task("t3")]);
        let outcome = run_matrix(
            &mut solver,
            &mut WedgeAfter(1),
            &spec,
            dir.path(),
            &mut |_| {},
            &mut |_| {},
        )
        .unwrap();
        assert!(outcome.aborted.unwrap().contains("wedged"));
        // Both attempted cells are on disk; the third never ran.
        assert_eq!(outcome.cells_saved, 2);
        assert_eq!(outcome.report.arms[0].tasks, 2);
    }

    #[test]
    fn the_event_stream_terminates_on_a_failed_run_with_no_dangling_started() {
        // The JSONL protocol on the abort path: started → one result per
        // persisted cell → error. The stream must always terminate — a
        // supervising harness reading a `started` with no terminal event
        // would wait forever (`Result`/`Error` were test-only before this).
        let dir = tempfile::tempdir().unwrap();
        let mut solver = MockSolver { fail_task: None };
        struct WedgeAfter(usize);
        impl Grader for WedgeAfter {
            fn grade(&mut self, _task: &TaskSpec) -> Result<GradeVerdict, String> {
                if self.0 == 0 {
                    return Err("docker looks wedged — yielding".to_string());
                }
                self.0 -= 1;
                Ok(GradeVerdict::skipped())
            }
        }
        let spec = spec_with(vec![arm("full")], vec![task("t1"), task("t2"), task("t3")]);
        let mut events: Vec<RunEvent> = Vec::new();
        let outcome = run_matrix(
            &mut solver,
            &mut WedgeAfter(1),
            &spec,
            dir.path(),
            &mut |_| {},
            &mut |e| events.push(e.clone()),
        )
        .unwrap();
        assert!(outcome.aborted.is_some());

        assert!(
            matches!(
                events.first(),
                Some(RunEvent::Started { total: Some(3), .. })
            ),
            "stream must open with started, got {events:?}"
        );
        let results = events
            .iter()
            .filter(|e| matches!(e, RunEvent::Result { .. }))
            .count();
        assert_eq!(results, 2, "one result per persisted cell");
        assert!(
            matches!(events.last(), Some(RunEvent::Error { .. })),
            "an aborted run must terminate the stream with error, got {events:?}"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, RunEvent::Completed { .. })),
            "an aborted run must not also claim completion"
        );

        // The happy path terminates with completed.
        let dir = tempfile::tempdir().unwrap();
        let mut events: Vec<RunEvent> = Vec::new();
        let spec = spec_with(vec![arm("full")], vec![task("t1")]);
        run_matrix(
            &mut solver,
            &mut NoGrader,
            &spec,
            dir.path(),
            &mut |_| {},
            &mut |e| events.push(e.clone()),
        )
        .unwrap();
        assert!(matches!(events.last(), Some(RunEvent::Completed { .. })));
    }

    #[test]
    fn the_markdown_report_carries_the_contamination_caveat() {
        let dir = tempfile::tempdir().unwrap();
        let mut solver = MockSolver { fail_task: None };
        let spec = spec_with(vec![arm("full")], vec![task("t1")]);
        let outcome = run_matrix(
            &mut solver,
            &mut NoGrader,
            &spec,
            dir.path(),
            &mut |_| {},
            &mut |_| {},
        )
        .unwrap();
        let rendered = render_capability_report(&outcome.report);
        assert!(rendered.contains("Contamination caveat"));
        assert!(rendered.contains("| full | apex | 1/1 | 100% |"));
    }
}
