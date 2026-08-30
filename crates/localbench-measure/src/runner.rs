//! The external-runner support: capability-scorecard parsing, the comparative
//! report with its anti-gaming paired-metric verdicts, the instrument
//! self-test gate, and cell persistence with deterministic offline rescore.
//!
//! Public corpora are contamination-suspect: absolute numbers are reported as
//! deltas between arms with the model pinned, never as trusted absolutes.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use localbench_scoring::stats::round_dp;
use localx_eval_core::Scorecard;

use crate::arms::{
    arm_effective_config, arm_isolation_summary, assert_arm_isolation, RawArmConfig,
};

/// A runner failure.
#[derive(Debug, thiserror::Error)]
pub enum RunnerError {
    /// The scorecard JSON did not parse or carried the wrong schema.
    #[error("not a capability scorecard: {0}")]
    BadScorecard(String),
    /// A task-data path lives under a LocalPilot checkout.
    #[error(
        "refusing external corpus path under a LocalPilot checkout (clean-room boundary): \
         '{0}'. External benchmark data must live in a user-local cache outside any \
         LocalPilot tree."
    )]
    CorpusInsideLocalPilot(String),
    /// The baseline arm failed isolation, so a delta would not grade the harness.
    #[error(
        "refusing a capability delta: the baseline arm failed isolation ({0}). A \
         contaminated baseline secretly runs the harness, so the delta does not grade \
         it. Fix the baseline's effective config first."
    )]
    ContaminatedBaselineDelta(String),
    /// One or more instruments failed their self-test.
    #[error(
        "instruments broken — refusing to spend on a model run: {0}. Each scorer/guard \
         must pass its good/bad self-test first."
    )]
    InstrumentsBroken(String),
    /// Cell persistence / rescore I/O failed.
    #[error("cell store: {0}")]
    Io(#[from] std::io::Error),
}

/// A capability scorecard normalized to the fields the report reads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParsedCard {
    pub task: String,
    pub arm: String,
    pub model: String,
    pub passed: bool,
    pub partial: f64,
    pub regression_safe: bool,
    pub diff_added: u32,
    pub diff_removed: u32,
    pub vs_gold: Option<f64>,
    pub tool_calls: u32,
    pub redundant: u32,
    pub retrieval: bool,
    pub exit_reason: String,
    pub wall_ms: u64,
    pub judge_overall: Option<f64>,
    /// Coach interventions during the run; `0` for undriven cells and for
    /// cells persisted before the field existed.
    #[serde(default)]
    pub interventions: u32,
}

/// Parse a capability scorecard (the shared wire contract) into its normalized
/// row. Pure and mock-testable: feed it captured JSON, no solver run needed.
///
/// # Errors
/// Returns [`RunnerError::BadScorecard`] on invalid JSON, a missing layer, or
/// an unsupported schema.
pub fn parse_capability_scorecard(json: &str) -> Result<ParsedCard, RunnerError> {
    let card: Scorecard =
        serde_json::from_str(json).map_err(|e| RunnerError::BadScorecard(e.to_string()))?;
    if card.schema != 1 {
        return Err(RunnerError::BadScorecard(format!(
            "unsupported capability scorecard schema {} (expected 1)",
            card.schema
        )));
    }
    Ok(ParsedCard {
        task: card.task,
        arm: card.arm,
        model: card.model,
        passed: card.results.passed,
        partial: card.results.partial_credit,
        regression_safe: card.results.regression_safe,
        diff_added: card.quality.diff_added,
        diff_removed: card.quality.diff_removed,
        vs_gold: card.quality.vs_gold_ratio,
        tool_calls: card.process.tool_calls,
        redundant: card.process.redundant_calls,
        retrieval: card.process.retrieval_used,
        exit_reason: card.process.exit_reason,
        wall_ms: card.speed.wall_ms,
        judge_overall: card.judge.map(|j| j.overall),
        interventions: card.process.interventions,
    })
}

/// True when no path segment is a `LocalPilot` checkout — the boundary that
/// keeps a copied public corpus out of the clean-room tree.
#[must_use]
pub fn path_outside_localpilot(path: &str) -> bool {
    !path
        .replace('\\', "/")
        .split('/')
        .any(|segment| segment == "LocalPilot")
}

/// Refuse a task-data path that lives under a LocalPilot checkout.
///
/// # Errors
/// Returns [`RunnerError::CorpusInsideLocalPilot`].
pub fn assert_external_corpus_path(path: &str) -> Result<(), RunnerError> {
    if path_outside_localpilot(path) {
        Ok(())
    } else {
        Err(RunnerError::CorpusInsideLocalPilot(path.to_string()))
    }
}

/// One arm's inputs to the comparative report.
#[derive(Debug, Clone)]
pub struct ArmCards {
    pub arm: String,
    pub model: String,
    pub cards: Vec<ParsedCard>,
    /// The arm's raw config, when the runner tracked isolation.
    pub arm_config: Option<RawArmConfig>,
}

/// One arm's row in the comparative report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArmReportRow {
    pub arm: String,
    pub model: String,
    pub tasks: usize,
    pub solved: usize,
    pub solve_rate: f64,
    pub avg_tool_calls: f64,
    pub avg_redundant: f64,
    pub avg_diff_added: f64,
    /// Mean coach interventions per task; `0.0` on undriven arms (and on
    /// rows persisted before the field existed).
    #[serde(default)]
    pub avg_interventions: f64,
    pub judge_overall: Option<f64>,
    /// Isolation provenance (`clean` / `CONTAMINATED: …` / `n/a (harness
    /// arm)`), when the runner tracked it.
    pub isolation: Option<String>,
}

/// The comparative capability report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapabilityReport {
    pub schema: u32,
    /// `first-party` or `external`.
    pub corpus: String,
    /// Public corpora are contamination-suspect: read numbers as deltas.
    pub contamination_suspect: bool,
    pub arms: Vec<ArmReportRow>,
}

fn avg(cards: &[ParsedCard], field: impl Fn(&ParsedCard) -> f64) -> f64 {
    if cards.is_empty() {
        return 0.0;
    }
    cards.iter().map(field).sum::<f64>() / cards.len() as f64
}

/// Build the comparative report from per-arm results.
#[must_use]
pub fn capability_report(results: &[ArmCards], corpus: &str) -> CapabilityReport {
    let arms = results
        .iter()
        .map(|result| {
            let n = result.cards.len().max(1);
            let solved = result.cards.iter().filter(|c| c.passed).count();
            let judged: Vec<f64> = result
                .cards
                .iter()
                .filter_map(|c| c.judge_overall)
                .collect();
            let isolation = result
                .arm_config
                .as_ref()
                .map(|config| arm_isolation_summary(&arm_effective_config(&result.arm, config)));
            ArmReportRow {
                arm: result.arm.clone(),
                model: result.model.clone(),
                tasks: result.cards.len(),
                solved,
                solve_rate: round_dp(solved as f64 / n as f64, 4),
                avg_tool_calls: round_dp(avg(&result.cards, |c| f64::from(c.tool_calls)), 3),
                avg_redundant: round_dp(avg(&result.cards, |c| f64::from(c.redundant)), 3),
                avg_diff_added: round_dp(avg(&result.cards, |c| f64::from(c.diff_added)), 3),
                avg_interventions: round_dp(avg(&result.cards, |c| f64::from(c.interventions)), 3),
                judge_overall: if judged.is_empty() {
                    None
                } else {
                    Some(round_dp(
                        judged.iter().sum::<f64>() / judged.len() as f64,
                        3,
                    ))
                },
                isolation,
            }
        })
        .collect();
    CapabilityReport {
        schema: 1,
        corpus: corpus.to_string(),
        contamination_suspect: corpus == "external",
        arms,
    }
}

/// The verdict on one delta row under the anti-gaming paired-metric invariant.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DeltaVerdict {
    Improvement,
    Neutral,
    Regression,
    /// A gameable-metric "win" with correctness regressed — doing less is not
    /// a win.
    NotAWin,
    /// A gameable-metric "win" with no correctness signal to pair against.
    Unverified,
}

/// One per-metric delta between a baseline row and an arm row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeltaRow {
    pub metric: String,
    pub baseline: f64,
    pub arm: f64,
    pub delta: f64,
    /// Whether the metric is gameable by under-delivery.
    pub gameable: bool,
    pub verdict: DeltaVerdict,
}

/// Metrics where "better" means higher.
const HIGHER_IS_BETTER: &[&str] = &["solveRate", "judgeOverall"];
/// Metrics gameable by under-delivery: a "better" value can just mean "did
/// less", so a win is gated on the correctness signal. Fewer coach
/// interventions counts as a win only when correctness held — a coach giving
/// up on a failing run also intervenes less.
const GAMEABLE_METRICS: &[&str] = &[
    "avgToolCalls",
    "avgRedundant",
    "avgDiffAdded",
    "avgInterventions",
];

/// Per-metric deltas between a baseline arm row and another arm row, each
/// annotated with the paired-metric verdict: a gameable-metric improvement is
/// `improvement` only when solve rate is present and did not regress; it is
/// `not-a-win` when correctness dropped, and `unverified` when there is no
/// correctness signal to pair against — never silently an improvement.
///
/// # Errors
/// Returns [`RunnerError::ContaminatedBaselineDelta`] when the baseline row
/// carries a `CONTAMINATED` isolation tag — a delta against it is meaningless.
pub fn arm_delta_verdicts(
    baseline: &ArmReportRow,
    arm: &ArmReportRow,
    correctness_tolerance: f64,
) -> Result<Vec<DeltaRow>, RunnerError> {
    if let Some(isolation) = &baseline.isolation {
        if isolation.starts_with("CONTAMINATED") {
            return Err(RunnerError::ContaminatedBaselineDelta(isolation.clone()));
        }
    }

    let metrics: [(&str, Option<f64>, Option<f64>); 6] = [
        ("solveRate", Some(baseline.solve_rate), Some(arm.solve_rate)),
        (
            "avgToolCalls",
            Some(baseline.avg_tool_calls),
            Some(arm.avg_tool_calls),
        ),
        (
            "avgRedundant",
            Some(baseline.avg_redundant),
            Some(arm.avg_redundant),
        ),
        (
            "avgDiffAdded",
            Some(baseline.avg_diff_added),
            Some(arm.avg_diff_added),
        ),
        (
            "avgInterventions",
            Some(baseline.avg_interventions),
            Some(arm.avg_interventions),
        ),
        ("judgeOverall", baseline.judge_overall, arm.judge_overall),
    ];

    let correctness_regressed = (arm.solve_rate - baseline.solve_rate) < -correctness_tolerance;

    Ok(metrics
        .into_iter()
        .filter_map(|(metric, b, a)| {
            let (b, a) = (b?, a?);
            let delta = a - b;
            let improved = if HIGHER_IS_BETTER.contains(&metric) {
                delta > 0.0
            } else {
                delta < 0.0
            };
            let gameable = GAMEABLE_METRICS.contains(&metric);
            let verdict = if !improved {
                if delta == 0.0 {
                    DeltaVerdict::Neutral
                } else {
                    DeltaVerdict::Regression
                }
            } else if gameable && correctness_regressed {
                DeltaVerdict::NotAWin
            } else {
                DeltaVerdict::Improvement
            };
            Some(DeltaRow {
                metric: metric.to_string(),
                baseline: b,
                arm: a,
                delta,
                gameable,
                verdict,
            })
        })
        .collect())
}

/// One instrument self-test outcome.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstrumentCheck {
    pub name: String,
    pub ok: bool,
}

/// A known-good capability scorecard for the parser self-test.
const INSTRUMENT_SAMPLE_CARD: &str = r#"{"schema":1,"task":"instr-1","arm":"full","model":"local","results":{"passed":true,"regression_safe":true,"partial_credit":1.0,"tests_total":1,"tests_passed":1},"quality":{"diff_added":3,"diff_removed":0,"diff_files":1,"vs_gold_ratio":1.0,"format_clean":true,"lint_clean":true,"typecheck_clean":true,"complexity_delta":0,"tests_added":false},"process":{"tool_calls":2,"redundant_calls":0,"reproduce_before_fix":true,"test_before_done":true,"retrieval_used":false,"retrieval_count":0,"exit_reason":"Done","recovered_after_failure":false,"discipline":null},"speed":{"wall_ms":100,"input_tokens":10,"output_tokens":5},"judge":null}"#;

/// Exercise every deterministic instrument on a known-good AND a known-bad
/// reference; an instrument is `ok` only when the good passes and the bad is
/// caught. A broken instrument blocks the run rather than silently producing
/// a wrong measurement.
#[must_use]
pub fn instrument_checks() -> Vec<InstrumentCheck> {
    let mut checks = Vec::new();

    // Scorecard parser: parses the sample, rejects a wrong-schema card.
    let good = parse_capability_scorecard(INSTRUMENT_SAMPLE_CARD)
        .map(|c| c.task == "instr-1")
        .unwrap_or(false);
    let bad = parse_capability_scorecard(r#"{"schema":2}"#).is_err();
    checks.push(InstrumentCheck {
        name: "scorecard-parser".to_string(),
        ok: good && bad,
    });

    // Clean-room boundary: an outside path passes, an inside path is refused.
    let good = path_outside_localpilot("C:/Users/x/.localbench/external-corpus/s");
    let bad = assert_external_corpus_path("D:/repos/LocalX/LocalPilot/x").is_err();
    checks.push(InstrumentCheck {
        name: "clean-room-boundary".to_string(),
        ok: good && bad,
    });

    // Arm isolation: a clean baseline passes, a retrieval-on baseline is refused.
    let good = assert_arm_isolation("baseline", &RawArmConfig::default()).is_ok();
    let bad = assert_arm_isolation(
        "baseline",
        &RawArmConfig {
            retrieval: true,
            ..RawArmConfig::default()
        },
    )
    .is_err();
    checks.push(InstrumentCheck {
        name: "arm-isolation".to_string(),
        ok: good && bad,
    });

    // Paired-metric verdict: a real win reads improvement; a gamed win is caught.
    let row = |solve: f64, diff: f64| ArmReportRow {
        arm: "x".to_string(),
        model: "m".to_string(),
        tasks: 10,
        solved: (solve * 10.0) as usize,
        solve_rate: solve,
        avg_tool_calls: 5.0,
        avg_redundant: 1.0,
        avg_diff_added: diff,
        avg_interventions: 0.0,
        judge_overall: None,
        isolation: None,
    };
    let verdict_of = |baseline: &ArmReportRow, arm: &ArmReportRow| {
        arm_delta_verdicts(baseline, arm, 0.0)
            .ok()
            .and_then(|rows| rows.into_iter().find(|r| r.metric == "avgDiffAdded"))
            .map(|r| r.verdict)
    };
    let win = verdict_of(&row(0.4, 30.0), &row(0.7, 18.0));
    let gamed = verdict_of(&row(0.7, 30.0), &row(0.4, 12.0));
    checks.push(InstrumentCheck {
        name: "paired-metric-verdict".to_string(),
        ok: win == Some(DeltaVerdict::Improvement) && gamed == Some(DeltaVerdict::NotAWin),
    });

    checks
}

/// Whether every instrument check passed.
#[must_use]
pub fn instruments_ready(checks: &[InstrumentCheck]) -> bool {
    checks.iter().all(|c| c.ok)
}

/// Refuse a model run when any instrument failed its self-test, naming the
/// broken ones — "instruments broken, refuse to spend".
///
/// # Errors
/// Returns [`RunnerError::InstrumentsBroken`] naming the failed instruments.
pub fn assert_instruments_ready(checks: &[InstrumentCheck]) -> Result<(), RunnerError> {
    let broken: Vec<&str> = checks
        .iter()
        .filter(|c| !c.ok)
        .map(|c| c.name.as_str())
        .collect();
    if broken.is_empty() {
        Ok(())
    } else {
        Err(RunnerError::InstrumentsBroken(broken.join(", ")))
    }
}

/// One persisted run cell: the raw scorecard plus arm metadata, preserved so
/// the comparative report can be recomputed offline after a metric/scorer
/// change — without re-running the solver.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cell {
    pub task: String,
    pub arm: String,
    pub model: String,
    #[serde(default)]
    pub arm_config: Option<RawArmConfig>,
    /// The raw scorecard JSON exactly as emitted.
    pub scorecard: String,
}

fn safe_name(part: &str) -> String {
    part.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Persist one run cell as JSON under `dir`. Returns the written path.
///
/// # Errors
/// Returns the I/O error when the cell cannot be written.
pub fn save_cell(dir: &Path, cell: &Cell) -> Result<PathBuf, RunnerError> {
    std::fs::create_dir_all(dir)?;
    let name = format!(
        "{}__{}__{}.json",
        safe_name(&cell.task),
        safe_name(&cell.arm),
        safe_name(&cell.model)
    );
    let path = dir.join(name);
    let json = serde_json::to_string_pretty(cell)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    std::fs::write(&path, json)?;
    Ok(path)
}

/// Recompute the comparative report from kept cells — no solver run, no model.
/// Cells are read in sorted filename order and grouped by arm (the first cell
/// carrying a config supplies the arm's isolation provenance), so the same
/// cell set always yields the identical report.
///
/// # Errors
/// Returns [`RunnerError::Io`] when the directory cannot be read, or
/// [`RunnerError::BadScorecard`] when a kept cell no longer parses.
pub fn rescore(dir: &Path, corpus: &str) -> Result<CapabilityReport, RunnerError> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "json"))
        .collect();
    files.sort();

    let mut by_arm: BTreeMap<String, ArmCards> = BTreeMap::new();
    for file in files {
        let raw = std::fs::read_to_string(&file)?;
        let cell: Cell = serde_json::from_str(&raw)
            .map_err(|e| RunnerError::BadScorecard(format!("{}: {e}", file.display())))?;
        let card = parse_capability_scorecard(&cell.scorecard)?;
        let entry = by_arm.entry(cell.arm.clone()).or_insert_with(|| ArmCards {
            arm: cell.arm.clone(),
            model: cell.model.clone(),
            cards: Vec::new(),
            arm_config: None,
        });
        entry.cards.push(card);
        if entry.arm_config.is_none() {
            entry.arm_config = cell.arm_config;
        }
    }
    let results: Vec<ArmCards> = by_arm.into_values().collect();
    Ok(capability_report(&results, corpus))
}

/// Annotate a graded cell's note, keeping a wall-clock timeout a **caveat**
/// rather than a capability result: a timed-out cell stays unsolved (a floor)
/// but is never stamped `0 tests ran (not solved)` as if the solve itself
/// proved empty.
#[must_use]
pub fn annotate_cell_note(tests_run: u32, grade_timed_out: bool, base_note: &str) -> String {
    if tests_run == 0 && !grade_timed_out {
        if base_note.is_empty() {
            "0 tests ran (not solved)".to_string()
        } else {
            format!("0 tests ran (not solved) | {base_note}")
        }
    } else {
        base_note.to_string()
    }
}

/// The fairness caveat for wall-clock-budget cells: they count unsolved, so
/// the arm's rate reads as a floor, never a pure capability number.
#[must_use]
pub fn timeout_caveat(arm: &str, timed_out_cells: usize) -> Option<String> {
    if timed_out_cells == 0 {
        return None;
    }
    Some(format!(
        "caveat: {timed_out_cells} {arm} cell(s) hit the wall-clock budget (counted unsolved) — \
         the {arm} rate is a floor"
    ))
}

/// The infrastructure-gap caveat for cells whose grade could not actually run —
/// an under-vendored offline cargo cache, say. Like a timeout, such a cell
/// counts unsolved (a floor), but the cause is the grade environment, not the
/// solver, so the fix is to re-warm the cache and rescore rather than to read a
/// deflated rate as a capability result.
#[must_use]
pub fn infra_gap_caveat(arm: &str, infra_gap_cells: usize) -> Option<String> {
    if infra_gap_cells == 0 {
        return None;
    }
    Some(format!(
        "caveat: {infra_gap_cells} {arm} cell(s) hit an infrastructure gap (the grade could not \
         run — likely an under-vendored offline cargo cache; counted unsolved) — the {arm} rate \
         is a floor; re-warm the cache and rescore"
    ))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn card_json(task: &str, arm: &str, passed: bool, tool_calls: u32, diff_added: u32) -> String {
        format!(
            r#"{{"schema":1,"task":"{task}","arm":"{arm}","model":"apex",
                "results":{{"passed":{passed},"regression_safe":true,"partial_credit":1.0,"tests_total":1,"tests_passed":1}},
                "quality":{{"diff_added":{diff_added},"diff_removed":0,"diff_files":1,"vs_gold_ratio":null,"format_clean":true,"lint_clean":true,"typecheck_clean":true,"complexity_delta":null,"tests_added":false}},
                "process":{{"tool_calls":{tool_calls},"redundant_calls":0,"reproduce_before_fix":true,"test_before_done":true,"retrieval_used":false,"retrieval_count":0,"exit_reason":"Done","recovered_after_failure":false,"discipline":null}},
                "speed":{{"wall_ms":100,"input_tokens":1,"output_tokens":1}},"judge":null}}"#
        )
    }

    #[test]
    fn scorecard_parses_and_rejects_bad_schema_or_missing_layers() {
        let card = parse_capability_scorecard(&card_json("t1", "full", true, 3, 10)).unwrap();
        assert_eq!(card.task, "t1");
        assert!(card.passed);
        assert_eq!(card.tool_calls, 3);
        assert!(matches!(
            parse_capability_scorecard(r#"{"schema":2}"#),
            Err(RunnerError::BadScorecard(_))
        ));
        assert!(parse_capability_scorecard("not json").is_err());
    }

    #[test]
    fn the_clean_room_path_boundary_holds() {
        assert!(path_outside_localpilot(
            "C:/Users/x/.localbench/corpus/task"
        ));
        assert!(!path_outside_localpilot(r"D:\work\LocalPilot\x"));
        assert!(assert_external_corpus_path("/home/x/.localbench/c").is_ok());
        assert!(assert_external_corpus_path("/repos/LocalPilot/data").is_err());
    }

    fn arm_cards(arm: &str, passes: &[bool]) -> ArmCards {
        ArmCards {
            arm: arm.to_string(),
            model: "apex".to_string(),
            cards: passes
                .iter()
                .enumerate()
                .map(|(i, passed)| {
                    parse_capability_scorecard(&card_json(
                        &format!("t{i}"),
                        arm,
                        *passed,
                        4 + i as u32,
                        10,
                    ))
                    .unwrap()
                })
                .collect(),
            arm_config: Some(RawArmConfig::default()),
        }
    }

    #[test]
    fn the_report_aggregates_per_arm_with_isolation_provenance() {
        let report = capability_report(
            &[
                arm_cards("baseline", &[true, false]),
                arm_cards("full", &[true, true]),
            ],
            "external",
        );
        assert!(report.contamination_suspect);
        let baseline = report.arms.iter().find(|a| a.arm == "baseline").unwrap();
        assert_eq!(baseline.solve_rate, 0.5);
        assert_eq!(baseline.isolation.as_deref(), Some("clean"));
        let full = report.arms.iter().find(|a| a.arm == "full").unwrap();
        assert_eq!(full.solve_rate, 1.0);
        assert_eq!(full.isolation.as_deref(), Some("n/a (harness arm)"));
    }

    #[test]
    fn gameable_wins_require_a_correctness_signal_that_did_not_regress() {
        let row = |solve: f64, diff: f64| ArmReportRow {
            arm: "x".to_string(),
            model: "m".to_string(),
            tasks: 10,
            solved: 0,
            solve_rate: solve,
            avg_tool_calls: 5.0,
            avg_redundant: 1.0,
            avg_diff_added: diff,
            avg_interventions: 0.0,
            judge_overall: None,
            isolation: None,
        };
        // Correctness up + smaller diff: a real improvement.
        let rows = arm_delta_verdicts(&row(0.4, 30.0), &row(0.7, 18.0), 0.0).unwrap();
        let diff = rows.iter().find(|r| r.metric == "avgDiffAdded").unwrap();
        assert_eq!(diff.verdict, DeltaVerdict::Improvement);
        assert!(diff.gameable);
        // Smaller diff with correctness DROPPED: not a win.
        let rows = arm_delta_verdicts(&row(0.7, 30.0), &row(0.4, 12.0), 0.0).unwrap();
        let diff = rows.iter().find(|r| r.metric == "avgDiffAdded").unwrap();
        assert_eq!(diff.verdict, DeltaVerdict::NotAWin);
        // The correctness row itself reads regression, not gameable.
        let solve = rows.iter().find(|r| r.metric == "solveRate").unwrap();
        assert_eq!(solve.verdict, DeltaVerdict::Regression);
        assert!(!solve.gameable);
    }

    #[test]
    fn a_delta_against_a_contaminated_baseline_is_refused() {
        let mut baseline = ArmReportRow {
            arm: "baseline".to_string(),
            model: "m".to_string(),
            tasks: 1,
            solved: 1,
            solve_rate: 1.0,
            avg_tool_calls: 0.0,
            avg_redundant: 0.0,
            avg_diff_added: 0.0,
            avg_interventions: 0.0,
            judge_overall: None,
            isolation: Some("CONTAMINATED: retrieval".to_string()),
        };
        let arm = baseline.clone();
        assert!(matches!(
            arm_delta_verdicts(&baseline, &arm, 0.0),
            Err(RunnerError::ContaminatedBaselineDelta(_))
        ));
        baseline.isolation = Some("clean".to_string());
        assert!(arm_delta_verdicts(&baseline, &arm, 0.0).is_ok());
    }

    #[test]
    fn every_instrument_passes_its_self_test() {
        let checks = instrument_checks();
        assert_eq!(checks.len(), 4);
        assert!(instruments_ready(&checks), "checks: {checks:?}");
        assert!(assert_instruments_ready(&checks).is_ok());
        // A broken instrument blocks the run, named.
        let mut broken = checks;
        broken[0].ok = false;
        let err = assert_instruments_ready(&broken).unwrap_err();
        assert!(err.to_string().contains("scorecard-parser"));
    }

    #[test]
    fn cells_round_trip_and_rescore_is_deterministic() {
        let dir = tempfile::tempdir().unwrap();
        for (task, arm, passed) in [
            ("t1", "baseline", false),
            ("t2", "baseline", true),
            ("t1", "full", true),
            ("t2", "full", true),
        ] {
            save_cell(
                dir.path(),
                &Cell {
                    task: task.to_string(),
                    arm: arm.to_string(),
                    model: "apex".to_string(),
                    arm_config: Some(RawArmConfig::default()),
                    scorecard: card_json(task, arm, passed, 3, 10),
                },
            )
            .unwrap();
        }
        let first = rescore(dir.path(), "external").unwrap();
        let second = rescore(dir.path(), "external").unwrap();
        assert_eq!(first, second, "identical cells → identical report");
        let baseline = first.arms.iter().find(|a| a.arm == "baseline").unwrap();
        assert_eq!(baseline.solve_rate, 0.5);
        assert_eq!(baseline.isolation.as_deref(), Some("clean"));
        let full = first.arms.iter().find(|a| a.arm == "full").unwrap();
        assert_eq!(full.solve_rate, 1.0);
    }

    #[test]
    fn cell_filenames_are_sanitized() {
        let dir = tempfile::tempdir().unwrap();
        let path = save_cell(
            dir.path(),
            &Cell {
                task: "astropy/astropy:12907".to_string(),
                arm: "full".to_string(),
                model: "apex i".to_string(),
                arm_config: None,
                scorecard: card_json("t", "full", true, 1, 1),
            },
        )
        .unwrap();
        let name = path.file_name().unwrap().to_str().unwrap();
        assert_eq!(name, "astropy_astropy_12907__full__apex_i.json");
    }

    #[test]
    fn a_timeout_is_a_caveat_not_a_capability_result() {
        // A genuine zero-test grade is stamped unsolved.
        assert_eq!(
            annotate_cell_note(0, false, "exit: 0"),
            "0 tests ran (not solved) | exit: 0"
        );
        // A timed-out grade keeps its own note — unsolved, but a floor.
        assert_eq!(
            annotate_cell_note(0, true, "grade timed out after 900s"),
            "grade timed out after 900s"
        );
        assert_eq!(annotate_cell_note(4, false, "ok"), "ok");
        // The rendered caveat names the arm and the floor semantics.
        let caveat = timeout_caveat("claude-code", 14).unwrap();
        assert!(caveat.contains("14 claude-code cell(s)"));
        assert!(caveat.contains("floor"));
        assert!(timeout_caveat("full", 0).is_none());
    }

    #[test]
    fn an_infra_gap_is_a_caveat_naming_the_environment_not_the_solver() {
        let caveat = infra_gap_caveat("full", 3).unwrap();
        assert!(caveat.contains("3 full cell(s)"));
        assert!(caveat.contains("infrastructure gap"));
        // The caveat points at the grade environment (re-warm), never a solve.
        assert!(caveat.contains("re-warm the cache"));
        assert!(caveat.contains("floor"));
        assert!(infra_gap_caveat("full", 0).is_none());
    }
}
