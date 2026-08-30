//! Lesson-uplift A/B statistics: the deterministic answer grader, trial-level
//! aggregation, the pooled-stddev significance signal, and the injection
//! contract that voids a result when an arm did not inject as configured.
//!
//! The signal is always significance, never a bare delta: a delta must clear
//! the noise band (pooled per-arm stddev, floored) to count as uplift or
//! regression; otherwise it is honestly "no effect (within noise)".

use regex::RegexBuilder;
use serde::{Deserialize, Serialize};

use crate::stats::{round_dp, sample_std_dev};

/// The default significance floor: a delta smaller than this never counts as
/// an effect, however tight the measured bands.
pub const SIGNIFICANCE_FLOOR: f64 = 0.05;

/// What a task's answer is graded against. No LLM judge — a normalized
/// substring / all-substrings / regex match, so the grader is deterministic
/// and self-testable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", content = "value", rename_all = "kebab-case")]
pub enum Expect {
    /// The normalized answer contains the normalized needle.
    Substring(String),
    /// The normalized answer contains every normalized needle.
    AllSubstrings(Vec<String>),
    /// The raw answer matches the regex (case-insensitive unless case-sensitive
    /// grading is on).
    Regex(String),
}

/// Grade an answer against an expectation. Whitespace is collapsed and
/// matching is case-insensitive unless `case_sensitive` is set.
#[must_use]
pub fn grade_answer(answer: &str, expect: &Expect, case_sensitive: bool) -> bool {
    let norm = |text: &str| -> String {
        let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
        if case_sensitive {
            collapsed
        } else {
            collapsed.to_lowercase()
        }
    };
    let haystack = norm(answer);

    match expect {
        Expect::Substring(value) => {
            let needle = norm(value);
            !needle.is_empty() && haystack.contains(&needle)
        }
        Expect::AllSubstrings(values) => {
            let needles: Vec<String> = values
                .iter()
                .map(|v| norm(v))
                .filter(|n| !n.is_empty())
                .collect();
            !needles.is_empty() && needles.iter().all(|n| haystack.contains(n))
        }
        Expect::Regex(pattern) => RegexBuilder::new(pattern)
            .case_insensitive(!case_sensitive)
            .build()
            .is_ok_and(|re| re.is_match(answer)),
    }
}

/// One memory recorded as used during a trial.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryUsed {
    pub id: String,
}

/// One task's per-trial outcomes within an arm.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskResult {
    /// The task's pass/fail per trial (index = trial ordinal).
    pub passes: Vec<bool>,
    /// The memories recorded as used per trial.
    pub memories_used: Vec<Vec<MemoryUsed>>,
}

/// One arm's results across its tasks and trials.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArmResult {
    pub arm: String,
    /// Whether this arm injects lessons (vs. the clean baseline).
    pub is_lesson_arm: bool,
    pub trials: u32,
    pub tasks: Vec<TaskResult>,
}

/// An uplift-statistics failure.
#[derive(Debug, thiserror::Error)]
pub enum UpliftError {
    /// An arm carried no tasks, so no rate is computable.
    #[error("uplift aggregate: arm '{0}' has no tasks")]
    NoTasks(String),
    /// An arm did not inject as configured; its result is void.
    #[error("injection assertion VOID: {0}")]
    InjectionVoid(String),
}

/// One arm's aggregate: the per-trial success rate (fraction of tasks passed
/// in that trial), then mean ± stddev across trials. Trial-level (not
/// task-level) variance is what the significance signal needs, so a noisy
/// model reads as wide bands, not a point.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Aggregate {
    pub arm: String,
    pub is_lesson_arm: bool,
    pub trials: u32,
    pub task_count: usize,
    pub per_trial_success_rate: Vec<f64>,
    pub mean: f64,
    pub stddev: f64,
}

/// Aggregate one arm's per-task/per-trial grades into the arm-level metric.
///
/// # Errors
/// Returns [`UpliftError::NoTasks`] when the arm has no tasks.
pub fn aggregate(arm: &ArmResult) -> Result<Aggregate, UpliftError> {
    if arm.tasks.is_empty() {
        return Err(UpliftError::NoTasks(arm.arm.clone()));
    }
    let per_trial: Vec<f64> = (0..arm.trials as usize)
        .map(|t| {
            let passed = arm
                .tasks
                .iter()
                .filter(|task| task.passes.get(t).copied().unwrap_or(false))
                .count();
            round_dp(passed as f64 / arm.tasks.len() as f64, 4)
        })
        .collect();
    let mean = if per_trial.is_empty() {
        0.0
    } else {
        per_trial.iter().sum::<f64>() / per_trial.len() as f64
    };
    Ok(Aggregate {
        arm: arm.arm.clone(),
        is_lesson_arm: arm.is_lesson_arm,
        trials: arm.trials,
        task_count: arm.tasks.len(),
        per_trial_success_rate: per_trial.clone(),
        mean: round_dp(mean, 4),
        stddev: round_dp(sample_std_dev(&per_trial), 4),
    })
}

/// The significance verdict for a baseline-vs-lesson comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Verdict {
    Uplift,
    Regression,
    /// The delta is within the noise band.
    NoEffect,
}

/// The significance signal between a baseline and a lesson aggregate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Significance {
    pub baseline_arm: String,
    pub lesson_arm: String,
    pub baseline_mean: f64,
    pub lesson_mean: f64,
    pub delta: f64,
    pub pooled_std_dev: f64,
    /// The noise band the delta must clear: `max(pooled stddev, floor)`.
    pub band: f64,
    /// `delta / pooled stddev`; `None` when the pooled stddev is zero.
    pub effect_size: Option<f64>,
    pub verdict: Verdict,
}

/// Turn a baseline vs lesson aggregate into a significance signal, never a
/// bare delta. The verdict comes from the delta against the pooled per-arm
/// stddev band (a conservative non-overlap proxy for a small trial count).
#[must_use]
pub fn significance(baseline: &Aggregate, lesson: &Aggregate, floor: f64) -> Significance {
    let delta = lesson.mean - baseline.mean;
    let pooled = ((baseline.stddev * baseline.stddev + lesson.stddev * lesson.stddev) / 2.0).sqrt();
    let band = pooled.max(floor);
    let effect_size = if pooled > 0.0 {
        Some(round_dp(delta / pooled, 3))
    } else {
        None
    };
    let verdict = if delta > band {
        Verdict::Uplift
    } else if delta < -band {
        Verdict::Regression
    } else {
        Verdict::NoEffect
    };
    Significance {
        baseline_arm: baseline.arm.clone(),
        lesson_arm: lesson.arm.clone(),
        baseline_mean: baseline.mean,
        lesson_mean: lesson.mean,
        delta: round_dp(delta, 4),
        pooled_std_dev: round_dp(pooled, 4),
        band: round_dp(band, 4),
        effect_size,
        verdict,
    }
}

/// The injection summary returned when an arm injected as configured.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InjectionSummary {
    pub arm: String,
    pub is_lesson_arm: bool,
    pub intended: Vec<String>,
    pub injected: Vec<String>,
}

/// Enforce the injection contract, so a result is VOID if the arm did not
/// inject as configured: a lesson arm must record, in EVERY trial, at least
/// one of its intended memory ids; a baseline arm must record NO memories in
/// any trial.
///
/// # Errors
/// Returns [`UpliftError::InjectionVoid`] naming the arm and trial on any
/// violation (including a lesson arm with no intended ids to verify against).
pub fn assert_injection(
    arm: &ArmResult,
    intended_ids: &[String],
) -> Result<InjectionSummary, UpliftError> {
    let used_per_trial: Vec<Vec<String>> = (0..arm.trials as usize)
        .map(|t| {
            let mut ids: Vec<String> = arm
                .tasks
                .iter()
                .flat_map(|task| task.memories_used.get(t).into_iter().flatten())
                .map(|m| m.id.clone())
                .collect();
            ids.sort();
            ids.dedup();
            ids
        })
        .collect();

    if arm.is_lesson_arm {
        if intended_ids.is_empty() {
            return Err(UpliftError::InjectionVoid(format!(
                "lesson arm '{}' has no intended ids to verify against.",
                arm.arm
            )));
        }
        for (t, used) in used_per_trial.iter().enumerate() {
            if !used.iter().any(|id| intended_ids.contains(id)) {
                return Err(UpliftError::InjectionVoid(format!(
                    "lesson arm '{}' trial {t} injected none of the intended lessons ({}). \
                     The arm did not inject as configured; the result is void.",
                    arm.arm,
                    intended_ids.join(", ")
                )));
            }
        }
    } else {
        for (t, used) in used_per_trial.iter().enumerate() {
            if !used.is_empty() {
                return Err(UpliftError::InjectionVoid(format!(
                    "baseline arm '{}' trial {t} recorded memories ({}). \
                     A baseline must inject nothing; the result is void.",
                    arm.arm,
                    used.join(", ")
                )));
            }
        }
    }

    let mut injected: Vec<String> = used_per_trial.into_iter().flatten().collect();
    injected.sort();
    injected.dedup();
    Ok(InjectionSummary {
        arm: arm.arm.clone(),
        is_lesson_arm: arm.is_lesson_arm,
        intended: intended_ids.to_vec(),
        injected,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn grader_normalizes_whitespace_and_case() {
        let expect = Expect::Substring("hello world".to_string());
        assert!(grade_answer("Well...  HELLO\n\tWORLD!", &expect, false));
        assert!(!grade_answer("Well...  HELLO\n\tWORLD!", &expect, true));
        assert!(!grade_answer("hello", &expect, false));
        assert!(!grade_answer(
            "anything",
            &Expect::Substring(String::new()),
            false
        ));
    }

    #[test]
    fn grader_all_substrings_requires_every_needle() {
        let expect = Expect::AllSubstrings(vec!["alpha".to_string(), "beta".to_string()]);
        assert!(grade_answer("Beta comes after Alpha.", &expect, false));
        assert!(!grade_answer("only alpha here", &expect, false));
        assert!(!grade_answer(
            "anything",
            &Expect::AllSubstrings(vec![]),
            false
        ));
    }

    #[test]
    fn grader_regex_matches_the_raw_answer() {
        let expect = Expect::Regex(r"answer:\s*42\b".to_string());
        assert!(grade_answer("The Answer:  42 (obviously)", &expect, false));
        assert!(!grade_answer("The Answer:  42 (obviously)", &expect, true));
        assert!(!grade_answer("answer: 421", &expect, false));
    }

    fn arm(name: &str, lesson: bool, passes_per_task: &[&[bool]]) -> ArmResult {
        let trials = passes_per_task.first().map_or(0, |p| p.len()) as u32;
        ArmResult {
            arm: name.to_string(),
            is_lesson_arm: lesson,
            trials,
            tasks: passes_per_task
                .iter()
                .map(|passes| TaskResult {
                    passes: passes.to_vec(),
                    memories_used: vec![Vec::new(); passes.len()],
                })
                .collect(),
        }
    }

    #[test]
    fn aggregate_computes_per_trial_rates_then_mean_and_stddev() {
        // 4 tasks x 2 trials: trial 0 passes 2/4, trial 1 passes 4/4.
        let a = aggregate(&arm(
            "baseline",
            false,
            &[&[true, true], &[true, true], &[false, true], &[false, true]],
        ))
        .expect("aggregate");
        assert_eq!(a.per_trial_success_rate, vec![0.5, 1.0]);
        assert_eq!(a.mean, 0.75);
        assert!((a.stddev - round_dp(sample_std_dev(&[0.5, 1.0]), 4)).abs() < 1e-12);
        assert_eq!(a.task_count, 4);
    }

    #[test]
    fn aggregate_refuses_an_empty_arm() {
        let empty = ArmResult {
            arm: "x".to_string(),
            is_lesson_arm: false,
            trials: 3,
            tasks: Vec::new(),
        };
        assert!(matches!(aggregate(&empty), Err(UpliftError::NoTasks(_))));
    }

    #[test]
    fn significance_verdicts_respect_the_noise_band() {
        let base = aggregate(&arm("baseline", false, &[&[false, false], &[false, false]]))
            .expect("aggregate");
        let lesson =
            aggregate(&arm("lesson", true, &[&[true, true], &[true, true]])).expect("aggregate");
        // 0% -> 100% with zero variance: a clear uplift.
        let sig = significance(&base, &lesson, SIGNIFICANCE_FLOOR);
        assert_eq!(sig.verdict, Verdict::Uplift);
        assert_eq!(sig.delta, 1.0);
        assert_eq!(
            sig.effect_size, None,
            "zero pooled stddev has no effect size"
        );

        // The reverse is a regression.
        let sig = significance(&lesson, &base, SIGNIFICANCE_FLOOR);
        assert_eq!(sig.verdict, Verdict::Regression);

        // A delta under the floor is no-effect even with tight bands.
        let close = aggregate(&arm("lesson2", true, &[&[false, false], &[false, true]]))
            .expect("aggregate");
        let sig = significance(&base, &close, 0.5);
        assert_eq!(sig.verdict, Verdict::NoEffect);
        assert_eq!(sig.band, 0.5, "the floor dominates a tighter pooled band");
    }

    #[test]
    fn lesson_arm_must_inject_an_intended_id_every_trial() {
        let mut a = arm("lesson", true, &[&[true, true]]);
        a.tasks[0].memories_used = vec![
            vec![MemoryUsed {
                id: "mem-1".to_string(),
            }],
            Vec::new(), // trial 1 injected nothing
        ];
        let intended = vec!["mem-1".to_string()];
        assert!(matches!(
            assert_injection(&a, &intended),
            Err(UpliftError::InjectionVoid(msg)) if msg.contains("trial 1")
        ));

        a.tasks[0].memories_used[1] = vec![MemoryUsed {
            id: "mem-1".to_string(),
        }];
        let summary = assert_injection(&a, &intended).expect("injects every trial");
        assert_eq!(summary.injected, vec!["mem-1".to_string()]);
    }

    #[test]
    fn lesson_arm_with_no_intended_ids_is_void() {
        let a = arm("lesson", true, &[&[true]]);
        assert!(matches!(
            assert_injection(&a, &[]),
            Err(UpliftError::InjectionVoid(_))
        ));
    }

    #[test]
    fn baseline_arm_must_inject_nothing() {
        let mut a = arm("baseline", false, &[&[true, true]]);
        assert!(assert_injection(&a, &[]).is_ok());
        a.tasks[0].memories_used[0] = vec![MemoryUsed {
            id: "leak-1".to_string(),
        }];
        assert!(matches!(
            assert_injection(&a, &[]),
            Err(UpliftError::InjectionVoid(msg)) if msg.contains("leak-1")
        ));
    }
}
