//! Candidate construction, beam selection, and cross-profile adoption.

use std::collections::BTreeMap;
use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use localbench_scoring::score::{
    balanced_score, trial_score, HostSignals, Optimize, Overrides, ScoreBreakdown, Trial,
    VramHeadroomParams, Workload,
};
use localbench_scoring::stats::round_dp;

use crate::overrides::candidate_signature;

/// Which score ranks candidates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ScoreProfile {
    /// Raw speed.
    Pure,
    /// Speed discounted by operational-risk factors.
    Balanced,
    /// Track both frontiers.
    Both,
}

/// The two rankable profiles (the `both` frontier resolves into these).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Profile {
    Pure,
    Balanced,
}

/// Everything candidate scoring needs, bundled so a phase can score many
/// candidates identically.
#[derive(Debug, Clone, Default)]
pub struct ScoringContext {
    pub workload: Workload,
    pub host: HostSignals,
    pub vram_params: VramHeadroomParams,
    /// Cross-phase stability CVs keyed by stability group key.
    pub stability_index: BTreeMap<String, f64>,
}

/// One scored candidate: a config, its measured trial, and its scores.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    pub overrides: Overrides,
    pub trial: Option<Trial>,
    /// Raw-speed score, rounded to 2dp.
    pub pure_score: f64,
    /// Risk-discounted score, rounded to 2dp.
    pub balanced_score: f64,
    /// The score the configured profile ranks by, rounded to 2dp.
    pub selected_score: f64,
    pub profile: ScoreProfile,
    pub phase: String,
    /// The canonical config signature (the dedup key).
    pub signature: String,
    pub score_breakdown: ScoreBreakdown,
}

/// Build a scored candidate from a config and its (optional) measured trial.
#[must_use]
pub fn new_candidate(
    overrides: &Overrides,
    trial: Option<&Trial>,
    profile: ScoreProfile,
    phase: &str,
    optimize: Optimize,
    ctx: &ScoringContext,
) -> Candidate {
    let pure = trial.map_or(0.0, |t| trial_score(t, optimize, &ctx.workload));
    let (balanced, breakdown) = match trial {
        Some(t) => {
            let b = balanced_score(
                t,
                overrides,
                pure,
                ctx.host,
                &ctx.vram_params,
                &ctx.stability_index,
            );
            (b.score, b.breakdown)
        }
        None => (0.0, ScoreBreakdown::default()),
    };
    let selected = match profile {
        ScoreProfile::Balanced => balanced,
        ScoreProfile::Both => pure.max(balanced),
        ScoreProfile::Pure => pure,
    };
    Candidate {
        overrides: overrides.clone(),
        trial: trial.cloned(),
        pure_score: round_dp(pure, 2),
        balanced_score: round_dp(balanced, 2),
        selected_score: round_dp(selected, 2),
        profile,
        phase: phase.to_string(),
        signature: candidate_signature(overrides),
        score_breakdown: breakdown,
    }
}

/// A candidate is *valid* (rankable) when it measured a healthy trial.
fn is_valid(candidate: &Candidate) -> bool {
    candidate
        .trial
        .as_ref()
        .is_some_and(Trial::is_measurement_usable)
}

/// The candidate's score under a rankable profile.
#[must_use]
pub fn profile_score(candidate: &Candidate, profile: Profile) -> f64 {
    match profile {
        Profile::Balanced => candidate.balanced_score,
        Profile::Pure => candidate.pure_score,
    }
}

/// Sort best-first by a score accessor, ties broken by ascending signature so
/// the order is total and reproducible.
fn sort_best_first(candidates: &mut [&Candidate], score: impl Fn(&Candidate) -> f64) {
    candidates.sort_by(|a, b| {
        score(b)
            .partial_cmp(&score(a))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.signature.cmp(&b.signature))
    });
}

/// Select the beam: dedup by signature (keeping the higher selected score),
/// drop OOM/failed trials, and keep the top `beam_width` per the profile. On
/// the `both` profile the beam is the union of the pure and balanced
/// frontiers, so neither ranking starves the other.
#[must_use]
pub fn select_beam(
    candidates: &[Candidate],
    beam_width: usize,
    profile: ScoreProfile,
) -> Vec<Candidate> {
    let beam_width = beam_width.max(1);

    // Dedup by signature, first-seen order, higher selected score wins.
    let mut order: Vec<&Candidate> = Vec::new();
    let mut index: HashMap<&str, usize> = HashMap::new();
    for candidate in candidates {
        match index.get(candidate.signature.as_str()) {
            Some(&pos) => {
                if candidate.selected_score > order[pos].selected_score {
                    order[pos] = candidate;
                }
            }
            None => {
                index.insert(candidate.signature.as_str(), order.len());
                order.push(candidate);
            }
        }
    }

    let mut valid: Vec<&Candidate> = order.into_iter().filter(|c| is_valid(c)).collect();

    if profile == ScoreProfile::Both {
        let mut frontier: Vec<&Candidate> = Vec::new();
        let mut seen: HashMap<&str, ()> = HashMap::new();
        for ranking in [Profile::Pure, Profile::Balanced] {
            let mut by_rank = valid.clone();
            sort_best_first(&mut by_rank, |c| profile_score(c, ranking));
            for candidate in by_rank.into_iter().take(beam_width) {
                if seen.insert(candidate.signature.as_str(), ()).is_none() {
                    frontier.push(candidate);
                }
            }
        }
        sort_best_first(&mut frontier, |c| c.pure_score.max(c.balanced_score));
        return frontier.into_iter().cloned().collect();
    }

    sort_best_first(&mut valid, |c| c.selected_score);
    valid.into_iter().take(beam_width).cloned().collect()
}

/// One profile's finished winner plus its soak validation, as the
/// cross-profile adoption reads it.
#[derive(Debug, Clone)]
pub struct ProfileWinner {
    pub winner: Candidate,
    /// Whether the winner passed its long-context soak; `None` means no soak
    /// verdict was recorded (treated as passed).
    pub soak_passed: Option<bool>,
}

/// A cross-profile adoption pick.
#[derive(Debug, Clone)]
pub struct CrossProfilePick {
    pub winner: Candidate,
    pub passed: bool,
    /// The pick's score under the adopting profile's metric.
    pub score: f64,
    pub source_profile: String,
}

/// Cross-profile validated-soak adoption.
///
/// Soak validity is a property of the config, not the score ranking. When a
/// profile's score-sorted scan cannot reach a soak-passing config within its
/// reserve (e.g. the top N are all VRAM-cliff candidates that gate-fail in a
/// correlated way), it can still adopt a sibling profile's already-validated
/// config and report it under its own scoring metric. **A passed-soak winner
/// beats a higher-scoring degraded one**; within a tier, the higher score
/// under this profile's metric wins. Returns `None` when nothing eligible is
/// available.
#[must_use]
pub fn cross_profile_winner(
    profile: Profile,
    winner_by_profile: &BTreeMap<String, ProfileWinner>,
    exclude_profile: &str,
) -> Option<CrossProfilePick> {
    let mut best: Option<CrossProfilePick> = None;
    for (source, entry) in winner_by_profile {
        if !exclude_profile.is_empty() && source == exclude_profile {
            continue;
        }
        let passed = entry.soak_passed.unwrap_or(true);
        let score = profile_score(&entry.winner, profile);
        let is_better = match &best {
            None => true,
            Some(current) => {
                (passed && !current.passed) || (passed == current.passed && score > current.score)
            }
        };
        if is_better {
            best = Some(CrossProfilePick {
                winner: entry.winner.clone(),
                passed,
                score,
                source_profile: source.clone(),
            });
        }
    }
    best
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::overrides::overrides_of;
    use localbench_scoring::score::Telemetry;

    fn trial(pp: f64, tg: f64, variance: f64, vram_free: f64, oom: bool) -> Trial {
        Trial {
            startup_ok: true,
            oom,
            measurement_usable: !oom,
            pp_tps: pp,
            tg_tps: tg,
            variance: Some(variance),
            telemetry: Telemetry {
                gpu_vram_free_gb_min: Some(vram_free),
                ..Telemetry::default()
            },
            ..Trial::default()
        }
    }

    fn ctx() -> ScoringContext {
        ScoringContext::default()
    }

    /// The canned regression quartet from the scoring pins.
    fn quartet() -> Vec<Candidate> {
        let comfortable = (
            overrides_of(&[("NCpuMoe", 10.into()), ("UbatchSize", 512.into())]),
            trial(700.0, 50.0, 0.01, 1.5, false),
        );
        let fast_but_tight = (
            overrides_of(&[("NCpuMoe", 20.into()), ("UbatchSize", 1024.into())]),
            trial(900.0, 40.0, 0.02, 0.4, false),
        );
        let noisy_slow = (
            overrides_of(&[("NCpuMoe", 30.into())]),
            trial(400.0, 60.0, 0.3, 3.0, false),
        );
        let oom = (
            overrides_of(&[("NCpuMoe", 5.into())]),
            trial(9999.0, 999.0, 0.01, 2.0, true),
        );
        [comfortable, fast_but_tight, noisy_slow, oom]
            .iter()
            .map(|(ov, t)| {
                new_candidate(
                    ov,
                    Some(t),
                    ScoreProfile::Both,
                    "seed",
                    Optimize::CodingAgent,
                    &ctx(),
                )
            })
            .collect()
    }

    fn rescored(profile: ScoreProfile) -> Vec<Candidate> {
        quartet()
            .into_iter()
            .map(|c| {
                new_candidate(
                    &c.overrides,
                    c.trial.as_ref(),
                    profile,
                    "seed",
                    Optimize::CodingAgent,
                    &ctx(),
                )
            })
            .collect()
    }

    #[test]
    fn candidate_carries_the_pinned_scores_and_signature() {
        let cards = quartet();
        assert_eq!(cards[0].pure_score, 396.67);
        assert_eq!(cards[1].pure_score, 397.4);
        assert_eq!(cards[1].balanced_score, 309.1);
        assert_eq!(cards[2].balanced_score, 270.0);
        assert_eq!(cards[1].signature, "NCpuMoe=20;UbatchSize=1024");
    }

    #[test]
    fn pure_profile_picks_the_fast_but_tight_candidate() {
        let beam = select_beam(&rescored(ScoreProfile::Pure), 1, ScoreProfile::Pure);
        assert_eq!(beam[0].signature, "NCpuMoe=20;UbatchSize=1024");
    }

    #[test]
    fn balanced_profile_picks_the_comfortable_candidate() {
        let beam = select_beam(&rescored(ScoreProfile::Balanced), 1, ScoreProfile::Balanced);
        assert_eq!(beam[0].signature, "NCpuMoe=10;UbatchSize=512");
    }

    #[test]
    fn both_profile_keeps_both_frontier_winners() {
        let beam = select_beam(&quartet(), 1, ScoreProfile::Both);
        let sigs: Vec<&str> = beam.iter().map(|c| c.signature.as_str()).collect();
        assert!(sigs.contains(&"NCpuMoe=20;UbatchSize=1024"));
        assert!(sigs.contains(&"NCpuMoe=10;UbatchSize=512"));
    }

    #[test]
    fn an_oom_trial_is_never_selected() {
        let beam = select_beam(&quartet(), 10, ScoreProfile::Both);
        assert!(!beam.iter().any(|c| c.signature == "NCpuMoe=5"));
    }

    #[test]
    fn beam_dedups_by_signature_keeping_the_higher_score() {
        let ov = overrides_of(&[("NCpuMoe", 10.into())]);
        let slow = new_candidate(
            &ov,
            Some(&trial(300.0, 30.0, 0.01, 3.0, false)),
            ScoreProfile::Pure,
            "seed",
            Optimize::CodingAgent,
            &ctx(),
        );
        let fast = new_candidate(
            &ov,
            Some(&trial(700.0, 50.0, 0.01, 3.0, false)),
            ScoreProfile::Pure,
            "beam_1",
            Optimize::CodingAgent,
            &ctx(),
        );
        let beam = select_beam(&[slow, fast.clone()], 5, ScoreProfile::Pure);
        assert_eq!(beam.len(), 1, "one signature, one slot");
        assert_eq!(beam[0].selected_score, fast.selected_score);
    }

    #[test]
    fn a_passed_soak_sibling_beats_a_higher_scoring_degraded_winner() {
        let degraded_fast = new_candidate(
            &overrides_of(&[("NCpuMoe", 20.into())]),
            Some(&trial(900.0, 40.0, 0.02, 3.0, false)),
            ScoreProfile::Pure,
            "seed",
            Optimize::CodingAgent,
            &ctx(),
        );
        let validated_slower = new_candidate(
            &overrides_of(&[("NCpuMoe", 10.into())]),
            Some(&trial(500.0, 35.0, 0.02, 3.0, false)),
            ScoreProfile::Pure,
            "seed",
            Optimize::CodingAgent,
            &ctx(),
        );
        assert!(
            profile_score(&degraded_fast, Profile::Pure)
                > profile_score(&validated_slower, Profile::Pure),
            "precondition: the degraded winner scores higher"
        );
        let mut winners = BTreeMap::new();
        winners.insert(
            "pure".to_string(),
            ProfileWinner {
                winner: degraded_fast,
                soak_passed: Some(false),
            },
        );
        winners.insert(
            "balanced".to_string(),
            ProfileWinner {
                winner: validated_slower,
                soak_passed: Some(true),
            },
        );
        let pick = cross_profile_winner(Profile::Pure, &winners, "").expect("a pick");
        assert!(pick.passed);
        assert_eq!(pick.source_profile, "balanced");
        assert_eq!(pick.winner.signature, "NCpuMoe=10");
    }

    #[test]
    fn within_a_soak_tier_the_higher_score_under_this_metric_wins() {
        let a = new_candidate(
            &overrides_of(&[("NCpuMoe", 10.into())]),
            Some(&trial(700.0, 50.0, 0.01, 3.0, false)),
            ScoreProfile::Pure,
            "seed",
            Optimize::CodingAgent,
            &ctx(),
        );
        let b = new_candidate(
            &overrides_of(&[("NCpuMoe", 15.into())]),
            Some(&trial(500.0, 35.0, 0.01, 3.0, false)),
            ScoreProfile::Pure,
            "seed",
            Optimize::CodingAgent,
            &ctx(),
        );
        let mut winners = BTreeMap::new();
        winners.insert(
            "pure".to_string(),
            ProfileWinner {
                winner: a,
                soak_passed: Some(true),
            },
        );
        winners.insert(
            "balanced".to_string(),
            ProfileWinner {
                winner: b,
                soak_passed: Some(true),
            },
        );
        let pick = cross_profile_winner(Profile::Pure, &winners, "").expect("a pick");
        assert_eq!(pick.winner.signature, "NCpuMoe=10");
        // Excluding the source profile skips it.
        let pick = cross_profile_winner(Profile::Pure, &winners, "pure").expect("a pick");
        assert_eq!(pick.source_profile, "balanced");
    }

    #[test]
    fn an_unrecorded_soak_verdict_counts_as_passed() {
        let c = new_candidate(
            &overrides_of(&[("NCpuMoe", 10.into())]),
            Some(&trial(700.0, 50.0, 0.01, 3.0, false)),
            ScoreProfile::Pure,
            "seed",
            Optimize::CodingAgent,
            &ctx(),
        );
        let mut winners = BTreeMap::new();
        winners.insert(
            "pure".to_string(),
            ProfileWinner {
                winner: c,
                soak_passed: None,
            },
        );
        assert!(
            cross_profile_winner(Profile::Balanced, &winners, "")
                .expect("a pick")
                .passed
        );
    }
}
