//! The findbest orchestration: drive the search phases over a
//! [`TrialRunner`], retain the strongest candidates as a beam between phases,
//! and hand back a verified winner ready for the best-config export.
//!
//! Phase order mirrors the shipped tuner: baseline → VRAM-fit → batching →
//! flash-attention → memory flags → SWA/cache flags → threads (CPU-offload
//! only) → KV types → MoE refinement → verification. Every phase spends from
//! one trial budget and dedups by config signature.

use std::collections::BTreeSet;

use localbench_scoring::score::Optimize;
use localbench_scoring::score::Overrides;
use localbench_scoring::score::Trial;
use localbench_scoring::score::{cross_phase_stability_index, HistoryTrial};
use localbench_search::candidate::{
    new_candidate, select_beam, Candidate, Profile, ScoreProfile, ScoringContext,
};
use localbench_search::overrides::{candidate_signature, join_overrides, overrides_of};
use localbench_search::seeds::SmartSeeds;
use localbench_search::space::{
    baseline_recovery_seed, dense_recovery_candidates, expand_phase_candidates,
    fine_tune_n_cpu_moe_candidates, kv_candidate_pairs, moe_candidate_values,
    moe_coverage_worklist, moe_edge_refine_values, mtp_minimum_n_cpu_moe,
    recovery_n_cpu_moe_candidates, resolve_allowed_kv_types, resolve_tuner_budget, seed_failed,
    swa_flag_overlays, KvPair, SearchSpace,
};
use serde_json::json;

use crate::trial::TrialRunner;

/// What a findbest run is asked to do.
#[derive(Debug, Clone)]
pub struct TunerParams {
    pub profile: ScoreProfile,
    pub optimize: Optimize,
    /// Trial budget (clamped to the supported range).
    pub budget: i64,
    /// The model's baseline KV pair.
    pub baseline_kv: KvPair,
    pub mode: localx_llama_core::Mode,
    /// Logical cores, for the thread sweep.
    pub logical_cores: u32,
    /// Candidates retained and expanded between search phases.
    pub beam_width: usize,
}

/// The finished run.
#[derive(Debug, Clone)]
pub struct TunerOutcome {
    pub winner: Candidate,
    /// Trials actually measured.
    pub trials: usize,
    /// The winner re-measured cleanly in the verification phase.
    pub verified: bool,
    /// The beam width that produced this winner.
    pub beam_width: usize,
}

/// Default number of candidate lineages retained between phases.
pub const DEFAULT_BEAM_WIDTH: usize = 3;

/// Failed fresh verification can discard a winner and try the next beam
/// candidate; reserve the complete retry ladder before spending on search.
const MAX_VERIFICATION_ATTEMPTS: i64 = 3;

/// The search phases in the order `run_tuner` runs them, so a phase can leave
/// room for the ones that come after it. `verify` is not here: it spends from
/// the reserve this list is subtracted from.
const SEARCH_PHASES: [&str; 9] = [
    "baseline",
    "vram-fit",
    "batching",
    "flash-attn",
    "memory-flags",
    "cache-flags",
    "threads",
    "kv-types",
    "refine",
];

/// Trials a phase that has not run yet is guaranteed. Beam retention
/// multiplies what a phase wants to measure by its width, and a phase spends
/// in full before the next one starts — so without a reserve the widest phase
/// takes the whole search budget and every later phase announces itself and
/// measures nothing. Two is the smallest useful floor: the flag phases are
/// A/B overlays, and one trial cannot compare anything.
const PHASE_TRIAL_FLOOR: i64 = 2;

/// The cumulative trial count `phase` may not exceed, so every later phase
/// keeps its floor. Computed once when a phase starts, from the trials already
/// spent. A phase always gets at least one trial while the global budget
/// lasts, so a small `--budget` degrades to one trial per phase instead of
/// stopping the search outright.
fn phase_ceiling(phase: &str, trials_at_phase_start: usize, search_budget: i64) -> i64 {
    let phases_after = SEARCH_PHASES
        .iter()
        .position(|known| *known == phase)
        .map_or(0, |index| SEARCH_PHASES.len() - index - 1) as i64;
    search_budget
        .saturating_sub(phases_after * PHASE_TRIAL_FLOOR)
        .max(trials_at_phase_start as i64 + 1)
}

/// The active phase's spend cap, remembered so the ceiling is computed from
/// the trial count at the phase's first measurement rather than its latest.
struct PhaseGate {
    phase: String,
    ceiling: i64,
    announced: bool,
}

/// The rankable profile used to derive profile-sensitive smart seeds (`both`
/// tracks both beam frontiers but uses pure-profile seed policy).
#[must_use]
pub fn rank_profile(profile: ScoreProfile) -> Profile {
    match profile {
        ScoreProfile::Balanced => Profile::Balanced,
        _ => Profile::Pure,
    }
}

/// The search-space spelling of a backend mode.
#[must_use]
pub fn space_mode(mode: localx_llama_core::Mode) -> localbench_search::space::Mode {
    match mode {
        localx_llama_core::Mode::Native => localbench_search::space::Mode::Native,
        localx_llama_core::Mode::Turboquant => localbench_search::space::Mode::Turboquant,
        localx_llama_core::Mode::Mtpturbo => localbench_search::space::Mode::Mtpturbo,
        localx_llama_core::Mode::PrismMl => localbench_search::space::Mode::PrismMl,
    }
}

/// Whether a batching pair is dominated by an already-OOM'd pair (equal or
/// larger on both axes never fits either).
#[must_use]
pub fn batching_dominated(ub: i64, b: i64, oomed: &[(i64, i64)]) -> bool {
    oomed
        .iter()
        .any(|(failed_ub, failed_b)| ub >= *failed_ub && b >= *failed_b)
}

fn trial_summary(trial: &Trial, score: f64) -> String {
    let evidence = trial
        .diagnostic
        .as_ref()
        .map(|diagnostic| {
            let path = if diagnostic.log_path.is_empty() {
                diagnostic.manifest_path.as_str()
            } else {
                diagnostic.log_path.as_str()
            };
            if path.is_empty() {
                String::new()
            } else {
                format!(" (evidence: {path})")
            }
        })
        .unwrap_or_default();
    if trial.is_measurement_usable() {
        format!("{score:.1}{evidence}")
    } else if let Some(failure) = &trial.failure {
        format!("{}{evidence}", failure.summary())
    } else if trial.oom {
        format!("readiness/out_of_memory{evidence}")
    } else {
        format!("unusable_measurement{evidence}")
    }
}

/// How many `--n-gpu-layers` offload steps a dense VRAM-fit sweep tries
/// (halving from the real layer count), on top of the KV-shrink candidates.
const DENSE_NGL_CANDIDATES: usize = 4;

/// Drive the full findbest search. `events` receives one plain progress line
/// per phase and per trial.
pub fn run_tuner(
    runner: &mut dyn TrialRunner,
    space: &SearchSpace,
    seeds: &SmartSeeds,
    ctx: &ScoringContext,
    params: &TunerParams,
    events: &mut dyn FnMut(String),
) -> Option<TunerOutcome> {
    let budget = resolve_tuner_budget(params.budget);
    let search_budget = budget.saturating_sub(MAX_VERIFICATION_ATTEMPTS).max(1);
    let beam_width = params.beam_width.max(1);
    events(format!("search: beam width {beam_width}"));
    let mut trials = 0_usize;
    let mut history: Vec<Candidate> = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut phase_gate: Option<PhaseGate> = None;

    let mut measure = |overrides: &Overrides,
                       phase: &str,
                       trials: &mut usize,
                       history: &mut Vec<Candidate>,
                       seen: &mut BTreeSet<String>,
                       events: &mut dyn FnMut(String)|
     -> Option<Candidate> {
        // Keep the verification ladder's slots for the fresh measurement.
        if *trials as i64 >= search_budget {
            return None;
        }
        // And keep every later phase's floor, so beam width buys breadth
        // within a phase instead of taking the phases that follow it.
        let ceiling = match &phase_gate {
            Some(gate) if gate.phase == phase => gate.ceiling,
            _ => {
                let ceiling = phase_ceiling(phase, *trials, search_budget);
                phase_gate = Some(PhaseGate {
                    phase: phase.to_string(),
                    ceiling,
                    announced: false,
                });
                ceiling
            }
        };
        if *trials as i64 >= ceiling {
            if let Some(gate) = phase_gate.as_mut() {
                if !gate.announced {
                    gate.announced = true;
                    events(format!(
                        "phase {phase}: reserve reached at {trials} trials; its remaining candidates go unmeasured so later phases can run"
                    ));
                }
            }
            return None;
        }
        let signature = candidate_signature(overrides);
        if !seen.insert(format!("{phase}|{signature}")) {
            return None;
        }
        *trials += 1;
        let trial = runner.measure(overrides, phase);
        let candidate = new_candidate(
            overrides,
            Some(&trial),
            params.profile,
            phase,
            params.optimize,
            ctx,
        );
        events(format!(
            "trial {trials}/{budget} [{phase}] {} -> {}",
            signature,
            trial_summary(&trial, candidate.selected_score)
        ));
        history.push(candidate.clone());
        Some(candidate)
    };

    // ----- Phase 1: baseline -----
    events("phase: baseline".to_string());
    let mut baseline = overrides_of(&[
        ("KvK", json!(params.baseline_kv.k.clone())),
        ("KvV", json!(params.baseline_kv.v.clone())),
    ]);
    if space.is_moe {
        baseline = join_overrides(
            &baseline,
            &overrides_of(&[("NCpuMoe", json!(space.baseline_n_cpu_moe))]),
        );
    }
    let baseline_candidate = measure(
        &baseline,
        "baseline",
        &mut trials,
        &mut history,
        &mut seen,
        events,
    );
    let baseline_needs_recovery = baseline_candidate
        .as_ref()
        .and_then(|candidate| candidate.trial.as_ref())
        .is_some_and(Trial::needs_memory_recovery);
    let baseline_seed_failed = seed_failed(baseline_candidate.as_ref());
    let baseline_unusable_without_recovery = baseline_candidate
        .as_ref()
        .and_then(|candidate| candidate.trial.as_ref())
        .is_none_or(|trial| !trial.is_measurement_usable() && !trial.needs_memory_recovery());
    if baseline_unusable_without_recovery {
        events(
            "stopped: baseline reached no usable measurement and supplied no startup/OOM fit evidence; fix the reported contract/content failure before retuning"
                .to_string(),
        );
        return None;
    }

    // ----- Phase 2: VRAM fit -----
    events("phase: vram-fit".to_string());
    if space.is_moe {
        let minimum = mtp_minimum_n_cpu_moe(
            space,
            seeds,
            space_mode(params.mode),
            !space.mtp_draft_candidates.is_empty(),
        );
        let moe_values = if baseline_needs_recovery || baseline_seed_failed {
            recovery_n_cpu_moe_candidates(
                space.baseline_n_cpu_moe,
                space.baseline_n_cpu_moe,
                space.moe_upper,
                minimum,
            )
        } else {
            moe_candidate_values(space, seeds, false, minimum)
        };
        let coverage_seeds = vec![baseline_recovery_seed(
            baseline_candidate.as_ref(),
            &baseline,
        )];
        let coverage_pairs = vec![params.baseline_kv.clone()];
        // Plan against this phase's own ceiling, not the whole remaining
        // budget: the per-phase reserve truncates the sweep, and a worklist
        // that ignored it would disclose full coverage for configurations the
        // phase never reaches — the exact misreading these counts exist to
        // prevent.
        let budget_remaining = usize::try_from(
            phase_ceiling("vram-fit", trials, search_budget).saturating_sub(trials as i64),
        )
        .unwrap_or(0);
        let coverage = moe_coverage_worklist(
            &coverage_seeds,
            &coverage_pairs,
            &moe_values,
            budget_remaining,
        );
        events(format!(
            "coverage: scheduled {}/{} MoE configurations ({} skipped by budget)",
            coverage.scheduled.len(),
            coverage.planned_count,
            coverage.skipped_count
        ));
        for overrides in coverage.scheduled {
            measure(
                &overrides,
                "vram-fit",
                &mut trials,
                &mut history,
                &mut seen,
                events,
            );
        }
    } else if baseline_needs_recovery {
        // A dense model has no expert lever: fit it into VRAM by shrinking the KV
        // cache first (turbo pairs, every layer still on the GPU) and then
        // lowering `--n-gpu-layers`, halved from the real layer count. This is a
        // recovery ladder, not an optimization sweep — a dense baseline that
        // already starts is running every layer on the GPU, and lowering `-ngl`
        // from there only makes it slower, so the branch runs only when the
        // baseline OOM'd. Without it a dense OOM leaves no surviving candidate
        // (LocalHub#76).
        let allowed = resolve_allowed_kv_types(&[], &params.baseline_kv, space_mode(params.mode));
        let kv_pairs = kv_candidate_pairs(&allowed, false, false);
        for overrides in dense_recovery_candidates(
            &params.baseline_kv,
            &kv_pairs,
            space.baseline_ngl,
            space.block_count,
            DENSE_NGL_CANDIDATES,
        ) {
            let ov = join_overrides(&baseline, &overrides);
            measure(
                &ov,
                "vram-fit",
                &mut trials,
                &mut history,
                &mut seen,
                events,
            );
        }
    }

    let beam_so_far = |history: &[Candidate]| select_beam(history, beam_width, params.profile);
    let best_so_far = |history: &[Candidate]| beam_so_far(history).into_iter().next();
    if best_so_far(&history).is_none() {
        events(
            "stopped: the startup/OOM recovery ladder produced no usable measurement".to_string(),
        );
        return None;
    }

    // ----- Phase 3: batching (ub, b) joint sweep, b >= ub, OOM-dominance pruned -----
    events("phase: batching".to_string());
    for parent in beam_so_far(&history) {
        let mut oomed_batching: Vec<(i64, i64)> = Vec::new();
        for &ub in &space.ubatch_candidates {
            for &b in &space.batch_candidates {
                if b < ub || batching_dominated(ub, b, &oomed_batching) {
                    continue;
                }
                let overrides = join_overrides(
                    &parent.overrides,
                    &overrides_of(&[("UbatchSize", json!(ub)), ("BatchSize", json!(b))]),
                );
                if let Some(candidate) = measure(
                    &overrides,
                    "batching",
                    &mut trials,
                    &mut history,
                    &mut seen,
                    events,
                ) {
                    if candidate.trial.as_ref().is_some_and(|t| t.oom) {
                        oomed_batching.push((ub, b));
                    }
                }
            }
        }
    }

    // ----- Phases 4-6: flag overlays off the current best -----
    let flag_phases: [(&str, Vec<Overrides>); 3] = [
        (
            "flash-attn",
            vec![
                overrides_of(&[("FlashAttn", json!(true))]),
                overrides_of(&[("FlashAttn", json!(false))]),
            ],
        ),
        (
            "memory-flags",
            vec![
                overrides_of(&[("Mlock", json!(true))]),
                overrides_of(&[("NoMmap", json!(true))]),
                overrides_of(&[("Mlock", json!(true)), ("NoMmap", json!(true))]),
            ],
        ),
        ("cache-flags", swa_flag_overlays()),
    ];
    for (phase, overlays) in flag_phases {
        events(format!("phase: {phase}"));
        let beam = beam_so_far(&history);
        for overrides in expand_phase_candidates(&beam, &overlays) {
            measure(
                &overrides,
                phase,
                &mut trials,
                &mut history,
                &mut seen,
                events,
            );
        }
    }

    // ----- Phase 7: threads (only when work actually runs on the CPU) -----
    events("phase: threads".to_string());
    for parent in beam_so_far(&history) {
        let cpu_offload = parent
            .overrides
            .get("NCpuMoe")
            .and_then(serde_json::Value::as_i64)
            .unwrap_or(0)
            > 0;
        if cpu_offload {
            for &threads in &seeds.thread_candidates {
                let overrides = join_overrides(
                    &parent.overrides,
                    &overrides_of(&[("Threads", json!(threads))]),
                );
                measure(
                    &overrides,
                    "threads",
                    &mut trials,
                    &mut history,
                    &mut seen,
                    events,
                );
            }
        }
    }

    // ----- Phase 8: KV cache types -----
    events("phase: kv-types".to_string());
    for parent in beam_so_far(&history) {
        let allowed = resolve_allowed_kv_types(&[], &params.baseline_kv, space_mode(params.mode));
        for pair in kv_candidate_pairs(&allowed, false, false) {
            let overrides = join_overrides(
                &parent.overrides,
                &overrides_of(&[("KvK", json!(pair.k)), ("KvV", json!(pair.v))]),
            );
            measure(
                &overrides,
                "kv-types",
                &mut trials,
                &mut history,
                &mut seen,
                events,
            );
        }
    }

    // ----- Phase 9: MoE refinement around the best -----
    events("phase: refine".to_string());
    if space.is_moe {
        let measured_stable: Vec<i64> = history
            .iter()
            .filter(|candidate| {
                candidate
                    .trial
                    .as_ref()
                    .is_some_and(Trial::is_measurement_usable)
            })
            .filter_map(|candidate| {
                candidate
                    .overrides
                    .get("NCpuMoe")
                    .and_then(serde_json::Value::as_i64)
            })
            .collect();
        for parent in beam_so_far(&history) {
            if let Some(current) = parent
                .overrides
                .get("NCpuMoe")
                .and_then(serde_json::Value::as_i64)
            {
                let mut values = fine_tune_n_cpu_moe_candidates(current, space.moe_upper);
                for value in
                    moe_edge_refine_values(&measured_stable, current, 5, 8, 0, space.moe_upper)
                {
                    if !values.contains(&value) {
                        values.push(value);
                    }
                }
                for value in values {
                    let overrides = join_overrides(
                        &parent.overrides,
                        &overrides_of(&[("NCpuMoe", json!(value))]),
                    );
                    measure(
                        &overrides,
                        "refine",
                        &mut trials,
                        &mut history,
                        &mut seen,
                        events,
                    );
                }
            }
        }
    }

    // ----- Final ranking: light the cross-phase stability factor up -----
    // Scores computed at measure time necessarily see an empty stability
    // index (a config's cross-phase variance only exists once several phases
    // have measured it). Rebuild the index from this run's own history and
    // re-score every candidate with it before picking the winner, so a
    // VRAM-marginal config that was fast in one phase and slow in another is
    // penalized the way the balanced profile documents — instead of the
    // factor sitting permanently at 1.0.
    let stability_trials: Vec<HistoryTrial> = history
        .iter()
        .filter_map(|c| {
            let t = c.trial.as_ref()?;
            t.is_measurement_usable().then_some(HistoryTrial {
                phase: c.phase.clone(),
                overrides: c.overrides.clone(),
                startup_ok: t.startup_ok,
                oom: t.oom,
                tg_tps: t.tg_tps,
            })
        })
        .collect();
    let stability_index = cross_phase_stability_index(&stability_trials);
    if !stability_index.is_empty() {
        events(format!(
            "stability: {} config group(s) carry cross-phase variance data",
            stability_index.len()
        ));
        let final_ctx = ScoringContext {
            workload: ctx.workload,
            host: ctx.host,
            vram_params: ctx.vram_params,
            stability_index,
        };
        history = history
            .iter()
            .map(|c| {
                new_candidate(
                    &c.overrides,
                    c.trial.as_ref(),
                    params.profile,
                    &c.phase,
                    params.optimize,
                    &final_ctx,
                )
            })
            .collect();
    }

    // ----- Phase 10: verify the winner with a fresh measurement -----
    events("phase: verify".to_string());
    let mut verified = false;
    for _ in 0..MAX_VERIFICATION_ATTEMPTS {
        if trials as i64 >= budget {
            break;
        }
        let Some(best) = best_so_far(&history) else {
            break;
        };
        trials += 1;
        let trial = runner.measure(&best.overrides, "verify");
        if trial.is_measurement_usable() {
            verified = true;
            break;
        }
        events(format!(
            "verify failed for {} -> {}; dropping it and retrying",
            best.signature,
            trial_summary(&trial, 0.0)
        ));
        // Remove every trace of the failed config so the next-best surfaces.
        history.retain(|c| c.signature != best.signature);
    }

    if !verified {
        events("stopped: no candidate passed a fresh usable verification trial".to_string());
        return None;
    }
    let winner = best_so_far(&history)?;
    events(format!(
        "winner: {} ({} = {:.1}, verified: {verified})",
        winner.signature,
        match params.profile {
            ScoreProfile::Pure => "pure",
            ScoreProfile::Balanced => "balanced",
            ScoreProfile::Both => "best-of",
        },
        winner.selected_score
    ));
    Some(TunerOutcome {
        winner,
        trials,
        verified,
        beam_width,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::trial::failed_trial;
    use localbench_scoring::score::{Telemetry, Trial, Workload};
    use localbench_search::candidate::ScoringContext;
    use localbench_search::space::{resolve_search_space, ModelAxes};

    /// Scripted runner: OOMs any config whose batch exceeds a ceiling, and
    /// otherwise scores higher for lower NCpuMoe (more GPU = faster).
    struct ScriptedRunner {
        measured: Vec<String>,
    }

    impl TrialRunner for ScriptedRunner {
        fn measure(&mut self, overrides: &Overrides, phase: &str) -> Trial {
            self.measured.push(phase.to_string());
            if overrides
                .get("BatchSize")
                .and_then(serde_json::Value::as_i64)
                .unwrap_or(0)
                > 1024
            {
                return failed_trial(true);
            }
            let moe = overrides
                .get("NCpuMoe")
                .and_then(serde_json::Value::as_i64)
                .unwrap_or(0) as f64;
            Trial {
                startup_ok: true,
                oom: false,
                measurement_usable: true,
                pp_tps: 900.0 - moe,
                tg_tps: 100.0 - moe,
                long_ctx_pp_tps: None,
                long_ctx_tg_tps: None,
                long_ctx_target_tokens: None,
                variance: Some(0.02),
                startup_failure: None,
                telemetry: Telemetry::default(),
                ..Trial::default()
            }
        }
    }

    fn space() -> SearchSpace {
        resolve_search_space(
            &ModelAxes {
                n_cpu_moe: Some(20),
                config_n_cpu_moe: None,
                n_gpu_layers: Some(999),
                moe_expert_layers: Some(48),
                spec_type: None,
                spec_draft_n_max: None,
                skip_phases: vec![],
            },
            1, // MoE
            48,
        )
    }

    fn seeds() -> SmartSeeds {
        localbench_search::seeds::resolve_smart_seeds(
            &space(),
            localbench_search::seeds::HostFacts {
                vram_gb: 24,
                logical_cores: 16,
                available_ram_gb: 64.0,
                gguf_size_gb: 21.0,
            },
            Profile::Pure,
        )
    }

    fn ctx() -> ScoringContext {
        ScoringContext {
            workload: Workload::default(),
            host: localbench_scoring::score::HostSignals { logical_cores: 16 },
            vram_params: Default::default(),
            stability_index: Default::default(),
        }
    }

    fn params() -> TunerParams {
        TunerParams {
            profile: ScoreProfile::Pure,
            optimize: Optimize::Both,
            budget: 25,
            baseline_kv: KvPair {
                k: "q8_0".to_string(),
                v: "q8_0".to_string(),
            },
            mode: localx_llama_core::Mode::Native,
            logical_cores: 16,
            beam_width: DEFAULT_BEAM_WIDTH,
        }
    }

    /// Beam retention multiplies what a phase wants to measure, and a phase
    /// spends in full before the next one starts. At the documented default
    /// (`--budget 30`, `--beam-width 3`) the batching sweep used to take the
    /// whole search budget: flash-attn, memory-flags, cache-flags, kv-types
    /// and refine announced themselves and measured nothing at all. The
    /// per-phase reserve is what stops that, so pin the coverage it buys at
    /// both the default width and at width one.
    #[test]
    fn every_search_phase_still_measures_at_the_default_budget() {
        for width in [1, DEFAULT_BEAM_WIDTH] {
            let mut runner = ScriptedRunner { measured: vec![] };
            let mut events = Vec::new();
            let mut wide = params();
            wide.budget = 30;
            wide.beam_width = width;
            let outcome = run_tuner(
                &mut runner,
                &space(),
                &seeds(),
                &ctx(),
                &wide,
                &mut |line| events.push(line),
            )
            .expect("the default budget still produces a verified winner");

            for phase in [
                "baseline",
                "vram-fit",
                "batching",
                "flash-attn",
                "memory-flags",
                "cache-flags",
                "kv-types",
                "refine",
            ] {
                assert!(
                    runner.measured.iter().any(|measured| measured == phase),
                    "beam width {width} starved phase {phase}: {:?}",
                    runner.measured
                );
            }
            assert!(outcome.trials as i64 <= 30);
        }
    }

    /// A capped phase says so. The failure this guards against was silent:
    /// the phase header printed, no trial followed, and the run gave the
    /// reader no way to tell a skipped phase from an empty one.
    #[test]
    fn a_phase_that_hits_its_reserve_says_so_instead_of_going_quiet() {
        let mut runner = ScriptedRunner { measured: vec![] };
        let mut events = Vec::new();
        let mut wide = params();
        wide.budget = 30;
        wide.beam_width = DEFAULT_BEAM_WIDTH;
        run_tuner(
            &mut runner,
            &space(),
            &seeds(),
            &ctx(),
            &wide,
            &mut |line| events.push(line),
        )
        .expect("a verified winner");
        assert!(
            events.iter().any(|line| line.contains("reserve reached")),
            "the capped phase must report the cap: {events:?}"
        );
    }

    #[test]
    fn the_tuner_respects_the_budget_and_returns_a_verified_winner() {
        let mut runner = ScriptedRunner { measured: vec![] };
        let mut events = Vec::new();
        let outcome = run_tuner(
            &mut runner,
            &space(),
            &seeds(),
            &ctx(),
            &params(),
            &mut |line| events.push(line),
        )
        .unwrap();

        assert!(outcome.trials as i64 <= 25);
        assert!(outcome.verified);
        // The scripted world rewards lower NCpuMoe, so the winner offloads less
        // than the catalog baseline of 20.
        let winner_moe = outcome
            .winner
            .overrides
            .get("NCpuMoe")
            .and_then(serde_json::Value::as_i64)
            .unwrap_or(0);
        assert!(winner_moe < 20);
        assert!(events.iter().any(|e| e.starts_with("winner:")));
    }

    #[test]
    fn a_one_trial_budget_measures_baseline_but_cannot_export_unverified_data() {
        let mut runner = ScriptedRunner { measured: vec![] };
        let mut one = params();
        one.budget = 1;
        let outcome = run_tuner(&mut runner, &space(), &seeds(), &ctx(), &one, &mut |_| {});
        assert!(outcome.is_none());
        assert_eq!(runner.measured, ["baseline"]);
    }

    struct BeamForkRunner {
        measured: Vec<Overrides>,
    }

    impl TrialRunner for BeamForkRunner {
        fn measure(&mut self, overrides: &Overrides, _phase: &str) -> Trial {
            self.measured.push(overrides.clone());
            let ubatch = overrides
                .get("UbatchSize")
                .and_then(serde_json::Value::as_i64);
            let flash = overrides
                .get("FlashAttn")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false);
            let tg_tps = match (ubatch, flash) {
                // This lineage is second-best after batching, but becomes the
                // global best only after the next phase overlays flash-attn.
                (Some(512), true) => 200.0,
                (Some(256), _) => 120.0,
                (Some(512), _) => 110.0,
                _ => 100.0,
            };
            Trial {
                startup_ok: true,
                measurement_usable: true,
                pp_tps: tg_tps * 9.0,
                tg_tps,
                variance: Some(0.01),
                ..Trial::default()
            }
        }
    }

    #[test]
    fn the_beam_finds_a_second_best_early_lineage_that_greedy_search_loses() {
        let fork_space = SearchSpace {
            is_moe: false,
            baseline_n_cpu_moe: 0,
            moe_upper: 0,
            baseline_ngl: 999,
            block_count: 40,
            ubatch_candidates: vec![256, 512],
            batch_candidates: vec![512],
            skip_phases: Vec::new(),
            mtp_draft_candidates: Vec::new(),
        };
        let tuner_params = TunerParams {
            optimize: Optimize::Gen,
            budget: 60,
            beam_width: 2,
            ..params()
        };
        let mut beam_runner = BeamForkRunner {
            measured: Vec::new(),
        };
        let beam_outcome = run_tuner(
            &mut beam_runner,
            &fork_space,
            &seeds(),
            &ctx(),
            &tuner_params,
            &mut |_| {},
        )
        .expect("the beam produces a verified winner");
        assert_eq!(
            beam_outcome
                .winner
                .overrides
                .get("UbatchSize")
                .and_then(serde_json::Value::as_i64),
            Some(512)
        );
        assert_eq!(
            beam_outcome
                .winner
                .overrides
                .get("FlashAttn")
                .and_then(serde_json::Value::as_bool),
            Some(true)
        );
        assert_eq!(beam_outcome.beam_width, 2);

        let mut greedy_runner = BeamForkRunner {
            measured: Vec::new(),
        };
        let greedy_params = TunerParams {
            beam_width: 1,
            ..tuner_params
        };
        let greedy_outcome = run_tuner(
            &mut greedy_runner,
            &fork_space,
            &seeds(),
            &ctx(),
            &greedy_params,
            &mut |_| {},
        )
        .expect("width one still produces a verified winner");
        assert_eq!(
            greedy_outcome
                .winner
                .overrides
                .get("UbatchSize")
                .and_then(serde_json::Value::as_i64),
            Some(256)
        );
        assert!(!greedy_runner.measured.iter().any(|overrides| {
            overrides
                .get("UbatchSize")
                .and_then(serde_json::Value::as_i64)
                == Some(512)
                && overrides
                    .get("FlashAttn")
                    .and_then(serde_json::Value::as_bool)
                    == Some(true)
        }));
    }

    /// A runner whose decode speed swings for any config group measured more
    /// than once (slow first, fast on the repeat): the eventual leader is a
    /// re-measured config, so its group carries exactly the cross-phase
    /// variance the stability factor must penalize.
    struct UnstableRunner {
        group_calls: std::collections::BTreeMap<String, usize>,
    }

    impl TrialRunner for UnstableRunner {
        fn measure(&mut self, overrides: &Overrides, _phase: &str) -> Trial {
            if overrides
                .get("BatchSize")
                .and_then(serde_json::Value::as_i64)
                .unwrap_or(0)
                > 1024
            {
                return failed_trial(true);
            }
            let key = localbench_scoring::score::stability_group_key(overrides);
            let count = self.group_calls.entry(key).or_insert(0);
            *count += 1;
            let tg = if *count == 1 { 60.0 } else { 100.0 };
            Trial {
                startup_ok: true,
                oom: false,
                measurement_usable: true,
                pp_tps: 9.0 * tg,
                tg_tps: tg,
                long_ctx_pp_tps: None,
                long_ctx_tg_tps: None,
                long_ctx_target_tokens: None,
                variance: Some(0.02),
                startup_failure: None,
                telemetry: Telemetry::default(),
                ..Trial::default()
            }
        }
    }

    #[test]
    fn the_final_ranking_rebuilds_the_stability_index_from_history() {
        // The failure this pins: candidates are scored at measure time with a
        // necessarily-empty stability index, and the old final ranking never
        // rebuilt it — the documented cross-phase penalty was permanently 1.0.
        // With an unstable world, the winner's balanced breakdown must now
        // carry a stability factor below full credit.
        let mut runner = UnstableRunner {
            group_calls: std::collections::BTreeMap::new(),
        };
        let mut events = Vec::new();
        let mut tuner_params = params();
        tuner_params.profile = ScoreProfile::Balanced;
        let outcome = run_tuner(
            &mut runner,
            &space(),
            &seeds(),
            &ctx(),
            &tuner_params,
            &mut |line| events.push(line),
        )
        .unwrap();

        assert!(
            events.iter().any(|e| e.starts_with("stability:")),
            "the final ranking must report the rebuilt index, got {events:?}"
        );
        assert!(
            outcome.winner.score_breakdown.stability_factor < 1.0,
            "an unstable config group must be penalized; breakdown: {:?}",
            outcome.winner.score_breakdown
        );
    }

    #[test]
    fn oom_dominated_batching_pairs_are_pruned_not_measured() {
        assert!(batching_dominated(2048, 2048, &[(1024, 2048)]));
        assert!(batching_dominated(1024, 2048, &[(1024, 2048)]));
        assert!(!batching_dominated(512, 512, &[(1024, 2048)]));

        let mut runner = ScriptedRunner { measured: vec![] };
        let mut sink = |_line: String| {};
        let outcome = run_tuner(
            &mut runner,
            &space(),
            &seeds(),
            &ctx(),
            &params(),
            &mut sink,
        )
        .unwrap();
        // The winner never carries an OOM'd batching config.
        let winner_batch = outcome
            .winner
            .overrides
            .get("BatchSize")
            .and_then(serde_json::Value::as_i64)
            .unwrap_or(0);
        assert!(winner_batch <= 1024);
    }

    #[test]
    fn profile_and_mode_spellings_map_across_the_crate_boundary() {
        assert_eq!(rank_profile(ScoreProfile::Both), Profile::Pure);
        assert_eq!(rank_profile(ScoreProfile::Balanced), Profile::Balanced);
        assert_eq!(
            space_mode(localx_llama_core::Mode::Turboquant),
            localbench_search::space::Mode::Turboquant
        );
        assert_eq!(
            space_mode(localx_llama_core::Mode::PrismMl),
            localbench_search::space::Mode::PrismMl
        );
    }

    /// A dense model that OOMs at its baseline (full offload, `q8_0` KV) but
    /// fits once the KV cache shrinks (a turbo pair) or layers are offloaded.
    struct DenseVramRunner {
        measured: Vec<(String, Overrides)>,
    }

    impl TrialRunner for DenseVramRunner {
        fn measure(&mut self, overrides: &Overrides, phase: &str) -> Trial {
            self.measured.push((phase.to_string(), overrides.clone()));
            let kv_k = overrides
                .get("KvK")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("q8_0");
            let ngl = overrides
                .get("NGpuLayers")
                .and_then(serde_json::Value::as_i64)
                .unwrap_or(999);
            // Full offload with the full-size KV cache does not fit; a smaller
            // KV cache (turbo) or fewer offloaded layers does.
            let fits = kv_k.starts_with("turbo") || ngl < 65;
            if !fits {
                return failed_trial(true);
            }
            Trial {
                startup_ok: true,
                oom: false,
                measurement_usable: true,
                pp_tps: 400.0,
                tg_tps: 60.0,
                long_ctx_pp_tps: None,
                long_ctx_tg_tps: None,
                long_ctx_target_tokens: None,
                variance: Some(0.02),
                startup_failure: None,
                telemetry: Telemetry::default(),
                ..Trial::default()
            }
        }
    }

    fn dense_space() -> SearchSpace {
        // expert_count 0 = dense; block_count 65 = the real layer count the GGUF
        // read supplies, so offload halves from 65, not the 999 sentinel.
        resolve_search_space(
            &ModelAxes {
                n_cpu_moe: None,
                config_n_cpu_moe: None,
                n_gpu_layers: None,
                moe_expert_layers: None,
                spec_type: None,
                spec_draft_n_max: None,
                skip_phases: vec![],
            },
            0,
            65,
        )
    }

    struct FixedOutcomeRunner {
        outcome: Trial,
        phases: Vec<String>,
    }

    impl TrialRunner for FixedOutcomeRunner {
        fn measure(&mut self, _overrides: &Overrides, phase: &str) -> Trial {
            self.phases.push(phase.to_string());
            self.outcome.clone()
        }
    }

    fn typed_failure(
        stage: localbench_scoring::score::TrialFailureStage,
        reason: localbench_scoring::score::TrialFailureReason,
        oom: bool,
    ) -> Trial {
        Trial {
            startup_ok: stage != localbench_scoring::score::TrialFailureStage::Launch
                && stage != localbench_scoring::score::TrialFailureStage::Readiness,
            oom,
            failure: Some(localbench_scoring::score::TrialFailure {
                stage,
                reason,
                detail: String::new(),
            }),
            ..Trial::default()
        }
    }

    #[test]
    fn contract_and_content_failures_stop_before_dense_or_moe_recovery() {
        use localbench_scoring::score::{TrialFailureReason as Reason, TrialFailureStage as Stage};

        for search_space in [dense_space(), space()] {
            for (stage, reason) in [
                (Stage::Launch, Reason::SpawnFailed),
                (Stage::Request, Reason::Transport),
                (Stage::Response, Reason::HttpStatus),
                (Stage::Response, Reason::ResponseDecode),
                (Stage::Response, Reason::ResponseSchema),
                (Stage::Response, Reason::MissingTimings),
                (Stage::Response, Reason::InvalidTimings),
                (Stage::Content, Reason::EmptyContent),
                (Stage::Content, Reason::ThinkingOnly),
                (Stage::Content, Reason::DegenerateContent),
            ] {
                let mut runner = FixedOutcomeRunner {
                    outcome: typed_failure(stage, reason, false),
                    phases: Vec::new(),
                };
                let outcome = run_tuner(
                    &mut runner,
                    &search_space,
                    &seeds(),
                    &ctx(),
                    &params(),
                    &mut |_| {},
                );
                assert!(outcome.is_none());
                assert_eq!(runner.phases, ["baseline"], "{stage:?}/{reason:?}");
            }
        }
    }

    #[test]
    fn readiness_failures_enter_dense_and_moe_recovery() {
        use localbench_scoring::score::{TrialFailureReason as Reason, TrialFailureStage as Stage};

        for search_space in [dense_space(), space()] {
            for (reason, oom) in [
                (Reason::ReadinessExitedOom, true),
                (Reason::ReadinessExited, false),
                (Reason::ReadinessTimeout, false),
            ] {
                let mut runner = FixedOutcomeRunner {
                    outcome: typed_failure(Stage::Readiness, reason, oom),
                    phases: Vec::new(),
                };
                let outcome = run_tuner(
                    &mut runner,
                    &search_space,
                    &seeds(),
                    &ctx(),
                    &params(),
                    &mut |_| {},
                );
                assert!(outcome.is_none());
                assert!(
                    runner.phases.iter().any(|phase| phase == "vram-fit"),
                    "{reason:?} must enter the recovery ladder"
                );
            }
        }
    }

    struct MoeSeedRecoveryRunner {
        measured: Vec<(String, i64)>,
    }

    impl TrialRunner for MoeSeedRecoveryRunner {
        fn measure(&mut self, overrides: &Overrides, phase: &str) -> Trial {
            let n_cpu_moe = overrides
                .get("NCpuMoe")
                .and_then(serde_json::Value::as_i64)
                .unwrap_or(0);
            self.measured.push((phase.to_string(), n_cpu_moe));
            if phase == "baseline" {
                return typed_failure(
                    localbench_scoring::score::TrialFailureStage::Readiness,
                    localbench_scoring::score::TrialFailureReason::ReadinessExitedOom,
                    true,
                );
            }
            Trial {
                startup_ok: true,
                measurement_usable: true,
                pp_tps: 400.0 + n_cpu_moe as f64,
                tg_tps: 60.0 + n_cpu_moe as f64,
                variance: Some(0.01),
                ..Trial::default()
            }
        }
    }

    /// The coverage line must count what the phase can actually run. The
    /// per-phase reserve truncates the offload sweep, so a worklist planned
    /// against the whole remaining budget announces complete coverage for
    /// configurations that are never measured.
    #[test]
    fn the_coverage_disclosure_counts_what_the_phase_can_measure() {
        let mut runner = ScriptedRunner { measured: vec![] };
        let mut tight = params();
        tight.budget = 20;
        let mut events = Vec::new();
        run_tuner(
            &mut runner,
            &space(),
            &seeds(),
            &ctx(),
            &tight,
            &mut |line| {
                events.push(line);
            },
        );

        let coverage = events
            .iter()
            .find(|line| line.starts_with("coverage:"))
            .expect("the MoE sweep discloses its coverage");
        let measured = runner
            .measured
            .iter()
            .filter(|phase| *phase == "vram-fit")
            .count();
        assert!(
            coverage.contains(&format!("scheduled {measured}/")),
            "coverage claims more than the phase measured ({measured}): {coverage}"
        );
        assert!(
            !coverage.contains("(0 skipped by budget)"),
            "a truncated sweep must report its skipped configurations: {coverage}"
        );
    }

    #[test]
    fn a_failed_moe_seed_measures_the_complete_recovery_worklist() {
        let recovery_space = SearchSpace {
            is_moe: true,
            baseline_n_cpu_moe: 2,
            moe_upper: 6,
            baseline_ngl: 999,
            block_count: 12,
            ubatch_candidates: Vec::new(),
            batch_candidates: Vec::new(),
            skip_phases: Vec::new(),
            mtp_draft_candidates: Vec::new(),
        };
        let expected = recovery_n_cpu_moe_candidates(2, 2, 6, 0);
        assert_eq!(expected, vec![3, 4, 5, 6]);
        let mut runner = MoeSeedRecoveryRunner {
            measured: Vec::new(),
        };
        let mut events = Vec::new();
        let outcome = run_tuner(
            &mut runner,
            &recovery_space,
            &seeds(),
            &ctx(),
            &TunerParams {
                budget: 40,
                ..params()
            },
            &mut |event| events.push(event),
        );
        assert!(outcome.is_some(), "a recovered MoE seed produces a winner");
        let measured_recovery: Vec<i64> = runner
            .measured
            .iter()
            .filter(|(phase, _)| phase == "vram-fit")
            .map(|(_, value)| *value)
            .collect();
        assert_eq!(
            measured_recovery, expected,
            "coverage visits every declared recovery value"
        );
        assert!(events.iter().any(|event| {
            event == "coverage: scheduled 4/4 MoE configurations (0 skipped by budget)"
        }));
    }

    struct VerifyFailureRunner {
        verify_calls: usize,
    }

    impl TrialRunner for VerifyFailureRunner {
        fn measure(&mut self, _overrides: &Overrides, phase: &str) -> Trial {
            if phase == "verify" {
                self.verify_calls += 1;
                return typed_failure(
                    localbench_scoring::score::TrialFailureStage::Response,
                    localbench_scoring::score::TrialFailureReason::MissingTimings,
                    false,
                );
            }
            Trial {
                startup_ok: true,
                measurement_usable: true,
                pp_tps: 400.0,
                tg_tps: 60.0,
                ..Trial::default()
            }
        }
    }

    #[test]
    fn an_unverified_candidate_never_becomes_a_winner() {
        let mut runner = VerifyFailureRunner { verify_calls: 0 };
        let outcome = run_tuner(
            &mut runner,
            &dense_space(),
            &seeds(),
            &ctx(),
            &params(),
            &mut |_| {},
        );
        assert!(outcome.is_none());
        assert_eq!(
            runner.verify_calls, 3,
            "all bounded verification retries ran"
        );
    }

    #[test]
    fn a_dense_model_recovers_from_a_baseline_oom_and_produces_a_winner() {
        // The LocalHub#76 regression: before the fix a dense model was swept on
        // the no-op NCpuMoe axis and no candidate survived. Now the VRAM-fit
        // phase shrinks the KV cache / offloads layers and a winner emerges.
        let space = dense_space();
        assert!(!space.is_moe, "expert_count 0 classifies as dense");
        assert_eq!(space.block_count, 65);

        let seeds = localbench_search::seeds::resolve_smart_seeds(
            &space,
            localbench_search::seeds::HostFacts {
                vram_gb: 16,
                logical_cores: 16,
                available_ram_gb: 64.0,
                gguf_size_gb: 12.0,
            },
            Profile::Pure,
        );
        // The dense `-ngl` ladder is owned by the VRAM-fit phase's
        // `dense_recovery_candidates`, not the seeds — which carry none.
        assert!(seeds.offload_candidates.is_empty());

        let params = TunerParams {
            mode: localx_llama_core::Mode::Turboquant,
            ..params()
        };
        let mut runner = DenseVramRunner { measured: vec![] };
        let mut events = Vec::new();
        let outcome = run_tuner(&mut runner, &space, &seeds, &ctx(), &params, &mut |line| {
            events.push(line)
        });

        let outcome = outcome.expect("a dense model must produce a winner");
        assert!(events.iter().any(|e| e.starts_with("winner:")));
        // The VRAM-fit phase tried dense levers (a KV shrink or an NGpuLayers
        // offload), never the no-op NCpuMoe axis.
        let vram_fit: Vec<&Overrides> = runner
            .measured
            .iter()
            .filter(|(phase, _)| phase == "vram-fit")
            .map(|(_, ov)| ov)
            .collect();
        assert!(!vram_fit.is_empty(), "the dense VRAM-fit phase ran trials");
        assert!(
            vram_fit.iter().all(|ov| !ov.contains_key("NCpuMoe")),
            "a dense model is never swept on the NCpuMoe axis"
        );
        assert!(
            vram_fit
                .iter()
                .any(|ov| ov.contains_key("NGpuLayers") || ov.contains_key("KvK")),
            "the dense VRAM-fit phase tries KV-shrink / layer-offload levers"
        );
        // A dense winner never carries an NCpuMoe override.
        assert!(!outcome.winner.overrides.contains_key("NCpuMoe"));
    }

    /// A dense runner whose baseline already fits: every config starts.
    struct DenseHealthyRunner {
        measured: Vec<(String, Overrides)>,
    }

    impl TrialRunner for DenseHealthyRunner {
        fn measure(&mut self, overrides: &Overrides, phase: &str) -> Trial {
            self.measured.push((phase.to_string(), overrides.clone()));
            Trial {
                startup_ok: true,
                oom: false,
                measurement_usable: true,
                pp_tps: 400.0,
                tg_tps: 60.0,
                long_ctx_pp_tps: None,
                long_ctx_tg_tps: None,
                long_ctx_target_tokens: None,
                variance: Some(0.02),
                startup_failure: None,
                telemetry: Telemetry::default(),
                ..Trial::default()
            }
        }
    }

    #[test]
    fn a_dense_model_whose_baseline_starts_spends_no_vram_fit_trials() {
        // VRAM-fit is a recovery ladder, not an optimization sweep: a dense
        // model that already fits every layer on the GPU must not be dragged
        // down the `-ngl` ladder (LocalHub#76 required behaviour).
        let space = dense_space();
        let seeds = localbench_search::seeds::resolve_smart_seeds(
            &space,
            localbench_search::seeds::HostFacts {
                vram_gb: 24,
                logical_cores: 16,
                available_ram_gb: 64.0,
                gguf_size_gb: 12.0,
            },
            Profile::Pure,
        );
        let params = TunerParams {
            mode: localx_llama_core::Mode::Turboquant,
            ..params()
        };
        let mut runner = DenseHealthyRunner { measured: vec![] };
        let outcome = run_tuner(&mut runner, &space, &seeds, &ctx(), &params, &mut |_| {});
        assert!(outcome.is_some(), "a healthy dense baseline still wins");
        let vram_fit = runner
            .measured
            .iter()
            .filter(|(phase, _)| phase == "vram-fit")
            .count();
        assert_eq!(
            vram_fit, 0,
            "a dense model whose baseline starts spends no VRAM-fit trials"
        );
    }
}
