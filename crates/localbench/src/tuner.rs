//! The findbest orchestration: drive the search phases over a
//! [`TrialRunner`], retain the strongest candidates as a beam between phases,
//! and hand back a verified winner ready for the best-config export.
//!
//! Phase order mirrors the shipped tuner: baseline → VRAM-fit → batching →
//! flash-attention → memory flags → SWA/cache flags → threads (CPU-offload
//! only) → KV types → MoE refinement → verification. Every phase spends from
//! one trial budget and dedups by config signature.

use std::cell::Cell;
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
    moe_coverage_worklist, moe_edge_refine_values, recovery_n_cpu_moe_candidates,
    resolve_allowed_kv_types, resolve_tuner_budget, seed_failed, swa_flag_overlays, KvPair,
    SearchSpace,
};
use localx_llama_core::fit::FitPlacement;
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
    /// Try n-gram speculative decoding (drafts from the context itself, no
    /// draft model). Off for a model whose catalog already sets a
    /// speculative type or draft model.
    pub spec_ngram: bool,
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

/// Where a candidate fits in device memory, answered by llama.cpp's own
/// memory fitter without starting a server.
///
/// The answer depends on the candidate's memory shape (context, KV types,
/// batch sizes, flash attention, load mode) and never on its placement
/// (`NCpuMoe` / `NGpuLayers`), which is what it decides.
pub trait FitOracle {
    /// The fitted placement for `overrides`, or `None` when the fitter could
    /// not answer.
    fn placement(&mut self, overrides: &Overrides) -> Option<FitPlacement>;

    /// Why llama.cpp cannot create this configuration at all, when the fitter
    /// ran and said so (for example a quantized V cache without flash
    /// attention). `None` when it fits or the fitter did not answer.
    fn rejection(&mut self, _overrides: &Overrides) -> Option<String> {
        None
    }
}

/// Orders a phase's candidates by `llama-bench` throughput so only the most
/// promising get a server measurement. A screen result is never a
/// measurement: it is not scored, cached, ranked against trials, or saved.
pub trait BenchScreen {
    /// Candidate indices, most promising first, or `None` when `llama-bench`
    /// could not answer (the phase then measures every candidate).
    fn rank(&mut self, candidates: &[Overrides]) -> Option<Vec<usize>>;
}

/// The optional helpers that bound the search. With neither, the search is
/// the trial-only one.
#[derive(Default)]
pub struct TunerAids<'a> {
    /// llama.cpp's memory fitter: where a candidate fits.
    pub oracle: Option<&'a mut dyn FitOracle>,
    /// `llama-bench`: which candidates of a phase are worth measuring.
    pub screen: Option<&'a mut dyn BenchScreen>,
}

/// Candidates per parent a screened phase still measures on the server: the
/// screen's best and runner-up, so a screen misjudgement costs one rank.
const SCREEN_KEEP: usize = 2;

/// Server trials a screen must save before it is worth running. One
/// `llama-bench` process loads the model and sweeps the grid; on a model that
/// loads in seconds that costs about two server trials, so smaller phases are
/// measured directly.
const SCREEN_MIN_SAVED: usize = 3;

/// How many placement steps past the oracle's edge the VRAM-fit phase probes.
/// The fitter keeps a free-memory margin, so one or two more layers of
/// experts on the GPU often still start; the first failure ends the probe.
const ORACLE_PROBE_STEPS: i64 = 2;

/// A probe step scoring below this fraction of the step before it has spilled
/// VRAM into system memory (Windows drivers fall back instead of failing):
/// it counts as past the edge. Measurement noise stays well above it; a spill
/// typically loses two thirds of the throughput.
const ORACLE_SPILL_RATIO: f64 = 0.8;

/// How far the VRAM-fit phase backs off when the oracle's own placement runs
/// out of memory (the oracle missed): one step at a time, this many at most.
const ORACLE_RECOVERY_STEPS: i64 = 3;

/// The refine grid around a retained `NCpuMoe` when the oracle bounded the
/// search: the edge is already known, so only its neighbours are worth a trial.
const ORACLE_REFINE_RADIUS: i64 = 2;

/// Failed fresh verification can discard a winner and try the next beam
/// candidate; reserve the complete retry ladder before spending on search.
const MAX_VERIFICATION_ATTEMPTS: i64 = 3;

/// The search phases in the order `run_tuner` runs them, so a phase can leave
/// room for the ones that come after it. `verify` is not here: it spends from
/// the reserve this list is subtracted from.
const SEARCH_PHASES: [&str; 11] = [
    "baseline",
    "kv-recovery",
    "vram-fit",
    "batching",
    "flash-attn",
    "memory-flags",
    "cache-flags",
    "threads",
    "kv-types",
    "spec-ngram",
    "refine",
];

/// The n-gram speculative type the tuner tries: a shared hash pool of the
/// context's n-grams, cheap to keep and strongest on repetitive text such as
/// code being edited.
const NGRAM_SPEC_TYPE: &str = "ngram-mod";

/// Trials the KV-type recovery may spend. The recovery exists to prove a
/// working KV pair exists, not to search for the best one — the `kv-types`
/// phase does that later, from a baseline that can produce text. Four covers
/// the identical/turbo3/turbo4/crossed pairs a turbo-capable build offers
/// without letting a broken model spend the run proving it is broken.
const KV_RECOVERY_MAX_TRIALS: usize = 4;

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

/// The memory-flag candidates worth measuring on this host: loading into RAM
/// (`NoMmap`) and locking (`Mlock`) only where the seeds say the RAM is there.
/// The launcher spells them for the target build (`--load-mode` or the legacy
/// flags), so these stay build-independent intents.
fn memory_flag_overlays(
    recommendation: localbench_search::seeds::MmapRecommendation,
) -> Vec<Overrides> {
    let mut overlays = Vec::new();
    if recommendation.mlock {
        overlays.push(overrides_of(&[("Mlock", json!(true))]));
    }
    if recommendation.no_mmap {
        overlays.push(overrides_of(&[("NoMmap", json!(true))]));
    }
    if recommendation.mlock && recommendation.no_mmap {
        overlays.push(overrides_of(&[
            ("Mlock", json!(true)),
            ("NoMmap", json!(true)),
        ]));
    }
    overlays
}

/// One progress line describing the oracle's placement.
fn oracle_note(fit: &FitPlacement, is_moe: bool) -> String {
    let device = fit
        .devices
        .first()
        .map(|d| {
            format!(
                " ({} MiB used, {} MiB free on {})",
                d.used_mib, d.free_mib, d.name
            )
        })
        .unwrap_or_default();
    if is_moe {
        format!("oracle: fits with NCpuMoe={}{device}", fit.n_cpu_moe())
    } else if fit.gpu_layers < 0 {
        format!("oracle: every layer fits on the GPU{device}")
    } else {
        format!("oracle: fits with NGpuLayers={}{device}", fit.gpu_layers)
    }
}

/// The candidate with its placement moved to what its memory shape needs
/// (per the oracle, corrected by the slack this host showed), when it asked
/// for less. `None` when it already fits or the oracle cannot answer.
fn refit_for_shape(
    oracle: &mut dyn FitOracle,
    overrides: &Overrides,
    is_moe: bool,
    slack: i64,
) -> Option<(Overrides, String)> {
    let fit = oracle.placement(overrides)?;
    let current = |key: &str| overrides.get(key).and_then(serde_json::Value::as_i64);
    if is_moe {
        let floor = (fit.n_cpu_moe() - slack).max(0);
        let asked = current("NCpuMoe").unwrap_or(0);
        (asked < floor).then(|| {
            (
                join_overrides(overrides, &overrides_of(&[("NCpuMoe", json!(floor))])),
                format!("NCpuMoe {asked} -> {floor}: this memory shape needs more of the model on the CPU"),
            )
        })
    } else {
        if fit.gpu_layers < 0 {
            return None;
        }
        let cap = (fit.gpu_layers + slack).max(1);
        let asked = current("NGpuLayers");
        asked.map_or(true, |layers| layers > cap).then(|| {
            (
                join_overrides(overrides, &overrides_of(&[("NGpuLayers", json!(cap))])),
                format!(
                    "NGpuLayers {} -> {cap}: this memory shape does not fit every layer",
                    asked.map_or_else(|| "all".to_string(), |l| l.to_string())
                ),
            )
        })
    }
}

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
    run_tuner_with(
        runner,
        space,
        seeds,
        ctx,
        params,
        TunerAids::default(),
        events,
    )
}

/// [`run_tuner`] with llama.cpp's memory fitter and `llama-bench` bounding
/// the search.
///
/// The screen (`llama-bench`) orders the batching, flash-attention, threads,
/// and KV-type candidates in one process per parent, and only the top
/// [`SCREEN_KEEP`] of each go to a server measurement; memory and cache flags
/// keep server-only semantics and are measured as before.
///
/// With an oracle, the baseline starts at the fitted placement instead of the
/// catalog default; the VRAM-fit phase probes at most [`ORACLE_PROBE_STEPS`]
/// past it (and backs off at most [`ORACLE_RECOVERY_STEPS`] when the fitted
/// placement itself runs out of memory) instead of finding the edge by
/// provoking OOMs; later phases that change the memory shape get their
/// placement raised to what that shape needs; and the refine grid shrinks to
/// the edge's neighbours. The oracle never produces a result: every number
/// still comes from a real server measurement. Without one, the search is
/// unchanged.
pub fn run_tuner_with(
    runner: &mut dyn TrialRunner,
    space: &SearchSpace,
    seeds: &SmartSeeds,
    ctx: &ScoringContext,
    params: &TunerParams,
    aids: TunerAids<'_>,
    events: &mut dyn FnMut(String),
) -> Option<TunerOutcome> {
    let TunerAids {
        mut oracle,
        mut screen,
    } = aids;
    let budget = resolve_tuner_budget(params.budget);
    let search_budget = budget.saturating_sub(MAX_VERIFICATION_ATTEMPTS).max(1);
    let beam_width = params.beam_width.max(1);
    events(format!("search: beam width {beam_width}"));
    let mut trials = 0_usize;
    let mut history: Vec<Candidate> = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut phase_gate: Option<PhaseGate> = None;

    // ----- Phase 1 setup: the baseline, placed by the oracle when there is one -----
    let mut baseline = overrides_of(&[
        ("KvK", json!(params.baseline_kv.k.clone())),
        ("KvV", json!(params.baseline_kv.v.clone())),
    ]);
    let fitted = oracle.as_deref_mut().and_then(|o| o.placement(&baseline));
    if let Some(fit) = &fitted {
        events(oracle_note(fit, space.is_moe));
    } else if oracle.is_some() {
        events(
            "oracle: llama-fit-params gave no answer; searching the placement by trial".to_string(),
        );
    }
    // The oracle's edge for the baseline shape: `NCpuMoe` for a MoE model,
    // `NGpuLayers` (-1 = every layer) for a dense one.
    let oracle_edge: Option<i64> = fitted.as_ref().map(|fit| {
        if space.is_moe {
            fit.n_cpu_moe()
        } else {
            fit.gpu_layers
        }
    });
    if space.is_moe {
        let n_cpu_moe = oracle_edge.unwrap_or(space.baseline_n_cpu_moe);
        baseline = join_overrides(&baseline, &overrides_of(&[("NCpuMoe", json!(n_cpu_moe))]));
    } else if let Some(layers) = oracle_edge.filter(|layers| *layers >= 0) {
        baseline = join_overrides(&baseline, &overrides_of(&[("NGpuLayers", json!(layers))]));
    }
    // How far below (positive) or above (negative) the oracle's edge this
    // host actually runs, learned in the VRAM-fit phase and applied to every
    // later memory shape.
    let slack = Cell::new(0_i64);
    let oracle_active = oracle_edge.is_some();

    // The oracle is shared by the measure step (placement, rejections) and the
    // screen (which must never be handed a configuration llama.cpp rejects:
    // one invalid variant aborts the whole llama-bench process).
    let oracle = std::cell::RefCell::new(oracle);
    let rejected = |overrides: &Overrides| -> Option<String> {
        if !oracle_active {
            return None;
        }
        oracle
            .borrow_mut()
            .as_deref_mut()
            .and_then(|o| o.rejection(overrides))
    };
    // What each phase has already had ranked: a candidate the screen placed
    // below the cut for one parent is not re-screened for another.
    let mut already_screened: BTreeSet<String> = BTreeSet::new();
    let mut screened = |phase: &str,
                        candidates: Vec<Overrides>,
                        keep: usize,
                        seen: &BTreeSet<String>,
                        events: &mut dyn FnMut(String)|
     -> Vec<Overrides> {
        let Some(screen) = screen.as_deref_mut() else {
            return candidates;
        };
        // Screen what the server would actually run: each candidate at the
        // placement its memory shape needs, minus anything this phase has
        // already measured — a parent past the edge (a spill) would otherwise
        // cost a whole screen for candidates that all collapse into trials
        // already taken.
        let mut unique: Vec<Overrides> = Vec::new();
        for candidate in candidates {
            let placed = if oracle_active {
                oracle
                    .borrow_mut()
                    .as_deref_mut()
                    .and_then(|o| refit_for_shape(o, &candidate, space.is_moe, slack.get()))
                    .map_or(candidate, |(adjusted, _)| adjusted)
            } else {
                candidate
            };
            let signature = candidate_signature(&placed);
            let key = format!("{phase}|{signature}");
            if seen.contains(&key)
                || already_screened.contains(&key)
                || unique.iter().any(|c| candidate_signature(c) == signature)
            {
                continue;
            }
            unique.push(placed);
        }
        let candidates = unique;
        // Configurations llama.cpp cannot create go straight to the measure
        // step, which skips and reports them; they are never screened.
        let (invalid, valid): (Vec<Overrides>, Vec<Overrides>) =
            candidates.into_iter().partition(|c| rejected(c).is_some());
        if valid.len() < keep + SCREEN_MIN_SAVED {
            return valid.into_iter().chain(invalid).collect();
        }
        match screen.rank(&valid) {
            Some(order) => {
                already_screened.extend(
                    valid
                        .iter()
                        .map(|c| format!("{phase}|{}", candidate_signature(c))),
                );
                events(format!(
                    "screen [{phase}]: llama-bench ranked {} candidates; measuring the top {keep}",
                    valid.len()
                ));
                order
                    .into_iter()
                    .take(keep)
                    .filter_map(|index| valid.get(index).cloned())
                    .chain(invalid)
                    .collect()
            }
            None => {
                events(format!(
                    "screen [{phase}]: llama-bench gave no answer; measuring all {}",
                    valid.len()
                ));
                valid.into_iter().chain(invalid).collect()
            }
        }
    };

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
        // A configuration llama.cpp cannot even create is not worth a trial.
        if oracle_active && !matches!(phase, "baseline" | "vram-fit") {
            if let Some(reason) = rejected(overrides) {
                if seen.insert(format!(
                    "{phase}|rejected|{}",
                    candidate_signature(overrides)
                )) {
                    events(format!(
                        "oracle [{phase}]: skipped {} — llama.cpp cannot create it: {reason}",
                        candidate_signature(overrides)
                    ));
                }
                return None;
            }
        }
        // A phase that changes the memory shape (KV type, batch, flags) keeps
        // its intent but gets the placement that shape needs, instead of an
        // OOM trial that only proves the old placement no longer fits.
        let refitted = if oracle_active && !matches!(phase, "baseline" | "vram-fit") {
            oracle
                .borrow_mut()
                .as_deref_mut()
                .and_then(|o| refit_for_shape(o, overrides, space.is_moe, slack.get()))
        } else {
            None
        };
        let overrides = match &refitted {
            Some((adjusted, note)) => {
                events(format!("oracle [{phase}]: {note}"));
                adjusted
            }
            None => overrides,
        };
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
    let baseline_candidate = measure(
        &baseline,
        "baseline",
        &mut trials,
        &mut history,
        &mut seen,
        events,
    );
    let baseline_trial = baseline_candidate
        .as_ref()
        .and_then(|candidate| candidate.trial.as_ref());
    let baseline_usable = baseline_trial.is_some_and(Trial::is_measurement_usable);
    let baseline_needs_recovery = baseline_trial.is_some_and(Trial::needs_memory_recovery);
    let baseline_needs_kv_recovery =
        !baseline_usable && baseline_trial.is_some_and(Trial::needs_kv_recovery);
    let baseline_seed_failed = seed_failed(baseline_candidate.as_ref());
    if !baseline_usable && !baseline_needs_recovery && !baseline_needs_kv_recovery {
        events(
            "stopped: baseline reached no usable measurement and supplied no startup/OOM fit evidence; fix the reported contract/content failure before retuning"
                .to_string(),
        );
        return None;
    }

    // ----- Phase 1b: KV-type recovery -----
    // A baseline that started, stayed inside memory, and still produced text
    // the content gates rejected has a likely culprit the search already knows
    // about: the KV cache pair. Recovering here rather than by widening
    // `needs_memory_recovery` keeps two things straight. A content failure is
    // never reported as memory pressure, and the pair adopted below becomes
    // the one the MoE vram-fit phase pins while it sweeps the expert lever —
    // so the rest of the run cannot spend its budget re-measuring a pair that
    // cannot produce text.
    let mut effective_kv = params.baseline_kv.clone();
    if baseline_needs_kv_recovery {
        events("phase: kv-recovery".to_string());
        let allowed = resolve_allowed_kv_types(&[], &params.baseline_kv, space_mode(params.mode));
        let alternatives: Vec<KvPair> = kv_candidate_pairs(&allowed, false, false)
            .into_iter()
            .filter(|pair| *pair != params.baseline_kv)
            .take(KV_RECOVERY_MAX_TRIALS)
            .collect();
        if alternatives.is_empty() {
            events(
                "kv-recovery: the allowed KV cache set offers no alternative to the baseline pair"
                    .to_string(),
            );
        }
        let mut recovered: Option<KvPair> = None;
        for pair in alternatives {
            let overrides = join_overrides(
                &baseline,
                &overrides_of(&[
                    ("KvK", json!(pair.k.clone())),
                    ("KvV", json!(pair.v.clone())),
                ]),
            );
            let candidate = measure(
                &overrides,
                "kv-recovery",
                &mut trials,
                &mut history,
                &mut seen,
                events,
            );
            if candidate
                .as_ref()
                .and_then(|candidate| candidate.trial.as_ref())
                .is_some_and(Trial::is_measurement_usable)
            {
                recovered = Some(pair);
                break;
            }
        }
        let Some(pair) = recovered else {
            events(
                "stopped: the baseline failed on content and every allowed KV cache pair failed the same way; the model or engine, not the KV cache type, is what needs fixing"
                    .to_string(),
            );
            return None;
        };
        events(format!(
            "kv-recovery: baseline recovered on KvK={};KvV={}; the rest of the run uses it",
            pair.k, pair.v
        ));
        baseline = join_overrides(
            &baseline,
            &overrides_of(&[
                ("KvK", json!(pair.k.clone())),
                ("KvV", json!(pair.v.clone())),
            ]),
        );
        effective_kv = pair;
    }

    // ----- Phase 2: VRAM fit -----
    events("phase: vram-fit".to_string());
    if let Some(edge) = oracle_edge {
        let usable = |candidate: &Option<Candidate>| {
            candidate
                .as_ref()
                .and_then(|c| c.trial.as_ref())
                .is_some_and(Trial::is_measurement_usable)
        };
        // A MoE model gets faster as experts move to the GPU (NCpuMoe down);
        // a dense one as layers do (NGpuLayers up).
        let (key, toward_gpu, lower, upper) = if space.is_moe {
            ("NCpuMoe", -1_i64, 0_i64, space.moe_upper)
        } else {
            ("NGpuLayers", 1_i64, 1_i64, space.block_count.max(1) + 1)
        };
        if space.is_moe || edge >= 0 {
            if baseline_usable {
                let mut previous = baseline_candidate
                    .as_ref()
                    .map_or(0.0, |candidate| candidate.selected_score);
                for step in 1..=ORACLE_PROBE_STEPS {
                    let value = edge + toward_gpu * step;
                    if value < lower || value > upper {
                        break;
                    }
                    let probe = join_overrides(&baseline, &overrides_of(&[(key, json!(value))]));
                    let measured = measure(
                        &probe,
                        "vram-fit",
                        &mut trials,
                        &mut history,
                        &mut seen,
                        events,
                    );
                    if !usable(&measured) {
                        break;
                    }
                    // More of the model on the GPU never makes a run slower —
                    // unless VRAM is overcommitted and the driver quietly spills
                    // to system memory, which starts but crawls. That step is
                    // past the real edge even though it did not OOM.
                    let score = measured.as_ref().map_or(0.0, |c| c.selected_score);
                    if score < previous * ORACLE_SPILL_RATIO {
                        events(format!(
                            "oracle: {key}={value} started but scored {score:.1} against {previous:.1} one step earlier — VRAM spilled to system memory; the edge is the step before"
                        ));
                        break;
                    }
                    previous = score;
                    slack.set(step);
                }
                events(format!(
                    "oracle: this host runs {} step(s) past the fitted {key}={edge}",
                    slack.get()
                ));
            } else if baseline_needs_recovery {
                events(format!(
                    "oracle miss: {key}={edge} did not start; backing off one step at a time"
                ));
                for step in 1..=ORACLE_RECOVERY_STEPS {
                    let value = edge - toward_gpu * step;
                    if value < lower || value > upper {
                        break;
                    }
                    let backoff = join_overrides(&baseline, &overrides_of(&[(key, json!(value))]));
                    if usable(&measure(
                        &backoff,
                        "vram-fit",
                        &mut trials,
                        &mut history,
                        &mut seen,
                        events,
                    )) {
                        slack.set(-step);
                        break;
                    }
                }
            }
        }
    } else if space.is_moe {
        // The MoE sweep has no floor: the only mode that ever imposed one was
        // mtpturbo, whose draft head competed with the main model for VRAM.
        let minimum = 0;
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
        let coverage_pairs = vec![effective_kv.clone()];
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
        let allowed = resolve_allowed_kv_types(&[], &effective_kv, space_mode(params.mode));
        let kv_pairs = kv_candidate_pairs(&allowed, false, false);
        for overrides in dense_recovery_candidates(
            &effective_kv,
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
    let batch_of = |overrides: &Overrides| {
        let read = |key: &str| {
            overrides
                .get(key)
                .and_then(serde_json::Value::as_i64)
                .unwrap_or(0)
        };
        (read("UbatchSize"), read("BatchSize"))
    };
    for parent in beam_so_far(&history) {
        let mut grid = Vec::new();
        for &ub in &space.ubatch_candidates {
            for &b in &space.batch_candidates {
                if b >= ub {
                    grid.push(join_overrides(
                        &parent.overrides,
                        &overrides_of(&[("UbatchSize", json!(ub)), ("BatchSize", json!(b))]),
                    ));
                }
            }
        }
        let mut oomed_batching: Vec<(i64, i64)> = Vec::new();
        for overrides in screened("batching", grid, SCREEN_KEEP, &seen, events) {
            let (ub, b) = batch_of(&overrides);
            if batching_dominated(ub, b, &oomed_batching) {
                continue;
            }
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
            memory_flag_overlays(seeds.mmap_recommendation),
        ),
        ("cache-flags", swa_flag_overlays()),
    ];
    for (phase, overlays) in flag_phases {
        events(format!("phase: {phase}"));
        if overlays.is_empty() {
            events(format!(
                "{phase}: skipped — not enough free RAM to load or lock this model in memory"
            ));
            continue;
        }
        let beam = beam_so_far(&history);
        let mut candidates = expand_phase_candidates(&beam, &overlays);
        if phase == "flash-attn" {
            // Flash attention is a two-way choice: keep the screen's pick per parent.
            candidates = screened(phase, candidates, beam.len().max(1), &seen, events);
        }
        for overrides in candidates {
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
            let candidates: Vec<Overrides> = seeds
                .thread_candidates
                .iter()
                .map(|threads| {
                    join_overrides(
                        &parent.overrides,
                        &overrides_of(&[("Threads", json!(threads))]),
                    )
                })
                .collect();
            for overrides in screened("threads", candidates, SCREEN_KEEP, &seen, events) {
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
        let allowed = resolve_allowed_kv_types(&[], &effective_kv, space_mode(params.mode));
        let candidates: Vec<Overrides> = kv_candidate_pairs(&allowed, false, false)
            .into_iter()
            .map(|pair| {
                join_overrides(
                    &parent.overrides,
                    &overrides_of(&[("KvK", json!(pair.k)), ("KvV", json!(pair.v))]),
                )
            })
            .collect();
        for overrides in screened("kv-types", candidates, SCREEN_KEEP, &seen, events) {
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

    // ----- Phase 8b: n-gram speculative decoding -----
    // Drafting from the context's own n-grams needs no draft model, so any
    // model can try it; it is kept only if the measured score wins.
    if params.spec_ngram {
        events("phase: spec-ngram".to_string());
        let beam = beam_so_far(&history);
        let overlay = [overrides_of(&[("SpecType", json!(NGRAM_SPEC_TYPE))])];
        for overrides in expand_phase_candidates(&beam, &overlay) {
            measure(
                &overrides,
                "spec-ngram",
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
                if oracle_active {
                    values.retain(|value| (value - current).abs() <= ORACLE_REFINE_RADIUS);
                } else {
                    for value in
                        moe_edge_refine_values(&measured_stable, current, 5, 8, 0, space.moe_upper)
                    {
                        if !values.contains(&value) {
                            values.push(value);
                        }
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
            spec_ngram: false,
        }
    }

    /// A runner that mimics LocalHub#159: one KV pair starts, runs, reports
    /// healthy timings and returns a `/` flood; every other pair answers.
    struct DegenerateKvRunner {
        broken: KvPair,
        measured: Vec<(String, String)>,
    }

    impl DegenerateKvRunner {
        fn pair_of(overrides: &Overrides) -> KvPair {
            let read = |key: &str| {
                overrides
                    .get(key)
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_default()
                    .to_string()
            };
            KvPair {
                k: read("KvK"),
                v: read("KvV"),
            }
        }
    }

    impl TrialRunner for DegenerateKvRunner {
        fn measure(&mut self, overrides: &Overrides, phase: &str) -> Trial {
            let pair = Self::pair_of(overrides);
            self.measured
                .push((phase.to_string(), format!("{};{}", pair.k, pair.v)));
            if pair == self.broken {
                // The shape that matters: the process is healthy in every way
                // the tuner can see, and only the text is wrong.
                return Trial {
                    startup_ok: true,
                    oom: false,
                    measurement_usable: false,
                    process_status: Some(1),
                    failure: Some(crate::trial::content_failure_for_test()),
                    ..Trial::default()
                };
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
                variance: Some(0.02),
                telemetry: Telemetry::default(),
                ..Trial::default()
            }
        }
    }

    /// LocalHub#160. A baseline that starts, stays inside memory, and returns
    /// degenerate text used to end the run at trial one — while the identical
    /// defect surfacing as a readiness timeout recovered and finished. The
    /// working pair was already in the allowed set the search had built.
    #[test]
    fn a_degenerate_baseline_recovers_on_another_kv_pair_instead_of_ending_the_run() {
        let mut runner = DegenerateKvRunner {
            broken: KvPair {
                k: "q8_0".to_string(),
                v: "q8_0".to_string(),
            },
            measured: vec![],
        };
        let mut events = Vec::new();
        let mut turbo = params();
        turbo.mode = localx_llama_core::Mode::Turboquant;
        let outcome = run_tuner(
            &mut runner,
            &space(),
            &seeds(),
            &ctx(),
            &turbo,
            &mut |line| events.push(line),
        )
        .expect("a degenerate baseline recovers onto a working KV pair");

        assert!(
            runner
                .measured
                .iter()
                .any(|(phase, _)| phase == "kv-recovery"),
            "the recovery phase never ran: {:?}",
            runner.measured
        );
        assert!(
            events
                .iter()
                .any(|line| line.contains("kv-recovery: baseline recovered")),
            "recovery was not reported under its own name: {events:?}"
        );
        assert!(
            !events.iter().any(|line| line.starts_with("stopped:")),
            "the run stopped despite a working pair being available: {events:?}"
        );

        // The winner must not be the pair that cannot produce text.
        let winner_kv = DegenerateKvRunner::pair_of(&outcome.winner.overrides);
        assert_ne!(
            winner_kv, runner.broken,
            "the run selected the KV pair that returns degenerate text"
        );

        // And the rest of the run must follow the recovered pair rather than
        // re-measuring the broken one: that is what makes the recovery worth
        // more than one lucky trial.
        let broken_after_recovery = runner
            .measured
            .iter()
            .filter(|(phase, pair)| {
                phase != "baseline" && *pair == format!("{};{}", runner.broken.k, runner.broken.v)
            })
            .count();
        assert_eq!(
            broken_after_recovery, 0,
            "later phases kept re-measuring the broken pair: {:?}",
            runner.measured
        );
    }

    /// When no allowed pair works, stopping is right — but the message has to
    /// say that every pair was tried, not that the baseline supplied no
    /// evidence.
    #[test]
    fn an_unrecoverable_content_failure_stops_only_after_every_pair_and_says_so() {
        struct AlwaysDegenerate;
        impl TrialRunner for AlwaysDegenerate {
            fn measure(&mut self, _overrides: &Overrides, _phase: &str) -> Trial {
                Trial {
                    startup_ok: true,
                    oom: false,
                    measurement_usable: false,
                    failure: Some(crate::trial::content_failure_for_test()),
                    ..Trial::default()
                }
            }
        }
        let mut events = Vec::new();
        let mut turbo = params();
        turbo.mode = localx_llama_core::Mode::Turboquant;
        let outcome = run_tuner(
            &mut AlwaysDegenerate,
            &space(),
            &seeds(),
            &ctx(),
            &turbo,
            &mut |line| events.push(line),
        );
        assert!(
            outcome.is_none(),
            "an unusable model must not produce a winner"
        );
        assert!(
            events
                .iter()
                .any(|line| line.contains("every allowed KV cache pair failed the same way")),
            "the stop message did not say the recovery was exhausted: {events:?}"
        );
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

    #[test]
    fn memory_flags_follow_the_host_ram_recommendation() {
        use localbench_search::seeds::MmapRecommendation;
        let keys = |overlays: Vec<Overrides>| -> Vec<Vec<String>> {
            overlays
                .into_iter()
                .map(|o| o.keys().cloned().collect())
                .collect()
        };
        let all = memory_flag_overlays(MmapRecommendation {
            mlock: true,
            no_mmap: true,
        });
        assert_eq!(
            keys(all),
            vec![
                vec!["Mlock".to_string()],
                vec!["NoMmap".to_string()],
                vec!["Mlock".to_string(), "NoMmap".to_string()],
            ]
        );
        let no_lock = memory_flag_overlays(MmapRecommendation {
            mlock: false,
            no_mmap: true,
        });
        assert_eq!(keys(no_lock), vec![vec!["NoMmap".to_string()]]);
        assert!(memory_flag_overlays(MmapRecommendation {
            mlock: false,
            no_mmap: false,
        })
        .is_empty());
    }

    /// A host whose real VRAM edge depends on the KV type: MoE configs need at
    /// least `edge(kv)` expert blocks on the CPU and OOM below it; a dense
    /// model OOMs above `dense_max` GPU layers. Faster with more on the GPU.
    struct VramRunner {
        moe_edge_q8: i64,
        dense_max: i64,
        measured: Vec<(String, Overrides, bool)>,
    }

    fn kv_extra(overrides: &Overrides) -> i64 {
        match overrides.get("KvK").and_then(serde_json::Value::as_str) {
            Some("f16") => 3,
            _ => 0,
        }
    }

    impl TrialRunner for VramRunner {
        fn measure(&mut self, overrides: &Overrides, phase: &str) -> Trial {
            let get = |key: &str| overrides.get(key).and_then(serde_json::Value::as_i64);
            let (fits, speed) = match get("NGpuLayers") {
                Some(layers) => (layers <= self.dense_max, layers as f64),
                None => match get("NCpuMoe") {
                    Some(moe) => (
                        moe >= self.moe_edge_q8 + kv_extra(overrides),
                        60.0 - moe as f64,
                    ),
                    None => (self.dense_max >= 65, 65.0),
                },
            };
            self.measured
                .push((phase.to_string(), overrides.clone(), fits));
            if !fits {
                return failed_trial(true);
            }
            Trial {
                startup_ok: true,
                oom: false,
                measurement_usable: true,
                pp_tps: 500.0 + speed,
                tg_tps: 20.0 + speed,
                variance: Some(0.01),
                telemetry: Telemetry::default(),
                ..Trial::default()
            }
        }
    }

    /// An oracle that answers `offset` blocks off the real edge (positive =
    /// conservative, as llama-fit-params is with its free-memory margin).
    struct ScriptedOracle {
        moe_edge_q8: i64,
        offset: i64,
        dense_layers: Option<i64>,
        calls: usize,
    }

    impl FitOracle for ScriptedOracle {
        fn rejection(&mut self, overrides: &Overrides) -> Option<String> {
            let fa_off = overrides.get("FlashAttn") == Some(&json!(false));
            let quantized_v = overrides
                .get("KvV")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|v| v != "f16");
            (fa_off && quantized_v).then(|| "V cache quantization requires flash_attn".to_string())
        }

        fn placement(&mut self, overrides: &Overrides) -> Option<FitPlacement> {
            self.calls += 1;
            let moe = self.moe_edge_q8 + kv_extra(overrides) + self.offset;
            Some(FitPlacement {
                context: Some(262_144),
                gpu_layers: self.dense_layers.unwrap_or(49),
                cpu_expert_blocks: if self.dense_layers.is_some() {
                    Vec::new()
                } else {
                    (0..u32::try_from(moe).unwrap()).collect()
                },
                tensor_overrides: None,
                devices: Vec::new(),
            })
        }
    }

    fn ooms(runner: &VramRunner) -> Vec<String> {
        runner
            .measured
            .iter()
            .filter(|(_, _, fits)| !fits)
            .map(|(phase, overrides, _)| format!("{phase}:{}", candidate_signature(overrides)))
            .collect()
    }

    #[test]
    fn a_conservative_oracle_finds_the_edge_with_one_failed_probe() {
        let mut runner = VramRunner {
            moe_edge_q8: 30,
            dense_max: 999,
            measured: Vec::new(),
        };
        let mut oracle = ScriptedOracle {
            moe_edge_q8: 30,
            offset: 1,
            dense_layers: None,
            calls: 0,
        };
        let mut lines = Vec::new();
        let outcome = run_tuner_with(
            &mut runner,
            &space(),
            &seeds(),
            &ctx(),
            &params(),
            TunerAids {
                oracle: Some(&mut oracle),
                screen: None,
            },
            &mut |line| lines.push(line),
        )
        .expect("a winner");
        // The baseline starts at the oracle's placement, not the catalog's 20.
        let (phase, first, _) = &runner.measured[0];
        assert_eq!(phase, "baseline");
        assert_eq!(first.get("NCpuMoe"), Some(&json!(31)));
        // One step past the edge ran, the second failed — the only OOM in the
        // whole run: refine candidates below the proven edge are refitted.
        assert_eq!(ooms(&runner), vec!["vram-fit:KvK=q8_0;KvV=q8_0;NCpuMoe=29"]);
        assert!(lines
            .iter()
            .any(|l| l.contains("runs 1 step(s) past the fitted NCpuMoe=31")));
        // FlashAttn=false with a q8_0 V cache cannot be created: skipped, not run.
        assert!(lines.iter().any(
            |l| l.starts_with("oracle [flash-attn]: skipped") && l.contains("FlashAttn=false")
        ));
        assert!(!runner
            .measured
            .iter()
            .any(|(_, o, _)| o.get("FlashAttn") == Some(&json!(false))));
        assert!(
            lines
                .iter()
                .any(|l| l.contains("oracle [refine]: NCpuMoe 29 -> 30")),
            "{lines:#?}"
        );
        assert_eq!(
            outcome.winner.overrides.get("NCpuMoe"),
            Some(&json!(30)),
            "the fastest placement that starts"
        );
        let vram_fit = runner
            .measured
            .iter()
            .filter(|(p, _, _)| p == "vram-fit")
            .count();
        assert_eq!(vram_fit, 2, "no ladder: two probes past the edge");
        assert!(oracle.calls >= 1);
    }

    #[test]
    fn an_optimistic_oracle_backs_off_one_step_at_a_time() {
        let mut runner = VramRunner {
            moe_edge_q8: 30,
            dense_max: 999,
            measured: Vec::new(),
        };
        let mut oracle = ScriptedOracle {
            moe_edge_q8: 30,
            offset: -2,
            dense_layers: None,
            calls: 0,
        };
        let mut lines = Vec::new();
        let outcome = run_tuner_with(
            &mut runner,
            &space(),
            &seeds(),
            &ctx(),
            &params(),
            TunerAids {
                oracle: Some(&mut oracle),
                screen: None,
            },
            &mut |line| lines.push(line),
        )
        .expect("recovered");
        assert!(lines
            .iter()
            .any(|l| l.starts_with("oracle miss: NCpuMoe=28")));
        // Baseline 28 and back-off 29 fail; 30 starts. Later shapes inherit the
        // correction, so nothing after the VRAM-fit phase runs out of memory.
        assert_eq!(
            ooms(&runner),
            vec![
                "baseline:KvK=q8_0;KvV=q8_0;NCpuMoe=28",
                "vram-fit:KvK=q8_0;KvV=q8_0;NCpuMoe=29",
            ]
        );
        assert_eq!(outcome.winner.overrides.get("NCpuMoe"), Some(&json!(30)));
    }

    #[test]
    fn a_dense_model_starts_at_the_fitted_layer_count_and_probes_upward() {
        let mut runner = VramRunner {
            moe_edge_q8: 0,
            dense_max: 41,
            measured: Vec::new(),
        };
        let mut oracle = ScriptedOracle {
            moe_edge_q8: 0,
            offset: 0,
            dense_layers: Some(40),
            calls: 0,
        };
        let outcome = run_tuner_with(
            &mut runner,
            &dense_space(),
            &seeds(),
            &ctx(),
            &params(),
            TunerAids {
                oracle: Some(&mut oracle),
                screen: None,
            },
            &mut |_| {},
        )
        .expect("a winner");
        assert_eq!(runner.measured[0].1.get("NGpuLayers"), Some(&json!(40)));
        assert_eq!(
            ooms(&runner),
            vec!["vram-fit:KvK=q8_0;KvV=q8_0;NGpuLayers=42"]
        );
        assert_eq!(outcome.winner.overrides.get("NGpuLayers"), Some(&json!(41)));
    }

    #[test]
    fn an_oracle_without_an_answer_leaves_the_trial_search_in_charge() {
        struct Silent;
        impl FitOracle for Silent {
            fn placement(&mut self, _overrides: &Overrides) -> Option<FitPlacement> {
                None
            }
        }
        let mut with = ScriptedRunner {
            measured: Vec::new(),
        };
        let mut without = ScriptedRunner {
            measured: Vec::new(),
        };
        let mut silent = Silent;
        let a = run_tuner_with(
            &mut with,
            &space(),
            &seeds(),
            &ctx(),
            &params(),
            TunerAids {
                oracle: Some(&mut silent),
                screen: None,
            },
            &mut |_| {},
        );
        let b = run_tuner(
            &mut without,
            &space(),
            &seeds(),
            &ctx(),
            &params(),
            &mut |_| {},
        );
        assert_eq!(with.measured, without.measured);
        assert_eq!(
            a.map(|o| candidate_signature(&o.winner.overrides)),
            b.map(|o| candidate_signature(&o.winner.overrides))
        );
    }

    #[test]
    fn a_heavier_memory_shape_is_refitted_not_crashed() {
        let mut oracle = ScriptedOracle {
            moe_edge_q8: 30,
            offset: 1,
            dense_layers: None,
            calls: 0,
        };
        // Proven slack 1: this host runs one block past the fitted edge.
        let f16 = overrides_of(&[
            ("KvK", json!("f16")),
            ("KvV", json!("f16")),
            ("NCpuMoe", json!(30)),
        ]);
        let (adjusted, note) = refit_for_shape(&mut oracle, &f16, true, 1).unwrap();
        assert_eq!(adjusted.get("NCpuMoe"), Some(&json!(33)));
        assert!(note.starts_with("NCpuMoe 30 -> 33"), "{note}");
        let q8 = overrides_of(&[("KvK", json!("q8_0")), ("NCpuMoe", json!(30))]);
        assert!(refit_for_shape(&mut oracle, &q8, true, 1).is_none());
        let dense = ScriptedOracle {
            moe_edge_q8: 0,
            offset: 0,
            dense_layers: Some(40),
            calls: 0,
        };
        let mut dense = dense;
        let all_layers = overrides_of(&[("KvK", json!("q8_0"))]);
        let (capped, _) = refit_for_shape(&mut dense, &all_layers, false, 1).unwrap();
        assert_eq!(capped.get("NGpuLayers"), Some(&json!(41)));
    }

    /// Ranks batching candidates by ubatch size, largest first; answers
    /// nothing when `silent`.
    struct UbatchScreen {
        silent: bool,
        calls: usize,
    }

    impl BenchScreen for UbatchScreen {
        fn rank(&mut self, candidates: &[Overrides]) -> Option<Vec<usize>> {
            self.calls += 1;
            if self.silent {
                return None;
            }
            let mut order: Vec<usize> = (0..candidates.len()).collect();
            let ub = |i: &usize| {
                candidates[*i]
                    .get("UbatchSize")
                    .and_then(serde_json::Value::as_i64)
                    .unwrap_or(0)
            };
            order.sort_by_key(|i| std::cmp::Reverse(ub(i)));
            Some(order)
        }
    }

    fn batching_trials(runner: &ScriptedRunner) -> usize {
        runner.measured.iter().filter(|p| *p == "batching").count()
    }

    #[test]
    fn a_screened_phase_measures_only_the_top_candidates_per_parent() {
        let mut screened = ScriptedRunner {
            measured: Vec::new(),
        };
        let mut screen = UbatchScreen {
            silent: false,
            calls: 0,
        };
        let mut lines = Vec::new();
        run_tuner_with(
            &mut screened,
            &space(),
            &seeds(),
            &ctx(),
            &params(),
            TunerAids {
                oracle: None,
                screen: Some(&mut screen),
            },
            &mut |line| lines.push(line),
        )
        .expect("a winner");
        assert!(screen.calls > 0);
        let batching = batching_trials(&screened);
        assert!((1..=SCREEN_KEEP * DEFAULT_BEAM_WIDTH).contains(&batching));
        assert!(lines
            .iter()
            .any(|l| l.starts_with("screen [batching]: llama-bench ranked")));
        let largest = space().ubatch_candidates.iter().copied().max().unwrap();
        let first = lines
            .iter()
            .find(|l| l.contains("[batching]") && l.starts_with("trial "))
            .unwrap();
        assert!(
            first.contains(&format!("UbatchSize={largest}")),
            "the screen's top pick is measured first: {first}"
        );
    }

    #[test]
    fn a_screen_without_an_answer_measures_every_candidate() {
        let mut plain = ScriptedRunner {
            measured: Vec::new(),
        };
        run_tuner(
            &mut plain,
            &space(),
            &seeds(),
            &ctx(),
            &params(),
            &mut |_| {},
        );
        let mut fallback = ScriptedRunner {
            measured: Vec::new(),
        };
        let mut screen = UbatchScreen {
            silent: true,
            calls: 0,
        };
        run_tuner_with(
            &mut fallback,
            &space(),
            &seeds(),
            &ctx(),
            &params(),
            TunerAids {
                oracle: None,
                screen: Some(&mut screen),
            },
            &mut |_| {},
        );
        assert_eq!(fallback.measured, plain.measured);
    }

    #[test]
    fn ngram_speculation_is_measured_when_enabled_and_kept_only_if_it_wins() {
        struct SpecRunner {
            measured: Vec<(String, Overrides)>,
            boost: f64,
        }
        impl TrialRunner for SpecRunner {
            fn measure(&mut self, overrides: &Overrides, phase: &str) -> Trial {
                self.measured.push((phase.to_string(), overrides.clone()));
                let spec = overrides.get("SpecType").is_some();
                Trial {
                    startup_ok: true,
                    measurement_usable: true,
                    pp_tps: 500.0,
                    tg_tps: if spec { 20.0 + self.boost } else { 20.0 },
                    variance: Some(0.01),
                    ..Trial::default()
                }
            }
        }
        for (boost, expect_spec) in [(8.0, true), (-8.0, false)] {
            let mut runner = SpecRunner {
                measured: Vec::new(),
                boost,
            };
            let with = TunerParams {
                spec_ngram: true,
                ..params()
            };
            let outcome = run_tuner(&mut runner, &space(), &seeds(), &ctx(), &with, &mut |_| {})
                .expect("a winner");
            assert!(runner
                .measured
                .iter()
                .any(|(phase, o)| phase == "spec-ngram"
                    && o.get("SpecType") == Some(&json!("ngram-mod"))));
            assert_eq!(
                outcome.winner.overrides.contains_key("SpecType"),
                expect_spec,
                "boost {boost}"
            );
        }
        let mut off = SpecRunner {
            measured: Vec::new(),
            boost: 8.0,
        };
        run_tuner(&mut off, &space(), &seeds(), &ctx(), &params(), &mut |_| {});
        assert!(!off.measured.iter().any(|(phase, _)| phase == "spec-ngram"));
    }

    /// Records every candidate it is asked to rank; ranks in the given order.
    struct RecordingScreen {
        asked: Vec<Vec<Overrides>>,
    }

    impl BenchScreen for RecordingScreen {
        fn rank(&mut self, candidates: &[Overrides]) -> Option<Vec<usize>> {
            self.asked.push(candidates.to_vec());
            Some((0..candidates.len()).collect())
        }
    }

    #[test]
    fn the_screen_never_sees_rejected_configs_and_skips_small_phases() {
        let mut runner = VramRunner {
            moe_edge_q8: 30,
            dense_max: 999,
            measured: Vec::new(),
        };
        let mut oracle = ScriptedOracle {
            moe_edge_q8: 30,
            offset: 1,
            dense_layers: None,
            calls: 0,
        };
        let mut screen = RecordingScreen { asked: Vec::new() };
        let mut lines = Vec::new();
        run_tuner_with(
            &mut runner,
            &space(),
            &seeds(),
            &ctx(),
            &params(),
            TunerAids {
                oracle: Some(&mut oracle),
                screen: Some(&mut screen),
            },
            &mut |line| lines.push(line),
        )
        .expect("a winner");
        assert!(
            !screen.asked.is_empty(),
            "batching is large enough to screen"
        );
        for batch in &screen.asked {
            assert!(
                batch.len() >= SCREEN_KEEP + SCREEN_MIN_SAVED,
                "{}",
                batch.len()
            );
            assert!(
                !batch
                    .iter()
                    .any(|o| o.get("FlashAttn") == Some(&json!(false))),
                "a rejected config reached the screen"
            );
        }
        assert!(!lines.iter().any(|l| l.starts_with("screen [threads]")));
        assert!(!lines.iter().any(|l| l.starts_with("screen [flash-attn]")));
    }

    /// Past `edge - 1` the driver spills to system memory: the server starts
    /// but crawls instead of running out of memory.
    struct SpillRunner {
        edge: i64,
        measured: Vec<i64>,
    }

    impl TrialRunner for SpillRunner {
        fn measure(&mut self, overrides: &Overrides, _phase: &str) -> Trial {
            let moe = overrides
                .get("NCpuMoe")
                .and_then(serde_json::Value::as_i64)
                .unwrap_or(0);
            self.measured.push(moe);
            let tg = if moe >= self.edge - 1 {
                25.0 + (self.edge - moe) as f64
            } else {
                8.0
            };
            Trial {
                startup_ok: true,
                measurement_usable: true,
                pp_tps: 200.0,
                tg_tps: tg,
                variance: Some(0.01),
                ..Trial::default()
            }
        }
    }

    #[test]
    fn a_probe_that_starts_but_crawls_is_past_the_edge() {
        let mut runner = SpillRunner {
            edge: 31,
            measured: Vec::new(),
        };
        let mut oracle = ScriptedOracle {
            moe_edge_q8: 30,
            offset: 1,
            dense_layers: None,
            calls: 0,
        };
        let mut lines = Vec::new();
        let outcome = run_tuner_with(
            &mut runner,
            &space(),
            &seeds(),
            &ctx(),
            &params(),
            TunerAids {
                oracle: Some(&mut oracle),
                screen: None,
            },
            &mut |line| lines.push(line),
        )
        .expect("a winner");
        assert!(
            lines.iter().any(|l| l.contains("VRAM spilled")),
            "{lines:#?}"
        );
        assert!(lines
            .iter()
            .any(|l| l.contains("runs 1 step(s) past the fitted NCpuMoe=31")));
        assert_eq!(outcome.winner.overrides.get("NCpuMoe"), Some(&json!(30)));
    }

    #[test]
    fn the_screen_only_sees_placements_at_or_above_the_proven_edge() {
        let mut runner = SpillRunner {
            edge: 31,
            measured: Vec::new(),
        };
        let mut oracle = ScriptedOracle {
            moe_edge_q8: 30,
            offset: 1,
            dense_layers: None,
            calls: 0,
        };
        let mut screen = RecordingScreen { asked: Vec::new() };
        run_tuner_with(
            &mut runner,
            &space(),
            &seeds(),
            &ctx(),
            &params(),
            TunerAids {
                oracle: Some(&mut oracle),
                screen: Some(&mut screen),
            },
            &mut |_| {},
        )
        .expect("a winner");
        // The spilled probe (29) may sit in the beam, but its candidates are
        // screened at the proven edge (30), never at 29.
        for batch in &screen.asked {
            for candidate in batch {
                let moe = candidate
                    .get("NCpuMoe")
                    .and_then(serde_json::Value::as_i64)
                    .unwrap();
                assert!(moe >= 30, "screened a spilled placement: {candidate:?}");
            }
        }
        let mut signatures: Vec<String> = screen
            .asked
            .iter()
            .flatten()
            .map(candidate_signature)
            .collect();
        let total = signatures.len();
        signatures.sort();
        signatures.dedup();
        assert_eq!(signatures.len(), total, "a candidate was screened twice");
    }
}
