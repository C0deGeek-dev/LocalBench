//! The tuner's candidate spaces: per-model axis bounds, KV-type pairs, MoE
//! offload values, fine-tune/edge/recovery ladders, and phase expansion.

use serde::{Deserialize, Serialize};

use localbench_scoring::score::Overrides;

use crate::candidate::Candidate;
use crate::overrides::join_overrides;
use crate::seeds::SmartSeeds;

/// The llama.cpp build mode the tuner runs against.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    Native,
    Turboquant,
    Mtpturbo,
    PrismMl,
}

/// The model axes the search space is derived from (catalog-resolved).
#[derive(Debug, Clone, Default)]
pub struct ModelAxes {
    /// Catalog `NCpuMoe` baseline, when set.
    pub n_cpu_moe: Option<i64>,
    /// Configured default when the catalog carries none.
    pub config_n_cpu_moe: Option<i64>,
    /// Catalog `NGpuLayers` baseline, when set.
    pub n_gpu_layers: Option<i64>,
    /// Catalog `MoeExpertLayers` upper bound, when set.
    pub moe_expert_layers: Option<i64>,
    /// Catalog speculative-decoding type, when set (enables MTP drafts).
    pub spec_type: Option<String>,
    /// Catalog `SpecDraftNMax`, when set.
    pub spec_draft_n_max: Option<i64>,
    /// Phases the catalog opts this model out of.
    pub skip_phases: Vec<String>,
}

/// Per-model axis bounds the tuner sweeps relative to.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchSpace {
    pub is_moe: bool,
    pub baseline_n_cpu_moe: i64,
    pub moe_upper: i64,
    pub baseline_ngl: i64,
    pub block_count: i64,
    pub ubatch_candidates: Vec<i64>,
    pub batch_candidates: Vec<i64>,
    pub skip_phases: Vec<String>,
    pub mtp_draft_candidates: Vec<i64>,
}

/// Derive per-model axis bounds. Pure (no I/O).
///
/// `expert_count` is the model's GGUF `expert_count`: `>= 0` means "known from
/// the file" (0 = dense, > 0 = MoE) and is authoritative; `-1` means "unknown"
/// and falls back to the catalog heuristic. A dense model must be tuned on the
/// `--n-gpu-layers` axis, never `--n-cpu-moe` (a no-op without expert tensors —
/// every value loads the full model and OOMs identically).
#[must_use]
pub fn resolve_search_space(axes: &ModelAxes, expert_count: i64, block_count: i64) -> SearchSpace {
    let baseline_n_cpu_moe = axes
        .n_cpu_moe
        .or(axes.config_n_cpu_moe)
        .unwrap_or(35)
        .max(0);
    let baseline_ngl = axes.n_gpu_layers.unwrap_or(999);
    let is_moe = if expert_count >= 0 {
        expert_count > 0
    } else {
        baseline_n_cpu_moe > 0
    };
    // Generous cap above the baseline (covers MoE coder variants whose top
    // expert-layer count tops out around 60 today) unless the catalog says.
    let moe_upper = axes
        .moe_expert_layers
        .unwrap_or_else(|| 60.max(baseline_n_cpu_moe + 20));

    SearchSpace {
        is_moe,
        baseline_n_cpu_moe,
        moe_upper,
        baseline_ngl,
        block_count,
        ubatch_candidates: vec![256, 512, 1024],
        batch_candidates: vec![512, 1024, 2048],
        skip_phases: axes.skip_phases.clone(),
        mtp_draft_candidates: match &axes.spec_type {
            Some(spec) if !spec.trim().is_empty() => {
                let mut drafts = vec![axes.spec_draft_n_max.unwrap_or(0), 2, 3];
                drafts.retain(|d| *d > 0);
                dedup_preserving_order(&drafts)
            }
            _ => Vec::new(),
        },
    }
}

/// A K/V cache-type pair.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvPair {
    pub k: String,
    pub v: String,
}

/// Resolve the KV cache types the search may use: an explicit allowlist wins;
/// otherwise the baseline K type (native/PrismML), plus the baseline pair and
/// the turbo types on turbo-capable builds.
#[must_use]
pub fn resolve_allowed_kv_types(allowed: &[String], baseline: &KvPair, mode: Mode) -> Vec<String> {
    let mut values: Vec<String> = Vec::new();
    if !allowed.is_empty() {
        values.extend(allowed.iter().filter(|s| !s.trim().is_empty()).cloned());
    } else if matches!(mode, Mode::Native | Mode::PrismMl) {
        values.push(baseline.k.clone());
    }
    if matches!(mode, Mode::Turboquant | Mode::Mtpturbo) {
        values.push(baseline.k.clone());
        values.push(baseline.v.clone());
        values.push("turbo3".to_string());
        values.push("turbo4".to_string());
    }
    let values: Vec<String> = values
        .into_iter()
        .filter(|s| !s.trim().is_empty())
        .collect();
    dedup_preserving_order(&values)
}

/// The KV pairs the search sweeps. Default: identical K/V per allowed type,
/// plus the turbo3/turbo4 crosses when both are allowed. `aggressive` sweeps
/// every non-identical pair; `full_matrix` sweeps everything.
#[must_use]
pub fn kv_candidate_pairs(allowed: &[String], aggressive: bool, full_matrix: bool) -> Vec<KvPair> {
    let mut pairs = Vec::new();
    if full_matrix {
        for k in allowed {
            for v in allowed {
                pairs.push(KvPair {
                    k: k.clone(),
                    v: v.clone(),
                });
            }
        }
    } else if !aggressive {
        for t in allowed {
            pairs.push(KvPair {
                k: t.clone(),
                v: t.clone(),
            });
        }
        let turbo3 = allowed.iter().find(|t| t.eq_ignore_ascii_case("turbo3"));
        let turbo4 = allowed.iter().find(|t| t.eq_ignore_ascii_case("turbo4"));
        if let (Some(t3), Some(t4)) = (turbo3, turbo4) {
            pairs.push(KvPair {
                k: t3.clone(),
                v: t4.clone(),
            });
            pairs.push(KvPair {
                k: t4.clone(),
                v: t3.clone(),
            });
        }
    } else {
        for k in allowed {
            for v in allowed {
                if k != v {
                    pairs.push(KvPair {
                        k: k.clone(),
                        v: v.clone(),
                    });
                }
            }
        }
    }
    pairs
}

/// The SWA/cache flag overlays swept as a phase.
#[must_use]
pub fn swa_flag_overlays() -> Vec<Overrides> {
    use crate::overrides::overrides_of;
    vec![
        overrides_of(&[("SwaFull", false.into()), ("CachePrompt", false.into())]),
        overrides_of(&[("SwaFull", true.into()), ("CachePrompt", false.into())]),
        overrides_of(&[
            ("SwaFull", false.into()),
            ("CachePrompt", true.into()),
            ("CacheReuse", 256.into()),
        ]),
        overrides_of(&[
            ("SwaFull", true.into()),
            ("CachePrompt", true.into()),
            ("CacheReuse", 256.into()),
        ]),
    ]
}

/// Clamp the trial budget to `[1, 100]`.
#[must_use]
pub fn resolve_tuner_budget(budget: i64) -> i64 {
    budget.clamp(1, 100)
}

/// The MoE-phase NCpuMoe values, in sweep order: the baseline (unless it
/// failed), the smart-seed offload candidates, stride-5 descent to the
/// estimated minimum, dense descent below it (baseline healthy only), and
/// stride-5 ascent to the upper bound. A failed baseline restricts the sweep
/// to strictly-higher offload (lower values are already known-unviable).
#[must_use]
pub fn moe_candidate_values(
    space: &SearchSpace,
    seeds: &SmartSeeds,
    baseline_failed: bool,
    estimated_minimum: i64,
) -> Vec<i64> {
    let base = space.baseline_n_cpu_moe;
    let minimum = estimated_minimum.clamp(0, space.moe_upper);
    let mut values: Vec<i64> = Vec::new();
    if !baseline_failed {
        values.push(base);
    }
    values.extend(seeds.offload_candidates.iter().copied());
    let mut n = (base - 5).max(minimum);
    while n >= minimum {
        values.push(n);
        n -= 5;
    }
    if !baseline_failed && minimum > 0 {
        values.push(minimum);
        for n in (0..minimum).rev() {
            values.push(n);
        }
    }
    let mut n = base + 5;
    while n <= space.moe_upper {
        values.push(n);
        n += 5;
    }
    if baseline_failed {
        values.retain(|v| *v > base);
    }
    dedup_preserving_order(&values)
}

/// Symmetric ±1..=5 fine-tune grid around the current NCpuMoe, clamped to
/// `[0, upper]`.
#[must_use]
pub fn fine_tune_n_cpu_moe_candidates(current: i64, upper: i64) -> Vec<i64> {
    let mut values = vec![current];
    for delta in 1..=5 {
        values.push(current - delta);
        values.push(current + delta);
    }
    values.retain(|v| *v >= 0 && *v <= upper);
    dedup_preserving_order(&values)
}

/// Stride-1 NCpuMoe probes for the viability edges, in execution order:
/// 1. **Cliff descent** — from just below the lowest stable value downward
///    (more GPU residency), top-down so monotonic-OOM pruning stops the
///    waterfall at the first OOM, pinning the bare-minimum-viable offload.
/// 2. **Best neighborhood** — ±radius around the top-scoring stable value, to
///    catch a peak sitting a step or two off the cliff.
///
/// Values already measured stably are dropped.
#[must_use]
pub fn moe_edge_refine_values(
    measured_stable: &[i64],
    best_moe: i64,
    radius: i64,
    max_descend: i64,
    lower: i64,
    upper: i64,
) -> Vec<i64> {
    let upper = if upper <= 0 { i64::MAX } else { upper };
    let lower = lower.max(0);
    let stable: Vec<i64> = measured_stable
        .iter()
        .copied()
        .filter(|v| *v >= 0)
        .collect();
    if stable.is_empty() && best_moe < 0 {
        return Vec::new();
    }

    let min_stable = stable.iter().copied().min().unwrap_or(best_moe);
    let best = if best_moe < 0 { min_stable } else { best_moe };

    let mut raw: Vec<i64> = Vec::new();
    let floor = lower.max(min_stable - max_descend.max(1));
    let mut n = min_stable - 1;
    while n >= floor {
        raw.push(n);
        n -= 1;
    }
    for d in 1..=radius.max(1) {
        raw.push(best - d);
        raw.push(best + d);
    }

    let mut result = Vec::new();
    for n in raw {
        if n < lower || n > upper || stable.contains(&n) || result.contains(&n) {
            continue;
        }
        result.push(n);
    }
    result
}

/// The climb-higher recovery ladder for a seed that does not even start:
/// exponential jumps from just above the failing value toward the baseline
/// (monotonic OOM pruning skips known-failing dense values), a dense bisection
/// fill between the first two jumps, then a dense tail up to the upper bound.
#[must_use]
pub fn recovery_n_cpu_moe_candidates(
    current: i64,
    baseline: i64,
    upper: i64,
    minimum: i64,
) -> Vec<i64> {
    let current = current.clamp(0, upper);
    let baseline = baseline.clamp(0, upper);
    let minimum = minimum.clamp(0, upper);

    let lower = minimum.max(current + 1);
    if lower > upper {
        return Vec::new();
    }

    let target = upper.min(baseline.max(lower));
    let first_jump = if current > 0 {
        lower.max(current * 2)
    } else {
        lower
    };
    let mut probe = target.min(first_jump);

    let mut values: Vec<i64> = Vec::new();
    loop {
        values.push(probe);
        if probe >= target {
            break;
        }
        let next = target.min((probe + 1).max(probe * 2));
        if next <= probe {
            break;
        }
        probe = next;
    }

    let jumps = dedup_preserving_order(&values);
    let high_jump;
    if jumps.len() >= 2 {
        let low_jump = jumps[0];
        high_jump = jumps[1];
        let mid = div_ceil(low_jump + high_jump, 2);
        let mut n = mid;
        while n > low_jump {
            values.push(n);
            n -= 1;
        }
        let mut n = mid + 1;
        while n < high_jump {
            values.push(n);
            n += 1;
        }
    } else {
        high_jump = jumps[0];
        let mid = div_ceil(current + high_jump, 2);
        let mut n = mid;
        while n > current {
            values.push(n);
            n -= 1;
        }
    }

    let mut n = high_jump + 1;
    while n <= upper {
        values.push(n);
        n += 1;
    }

    values.retain(|v| *v >= minimum && *v <= upper);
    dedup_preserving_order(&values)
}

/// A baseline/fine-tune seed has "failed" when it never produced a usable
/// trial: it was skipped before it ran (no candidate), or it ran but did not
/// start up / OOMed. Both must steer the MoE phase into the climb-higher
/// recovery ladder rather than the symmetric fine-tune grid (whose lower half
/// is already known-unviable).
#[must_use]
pub fn seed_failed(candidate: Option<&Candidate>) -> bool {
    let Some(candidate) = candidate else {
        return true;
    };
    let Some(trial) = &candidate.trial else {
        return true;
    };
    !trial.startup_ok || trial.oom
}

/// The overrides the recovery phase overlays onto. When the baseline candidate
/// is `None` (skipped before it ran), the baseline's own overrides are the
/// seed — recovery must always have a base to overlay onto, or the phase would
/// build zero candidates even though viable higher-offload configs exist.
#[must_use]
pub fn baseline_recovery_seed(
    candidate: Option<&Candidate>,
    baseline_overrides: &Overrides,
) -> Overrides {
    candidate.map_or_else(|| baseline_overrides.clone(), |c| c.overrides.clone())
}

/// A dense model that OOMs at the baseline has two recovery levers, in
/// preference order:
/// 1. **Shrink the KV cache** (turbo pairs at full GPU offload) — the lever
///    for a context-bound OOM; it keeps every layer on the GPU, so it is both
///    the most likely to fit and the fastest.
/// 2. **Offload layers** (lower `--n-gpu-layers`), halved from the *real*
///    layer count — a value at/above the layer count is a no-op and must never
///    anchor the sweep.
#[must_use]
pub fn dense_recovery_candidates(
    baseline_kv: &KvPair,
    kv_pairs: &[KvPair],
    baseline_ngl: i64,
    layer_count: i64,
    max_ngl_candidates: usize,
) -> Vec<Overrides> {
    use crate::overrides::overrides_of;

    let mut candidates: Vec<Overrides> = Vec::new();
    let mut seen: Vec<String> = vec![format!("{}|{}", baseline_kv.k, baseline_kv.v)];

    for pair in kv_pairs {
        let is_turbo = pair.k.starts_with("turbo") || pair.v.starts_with("turbo");
        if !is_turbo {
            continue;
        }
        let key = format!("{}|{}", pair.k, pair.v);
        if seen.contains(&key) {
            continue;
        }
        seen.push(key);
        candidates.push(overrides_of(&[
            ("KvK", pair.k.clone().into()),
            ("KvV", pair.v.clone().into()),
        ]));
    }

    let top = if layer_count > 0 {
        layer_count
    } else if baseline_ngl > 0 && baseline_ngl < 999 {
        baseline_ngl
    } else {
        0
    };
    let mut ngl_values: Vec<i64> = Vec::new();
    if top > 0 {
        let mut cur = top;
        while cur >= 1 && ngl_values.len() < max_ngl_candidates {
            ngl_values.push(cur);
            cur /= 2;
        }
    } else {
        // Layer count unknown: legacy halving from the sentinel.
        let start = if baseline_ngl > 0 { baseline_ngl } else { 999 };
        let mut cur = start / 2;
        while cur > 0 && ngl_values.len() < max_ngl_candidates {
            ngl_values.push(cur);
            cur /= 2;
        }
    }
    for ngl in dedup_preserving_order(&ngl_values) {
        candidates.push(overrides_of(&[
            ("KvK", baseline_kv.k.clone().into()),
            ("KvV", baseline_kv.v.clone().into()),
            ("NGpuLayers", ngl.into()),
        ]));
    }

    candidates
}

/// The minimum NCpuMoe the MTP regime starts from. MTP's draft head needs
/// VRAM headroom: when the main GGUF is already near/above device VRAM (or the
/// card is 24GB or smaller), start near the known offload boundary instead of
/// testing GPU-only MoE or tiny CPU-offload values. Zero (no floor) off the
/// mtpturbo mode or for dense models.
#[must_use]
pub fn mtp_minimum_n_cpu_moe(
    space: &SearchSpace,
    seeds: &SmartSeeds,
    mode: Mode,
    mtp_enabled: bool,
) -> i64 {
    if !mtp_enabled || !space.is_moe || mode != Mode::Mtpturbo {
        return 0;
    }
    let base = space.baseline_n_cpu_moe;
    // The floor applies when the main GGUF is within 2GB of device VRAM (the
    // draft head will not fit above the boundary) or the card is 24GB or less.
    let gguf_near_vram =
        seeds.gguf_size_gb > 0.0 && seeds.gguf_size_gb >= (f64::from(seeds.vram_gb) - 2.0);
    let floor = if seeds.vram_gb > 0 && (gguf_near_vram || seeds.vram_gb <= 24) {
        (base - 30).max(5)
    } else {
        0
    };
    floor.clamp(0, space.moe_upper)
}

/// A planned coverage worklist: seeds × MoE values × KV pairs, truncated to
/// the remaining budget, with the planned/skipped counts reported so silent
/// truncation never reads as full coverage.
#[derive(Debug, Clone, PartialEq)]
pub struct CoverageWorklist {
    pub scheduled: Vec<Overrides>,
    pub planned_count: usize,
    pub skipped_count: usize,
}

/// Build the turbo-mode MoE coverage worklist.
#[must_use]
pub fn moe_coverage_worklist(
    seeds: &[Overrides],
    kv_pairs: &[KvPair],
    moe_values: &[i64],
    budget_remaining: usize,
) -> CoverageWorklist {
    use crate::overrides::overrides_of;

    let mut planned: Vec<Overrides> = Vec::new();
    for seed in seeds {
        for n in moe_values {
            for pair in kv_pairs {
                planned.push(join_overrides(
                    seed,
                    &overrides_of(&[
                        ("KvK", pair.k.clone().into()),
                        ("KvV", pair.v.clone().into()),
                        ("NCpuMoe", (*n).into()),
                    ]),
                ));
            }
        }
    }
    let planned_count = planned.len();
    let scheduled: Vec<Overrides> = planned.into_iter().take(budget_remaining).collect();
    CoverageWorklist {
        skipped_count: planned_count - scheduled.len(),
        planned_count,
        scheduled,
    }
}

/// Expand a beam by overlaying each overlay onto each beam candidate's config.
#[must_use]
pub fn expand_phase_candidates(beam: &[Candidate], overlays: &[Overrides]) -> Vec<Overrides> {
    let mut expanded = Vec::new();
    for candidate in beam {
        for overlay in overlays {
            expanded.push(join_overrides(&candidate.overrides, overlay));
        }
    }
    expanded
}

/// First-occurrence-preserving dedup (the PS `Select-Object -Unique` shape).
fn dedup_preserving_order<T: Clone + PartialEq>(values: &[T]) -> Vec<T> {
    let mut out: Vec<T> = Vec::new();
    for value in values {
        if !out.contains(value) {
            out.push(value.clone());
        }
    }
    out
}

/// Ceiling division for non-negative sums.
fn div_ceil(sum: i64, by: i64) -> i64 {
    (sum + by - 1) / by
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::seeds::SmartSeeds;

    fn kv(k: &str, v: &str) -> KvPair {
        KvPair {
            k: k.to_string(),
            v: v.to_string(),
        }
    }

    fn seeds_with(offload: Vec<i64>, vram_gb: u32, gguf_size_gb: f64) -> SmartSeeds {
        SmartSeeds {
            offload_candidates: offload,
            vram_gb,
            gguf_size_gb,
            ..SmartSeeds::default()
        }
    }

    #[test]
    fn gguf_expert_count_is_authoritative_for_moe_detection() {
        let axes = ModelAxes {
            n_cpu_moe: Some(35),
            ..ModelAxes::default()
        };
        // expert_count 0 = dense, even with a MoE-looking catalog baseline.
        assert!(!resolve_search_space(&axes, 0, -1).is_moe);
        assert!(resolve_search_space(&axes, 128, -1).is_moe);
        // Unknown expert count falls back to the catalog heuristic.
        assert!(resolve_search_space(&axes, -1, -1).is_moe);
        let dense_axes = ModelAxes::default();
        // No catalog NCpuMoe → config default 35 → heuristic says MoE.
        assert!(resolve_search_space(&dense_axes, -1, -1).is_moe);
    }

    #[test]
    fn moe_upper_is_catalog_or_generous_cap() {
        let axes = ModelAxes {
            n_cpu_moe: Some(50),
            ..ModelAxes::default()
        };
        assert_eq!(resolve_search_space(&axes, -1, -1).moe_upper, 70);
        let low = ModelAxes {
            n_cpu_moe: Some(10),
            ..ModelAxes::default()
        };
        assert_eq!(resolve_search_space(&low, -1, -1).moe_upper, 60);
        let explicit = ModelAxes {
            n_cpu_moe: Some(10),
            moe_expert_layers: Some(48),
            ..ModelAxes::default()
        };
        assert_eq!(resolve_search_space(&explicit, -1, -1).moe_upper, 48);
    }

    #[test]
    fn mtp_drafts_require_a_spec_type() {
        let none = resolve_search_space(&ModelAxes::default(), -1, -1);
        assert!(none.mtp_draft_candidates.is_empty());
        let with_spec = ModelAxes {
            spec_type: Some("mtp".to_string()),
            spec_draft_n_max: Some(4),
            ..ModelAxes::default()
        };
        assert_eq!(
            resolve_search_space(&with_spec, -1, -1).mtp_draft_candidates,
            vec![4, 2, 3]
        );
    }

    #[test]
    fn allowed_kv_types_by_mode() {
        let baseline = kv("q8_0", "q8_0");
        assert_eq!(
            resolve_allowed_kv_types(&[], &baseline, Mode::Native),
            vec!["q8_0"]
        );
        assert_eq!(
            resolve_allowed_kv_types(&[], &baseline, Mode::PrismMl),
            vec!["q8_0"]
        );
        assert_eq!(
            resolve_allowed_kv_types(&[], &baseline, Mode::Turboquant),
            vec!["q8_0", "turbo3", "turbo4"]
        );
        // An explicit allowlist wins, turbo types still appended on turbo modes.
        assert_eq!(
            resolve_allowed_kv_types(&["f16".to_string()], &baseline, Mode::Mtpturbo),
            vec!["f16", "q8_0", "turbo3", "turbo4"]
        );
        assert_eq!(
            resolve_allowed_kv_types(&["f16".to_string()], &baseline, Mode::Native),
            vec!["f16"]
        );
    }

    #[test]
    fn kv_pairs_default_identity_plus_turbo_crosses() {
        let allowed = vec![
            "q8_0".to_string(),
            "turbo3".to_string(),
            "turbo4".to_string(),
        ];
        let pairs = kv_candidate_pairs(&allowed, false, false);
        assert!(pairs.contains(&kv("q8_0", "q8_0")));
        assert!(pairs.contains(&kv("turbo3", "turbo3")));
        assert!(pairs.contains(&kv("turbo3", "turbo4")));
        assert!(pairs.contains(&kv("turbo4", "turbo3")));
        assert_eq!(pairs.len(), 5);

        // Aggressive: every non-identical pair.
        let aggressive = kv_candidate_pairs(&allowed, true, false);
        assert_eq!(aggressive.len(), 6);
        assert!(!aggressive.contains(&kv("q8_0", "q8_0")));

        // Full matrix: everything.
        assert_eq!(kv_candidate_pairs(&allowed, false, true).len(), 9);
    }

    #[test]
    fn budget_clamps_to_1_100() {
        assert_eq!(resolve_tuner_budget(0), 1);
        assert_eq!(resolve_tuner_budget(30), 30);
        assert_eq!(resolve_tuner_budget(500), 100);
    }

    #[test]
    fn moe_values_sweep_baseline_seeds_descent_and_ascent() {
        let axes = ModelAxes {
            n_cpu_moe: Some(20),
            ..ModelAxes::default()
        };
        let space = resolve_search_space(&axes, 128, -1);
        let values = moe_candidate_values(&space, &seeds_with(vec![20, 15, 25], 0, 0.0), false, 10);
        // Baseline first, then seeds, stride-5 descent to the minimum, dense
        // below it, stride-5 ascent.
        assert_eq!(values[0], 20);
        assert!(values.contains(&15));
        assert!(values.contains(&10), "stride-5 descent reaches the minimum");
        assert!(
            values.contains(&9) && values.contains(&0),
            "dense below the minimum"
        );
        assert!(
            values.contains(&25) && values.contains(&60),
            "ascent to upper"
        );
    }

    #[test]
    fn a_failed_baseline_restricts_the_sweep_to_higher_offload() {
        let axes = ModelAxes {
            n_cpu_moe: Some(20),
            ..ModelAxes::default()
        };
        let space = resolve_search_space(&axes, 128, -1);
        let values = moe_candidate_values(&space, &seeds_with(vec![15, 25], 0, 0.0), true, 0);
        assert!(
            values.iter().all(|v| *v > 20),
            "lower offload is known-unviable: {values:?}"
        );
        assert!(values.contains(&25));
    }

    #[test]
    fn fine_tune_grid_is_symmetric_and_clamped() {
        assert_eq!(
            fine_tune_n_cpu_moe_candidates(3, 60),
            vec![3, 2, 4, 1, 5, 0, 6, 7, 8]
        );
        let hi = fine_tune_n_cpu_moe_candidates(59, 60);
        assert!(hi.contains(&60) && !hi.contains(&61));
    }

    #[test]
    fn edge_refine_descends_the_cliff_top_down_then_rings_the_best() {
        let values = moe_edge_refine_values(&[12, 15, 20], 15, 2, 8, 0, 60);
        // Cliff descent from 11 downward first (monotonic-OOM pruning stops it).
        assert_eq!(&values[..3], &[11, 10, 9]);
        // Then the ±2 ring around the best (15), minus already-stable values.
        assert!(values.contains(&14) && values.contains(&16) && values.contains(&13));
        assert!(!values.contains(&15) && !values.contains(&12));
        assert!(moe_edge_refine_values(&[], -1, 2, 8, 0, 60).is_empty());
    }

    #[test]
    fn recovery_ladder_jumps_then_bisects_then_fills_the_tail() {
        // current=5 (failing), baseline=40, upper=60: first jump 10, then 20, 40.
        let values = recovery_n_cpu_moe_candidates(5, 40, 60, 0);
        assert_eq!(values[0], 10, "first jump doubles the failing value");
        assert!(values.contains(&20) && values.contains(&40));
        // Bisection fill lands between the first two jumps.
        assert!(values.contains(&15) && values.contains(&12));
        // Dense tail above the second jump up to upper.
        assert!(values.contains(&21) && values.contains(&60));
        // Everything strictly above the failing value.
        assert!(values.iter().all(|v| *v > 5));
        // Exhausted range: nothing to climb to.
        assert!(recovery_n_cpu_moe_candidates(60, 40, 60, 0).is_empty());
    }

    #[test]
    fn seed_failed_covers_skipped_and_unhealthy() {
        use crate::candidate::{new_candidate, ScoreProfile, ScoringContext};
        use localbench_scoring::score::{Optimize, Trial};

        assert!(seed_failed(None));
        let ov = crate::overrides::overrides_of(&[("NCpuMoe", 10.into())]);
        let unmeasured = new_candidate(
            &ov,
            None,
            ScoreProfile::Pure,
            "seed",
            Optimize::CodingAgent,
            &ScoringContext::default(),
        );
        assert!(seed_failed(Some(&unmeasured)));
        let oom = Trial {
            startup_ok: true,
            oom: true,
            ..Trial::default()
        };
        let failed = new_candidate(
            &ov,
            Some(&oom),
            ScoreProfile::Pure,
            "seed",
            Optimize::CodingAgent,
            &ScoringContext::default(),
        );
        assert!(seed_failed(Some(&failed)));
        let healthy = Trial {
            startup_ok: true,
            oom: false,
            measurement_usable: true,
            pp_tps: 100.0,
            tg_tps: 10.0,
            ..Trial::default()
        };
        let ok = new_candidate(
            &ov,
            Some(&healthy),
            ScoreProfile::Pure,
            "seed",
            Optimize::CodingAgent,
            &ScoringContext::default(),
        );
        assert!(!seed_failed(Some(&ok)));
    }

    #[test]
    fn recovery_seed_synthesizes_from_baseline_overrides_when_skipped() {
        let baseline = crate::overrides::overrides_of(&[("NCpuMoe", 35.into())]);
        let seed = baseline_recovery_seed(None, &baseline);
        assert_eq!(seed, baseline);
    }

    #[test]
    fn dense_recovery_prefers_kv_compression_then_anchored_ngl_halving() {
        let baseline = kv("q8_0", "q8_0");
        let pairs = vec![kv("q8_0", "q8_0"), kv("turbo3", "turbo3"), kv("f16", "f16")];
        let candidates = dense_recovery_candidates(&baseline, &pairs, 999, 48, 6);
        // Turbo pair first (context-bound lever), non-turbo pairs skipped.
        assert_eq!(candidates[0]["KvK"], serde_json::json!("turbo3"));
        assert!(!candidates[0].contains_key("NGpuLayers"));
        // Then NGL halving anchored at the REAL layer count: 48, 24, 12, 6, 3, 1.
        let ngls: Vec<i64> = candidates
            .iter()
            .filter_map(|c| c.get("NGpuLayers").and_then(serde_json::Value::as_i64))
            .collect();
        assert_eq!(ngls, vec![48, 24, 12, 6, 3, 1]);
        // Unknown layer count: legacy halving from the sentinel never emits 999.
        let legacy = dense_recovery_candidates(&baseline, &[], 999, -1, 6);
        let ngls: Vec<i64> = legacy
            .iter()
            .filter_map(|c| c.get("NGpuLayers").and_then(serde_json::Value::as_i64))
            .collect();
        assert_eq!(ngls[0], 499);
        assert!(!ngls.contains(&999));
    }

    #[test]
    fn mtp_minimum_needs_mtpturbo_moe_and_a_tight_card() {
        let axes = ModelAxes {
            n_cpu_moe: Some(40),
            ..ModelAxes::default()
        };
        let space = resolve_search_space(&axes, 128, -1);
        let tight = seeds_with(vec![], 24, 23.0);
        assert_eq!(
            mtp_minimum_n_cpu_moe(&space, &tight, Mode::Mtpturbo, true),
            10
        );
        // GGUF near VRAM also floors it on a big card.
        let big_card_big_model = seeds_with(vec![], 48, 47.0);
        assert_eq!(
            mtp_minimum_n_cpu_moe(&space, &big_card_big_model, Mode::Mtpturbo, true),
            10
        );
        // Roomy card, small model: no floor.
        let roomy = seeds_with(vec![], 48, 20.0);
        assert_eq!(
            mtp_minimum_n_cpu_moe(&space, &roomy, Mode::Mtpturbo, true),
            0
        );
        // Off outside mtpturbo / MTP-off / dense.
        assert_eq!(mtp_minimum_n_cpu_moe(&space, &tight, Mode::Native, true), 0);
        assert_eq!(
            mtp_minimum_n_cpu_moe(&space, &tight, Mode::Mtpturbo, false),
            0
        );
    }

    #[test]
    fn coverage_worklist_reports_planned_and_skipped() {
        let seeds = vec![crate::overrides::overrides_of(&[("Threads", 8.into())])];
        let pairs = vec![kv("turbo3", "turbo3"), kv("turbo4", "turbo4")];
        let list = moe_coverage_worklist(&seeds, &pairs, &[10, 15, 20], 4);
        assert_eq!(list.planned_count, 6);
        assert_eq!(list.scheduled.len(), 4);
        assert_eq!(list.skipped_count, 2);
        assert_eq!(list.scheduled[0]["Threads"], serde_json::json!(8));
        assert_eq!(list.scheduled[0]["NCpuMoe"], serde_json::json!(10));
        // Zero budget schedules nothing but still reports the plan.
        let none = moe_coverage_worklist(&seeds, &pairs, &[10], 0);
        assert!(none.scheduled.is_empty());
        assert_eq!(none.planned_count, 2);
    }

    #[test]
    fn swa_overlays_cover_the_flag_grid() {
        let overlays = swa_flag_overlays();
        assert_eq!(overlays.len(), 4);
        assert!(overlays
            .iter()
            .any(|o| o["SwaFull"] == serde_json::json!(true)
                && o["CachePrompt"] == serde_json::json!(true)
                && o["CacheReuse"] == serde_json::json!(256)));
    }
}
