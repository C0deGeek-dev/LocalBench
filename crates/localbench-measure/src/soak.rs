//! Context-soak validation and the stress-recovery override ladder.
//!
//! The soak proves a winner survives its configured long-context workload
//! before it is saved. Evaluation here is pure: the caller measures each
//! target and records what happened; this module renders the verdict — any
//! startup failure/OOM, any breach of the free-VRAM floor, or an exhausted
//! budget fails the validation.

use serde::{Deserialize, Serialize};

use localbench_scoring::score::Overrides;
use localbench_scoring::stats::round_dp;
use localbench_search::overrides::candidate_signature;

/// What one soak target measured.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SoakTargetOutcome {
    pub target_prompt_tokens: u32,
    pub startup_ok: bool,
    pub oom: bool,
    /// The classified error type, empty when none.
    pub error_type: String,
    /// The raw error detail, empty when none.
    pub error: String,
    pub pp_tps: f64,
    pub tg_tps: f64,
    /// Minimum free VRAM observed during the target, when telemetry caught it.
    pub gpu_vram_free_gb_min: Option<f64>,
}

/// The soak verdict.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SoakValidation {
    pub enabled: bool,
    pub passed: bool,
    /// `passed` / `failed`.
    pub status: String,
    pub targets: Vec<SoakTargetOutcome>,
    pub min_free_vram_gb_required: f64,
    pub min_free_vram_gb_observed: Option<f64>,
    pub failures: Vec<String>,
}

/// Evaluate a soak from its measured per-target outcomes. `budget_exhausted_at`
/// names a target that could not run because the trial budget ran out — that is
/// a failure (an unproven soak must not read as a pass).
#[must_use]
pub fn evaluate_soak(
    outcomes: &[SoakTargetOutcome],
    min_free_vram_gb: f64,
    budget_exhausted_at: Option<u32>,
    phase_prefix: &str,
) -> SoakValidation {
    let mut failures: Vec<String> = Vec::new();
    let mut observed: Option<f64> = None;

    for outcome in outcomes {
        let target = outcome.target_prompt_tokens;
        if let Some(free) = outcome.gpu_vram_free_gb_min {
            let rounded = round_dp(free, 2);
            observed = Some(observed.map_or(rounded, |o: f64| o.min(rounded)));
        }
        if !outcome.startup_ok || outcome.oom {
            failures.push(format!(
                "{phase_prefix}_{target} failed: {} {}",
                outcome.error_type, outcome.error
            ));
        } else if let Some(free) = outcome.gpu_vram_free_gb_min {
            if free < min_free_vram_gb {
                failures.push(format!(
                    "{phase_prefix}_{target} left only {free:.2} GB free VRAM; \
                     required >= {min_free_vram_gb:.2} GB"
                ));
            }
        }
    }
    if let Some(target) = budget_exhausted_at {
        failures.push(format!(
            "{phase_prefix} budget exhausted before target {target}."
        ));
    }

    let passed = failures.is_empty();
    SoakValidation {
        enabled: true,
        passed,
        status: if passed { "passed" } else { "failed" }.to_string(),
        targets: outcomes.to_vec(),
        min_free_vram_gb_required: round_dp(min_free_vram_gb, 2),
        min_free_vram_gb_observed: observed,
        failures,
    }
}

/// The recovery configs to try when a winner fails its soak, most-preferred
/// first:
/// 1. The **smallest CPU-offload bumps** (`NCpuMoe +1, +2, ...`) so recovery
///    lands on the most GPU-resident (fastest) config that still survives,
///    instead of jumping straight to a slow deep offload.
/// 2. A batch-safe variant (ubatch/batch 512).
/// 3. A draft-safe variant (`SpecDraftNMax` 1) when speculative decoding is on.
/// 4. MTP dropped entirely (and its batch-safe variant).
/// 5. Each offload bump combined with the batch/draft-safe settings.
///
/// Duplicates (by config signature) are emitted once.
#[must_use]
pub fn stress_recovery_overrides(
    base: &Overrides,
    moe_upper: i64,
    moe_step_count: i64,
) -> Vec<Overrides> {
    let mut items: Vec<Overrides> = Vec::new();
    let mut seen: Vec<String> = Vec::new();
    let add = |candidate: Overrides, items: &mut Vec<Overrides>, seen: &mut Vec<String>| {
        let sig = candidate_signature(&candidate);
        if !seen.contains(&sig) {
            seen.push(sig);
            items.push(candidate);
        }
    };

    let current_moe = base.get("NCpuMoe").and_then(serde_json::Value::as_i64);
    let mut moe_values: Vec<i64> = Vec::new();
    if let Some(current) = current_moe {
        for delta in 1..=moe_step_count.max(1) {
            let next = current + delta;
            if moe_upper <= 0 || next <= moe_upper {
                moe_values.push(next);
                let mut cand = base.clone();
                cand.insert("NCpuMoe".to_string(), next.into());
                add(cand, &mut items, &mut seen);
            }
        }
    }

    let mut batch_safe = base.clone();
    batch_safe.insert("UbatchSize".to_string(), 512.into());
    batch_safe.insert("BatchSize".to_string(), 512.into());
    add(batch_safe, &mut items, &mut seen);

    let draft = base
        .get("SpecDraftNMax")
        .and_then(serde_json::Value::as_i64);
    if draft.is_some_and(|d| d > 1) {
        let mut draft_safe = base.clone();
        draft_safe.insert("SpecDraftNMax".to_string(), 1.into());
        add(draft_safe, &mut items, &mut seen);
    }

    if base.contains_key("SpecType") {
        let mut no_mtp = base.clone();
        no_mtp.remove("SpecType");
        no_mtp.remove("SpecDraftNMax");
        let mut no_mtp_batch = no_mtp.clone();
        add(no_mtp, &mut items, &mut seen);
        no_mtp_batch.insert("UbatchSize".to_string(), 512.into());
        no_mtp_batch.insert("BatchSize".to_string(), 512.into());
        add(no_mtp_batch, &mut items, &mut seen);
    }

    for next in moe_values {
        let mut combo = base.clone();
        combo.insert("NCpuMoe".to_string(), next.into());
        combo.insert("UbatchSize".to_string(), 512.into());
        combo.insert("BatchSize".to_string(), 512.into());
        if combo
            .get("SpecDraftNMax")
            .and_then(serde_json::Value::as_i64)
            .is_some_and(|d| d > 1)
        {
            combo.insert("SpecDraftNMax".to_string(), 1.into());
        }
        add(combo, &mut items, &mut seen);
    }

    items
}

/// Merge telemetry maps: later sources win per key (the stress snapshot
/// overlays the base).
#[must_use]
pub fn join_stress_telemetry(
    base: &serde_json::Map<String, serde_json::Value>,
    stress: &serde_json::Map<String, serde_json::Value>,
) -> serde_json::Map<String, serde_json::Value> {
    let mut merged = base.clone();
    for (key, value) in stress {
        merged.insert(key.clone(), value.clone());
    }
    merged
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use localbench_search::overrides::overrides_of;

    fn ok_target(target: u32, free: f64) -> SoakTargetOutcome {
        SoakTargetOutcome {
            target_prompt_tokens: target,
            startup_ok: true,
            oom: false,
            error_type: String::new(),
            error: String::new(),
            pp_tps: 500.0,
            tg_tps: 40.0,
            gpu_vram_free_gb_min: Some(free),
        }
    }

    #[test]
    fn a_clean_soak_passes_and_records_the_worst_free_vram() {
        let v = evaluate_soak(
            &[ok_target(16_384, 1.4), ok_target(32_768, 0.9)],
            0.5,
            None,
            "context_soak",
        );
        assert!(v.passed);
        assert_eq!(v.status, "passed");
        assert_eq!(v.min_free_vram_gb_observed, Some(0.9));
        assert!(v.failures.is_empty());
    }

    #[test]
    fn an_oom_target_fails_the_soak() {
        let mut bad = ok_target(32_768, 0.2);
        bad.oom = true;
        bad.error_type = "oom".to_string();
        bad.error = "CUDA error: out of memory".to_string();
        let v = evaluate_soak(&[ok_target(16_384, 1.0), bad], 0.5, None, "context_soak");
        assert!(!v.passed);
        assert!(v.failures[0].contains("context_soak_32768 failed"));
        assert!(v.failures[0].contains("oom"));
    }

    #[test]
    fn breaching_the_free_vram_floor_fails_even_when_the_run_survives() {
        let v = evaluate_soak(&[ok_target(16_384, 0.3)], 0.5, None, "context_soak");
        assert!(!v.passed);
        assert!(v.failures[0].contains("left only 0.30 GB free VRAM"));
        assert!(v.failures[0].contains("required >= 0.50 GB"));
    }

    #[test]
    fn an_exhausted_budget_is_a_failure_not_a_silent_pass() {
        let v = evaluate_soak(&[ok_target(16_384, 1.0)], 0.5, Some(32_768), "context_soak");
        assert!(!v.passed);
        assert!(v.failures[0].contains("budget exhausted before target 32768"));
    }

    #[test]
    fn recovery_ladder_probes_the_smallest_offload_bumps_first() {
        let base = overrides_of(&[
            ("NCpuMoe", 20.into()),
            ("UbatchSize", 1024.into()),
            ("BatchSize", 2048.into()),
        ]);
        let ladder = stress_recovery_overrides(&base, 60, 4);
        // +1 first — most GPU-resident recovery candidate.
        assert_eq!(ladder[0]["NCpuMoe"], serde_json::json!(21));
        assert_eq!(ladder[1]["NCpuMoe"], serde_json::json!(22));
        assert_eq!(ladder[3]["NCpuMoe"], serde_json::json!(24));
        // Then the batch-safe variant at the original offload.
        assert_eq!(ladder[4]["NCpuMoe"], serde_json::json!(20));
        assert_eq!(ladder[4]["UbatchSize"], serde_json::json!(512));
        assert_eq!(ladder[4]["BatchSize"], serde_json::json!(512));
        // Combined bumps at the tail.
        let last = ladder.last().unwrap();
        assert_eq!(last["NCpuMoe"], serde_json::json!(24));
        assert_eq!(last["UbatchSize"], serde_json::json!(512));
    }

    #[test]
    fn recovery_respects_the_moe_upper_bound() {
        let base = overrides_of(&[("NCpuMoe", 59.into())]);
        let ladder = stress_recovery_overrides(&base, 60, 4);
        let moes: Vec<i64> = ladder
            .iter()
            .filter_map(|c| c.get("NCpuMoe").and_then(serde_json::Value::as_i64))
            .collect();
        assert!(moes.iter().all(|m| *m <= 60), "bounded by upper: {moes:?}");
    }

    #[test]
    fn recovery_drops_mtp_and_clamps_the_draft() {
        let base = overrides_of(&[
            ("NCpuMoe", 20.into()),
            ("SpecType", "mtp".into()),
            ("SpecDraftNMax", 3.into()),
        ]);
        let ladder = stress_recovery_overrides(&base, 60, 2);
        // A draft-safe variant keeps MTP but clamps the draft to 1.
        assert!(ladder.iter().any(|c| c.get("SpecType").is_some()
            && c["SpecDraftNMax"] == serde_json::json!(1)
            && !c.contains_key("UbatchSize")));
        // A no-MTP variant drops both spec keys.
        assert!(ladder
            .iter()
            .any(|c| !c.contains_key("SpecType") && !c.contains_key("SpecDraftNMax")));
        // The combined offload bumps clamp the draft too.
        let combo = ladder
            .iter()
            .find(|c| {
                c.get("NCpuMoe") == Some(&serde_json::json!(21)) && c.contains_key("UbatchSize")
            })
            .expect("combined bump");
        assert_eq!(combo["SpecDraftNMax"], serde_json::json!(1));
    }

    #[test]
    fn recovery_emits_each_distinct_config_once() {
        // Base already at 512/512: the batch-safe variant duplicates the combos.
        let base = overrides_of(&[
            ("NCpuMoe", 20.into()),
            ("UbatchSize", 512.into()),
            ("BatchSize", 512.into()),
        ]);
        let ladder = stress_recovery_overrides(&base, 60, 2);
        let sigs: Vec<String> = ladder.iter().map(candidate_signature).collect();
        let mut deduped = sigs.clone();
        deduped.dedup();
        assert_eq!(sigs.len(), deduped.len(), "no duplicate configs");
    }

    #[test]
    fn stress_telemetry_overlays_the_base() {
        let mut base = serde_json::Map::new();
        base.insert("cpu_avg_pct".to_string(), serde_json::json!(50.0));
        base.insert("gpu_vram_free_gb_min".to_string(), serde_json::json!(2.0));
        let mut stress = serde_json::Map::new();
        stress.insert("gpu_vram_free_gb_min".to_string(), serde_json::json!(0.4));
        let merged = join_stress_telemetry(&base, &stress);
        assert_eq!(merged["gpu_vram_free_gb_min"], serde_json::json!(0.4));
        assert_eq!(merged["cpu_avg_pct"], serde_json::json!(50.0));
    }
}
