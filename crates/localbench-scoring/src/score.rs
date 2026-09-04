//! Trial ("pure") score and the balanced score with its factor breakdown.
//!
//! The pure score measures raw speed for the chosen optimization target; the
//! balanced score discounts it by the operational-risk factors — CPU/RAM/VRAM
//! headroom, run-to-run variance, and cross-phase throughput stability — so a
//! config that is fast but marginal never silently wins over one that is fast
//! and safe. Every constant here is a pinned scoring decision; a change must be
//! deliberate, not a port side effect.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::stats::round_dp;

/// What the score optimizes for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Optimize {
    /// Decode (generation) tokens per second.
    Gen,
    /// Prompt (prefill) tokens per second.
    Prompt,
    /// Geometric mean of both — symmetric, penalizes lopsided wins.
    Both,
    /// Estimated end-to-end tokens/sec for a coding-agent request: input plus
    /// output tokens over total wall-clock. No penalty, no weighted mean —
    /// raw e2e speed, so a config with faster prefill but slower decode (or
    /// vice versa) wins only when it is faster end to end.
    CodingAgent,
}

/// The benchmark workload the pure score is computed against.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Workload {
    /// Whether the run measures against the long prompt profile.
    pub long_prompt_profile: bool,
    /// The measured benchmark prompt size (used only on the long profile).
    pub bench_prompt_tokens: f64,
    /// Tokens generated per measurement.
    pub n_predict: u32,
}

impl Default for Workload {
    fn default() -> Self {
        Self {
            long_prompt_profile: false,
            bench_prompt_tokens: 512.0,
            n_predict: 256,
        }
    }
}

/// The representative coding-agent prompt size assumed when the run did not
/// measure a long-context probe and is not on the long prompt profile.
const CODING_AGENT_PROMPT_TOKENS: f64 = 4096.0;

/// Why a trial failed at startup, when it did. Distinguishes an exited process
/// (with or without an OOM signature in its log) from a process that was still
/// running but never answered `/health` within the startup budget — three
/// diagnoses that otherwise collapse into the same `startup_ok = false`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StartupFailure {
    /// The server process exited during startup with an OOM signature in its log.
    ExitedOom,
    /// The server process exited during startup with no OOM signature.
    Exited,
    /// The server process was still running but never answered within the budget.
    TimedOut,
}

/// The lifecycle stage at which a trial became unusable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrialFailureStage {
    /// Candidate overrides, launcher arguments, binary resolution, or spawn.
    Launch,
    /// The spawned server did not reach the ready state.
    Readiness,
    /// The chat request could not be transported.
    Request,
    /// The server returned an HTTP or response-envelope failure.
    Response,
    /// The response had no acceptable user-visible content.
    Content,
}

/// A stable, machine-readable reason for an unusable trial.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrialFailureReason {
    InvalidOverrides,
    PortUnavailable,
    ArgumentConstruction,
    BinaryUnavailable,
    SpawnFailed,
    ReadinessExitedOom,
    ReadinessExited,
    ReadinessTimeout,
    Transport,
    HttpStatus,
    ResponseDecode,
    ResponseSchema,
    MissingTimings,
    InvalidTimings,
    EmptyContent,
    ThinkingOnly,
    DegenerateContent,
}

/// Typed trial failure details shared by scoring, search control, cache, CLI,
/// and diagnostics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrialFailure {
    pub stage: TrialFailureStage,
    pub reason: TrialFailureReason,
    /// Bounded, sanitized detail intended for operators, not request bodies.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub detail: String,
}

impl TrialFailure {
    #[must_use]
    pub fn summary(&self) -> String {
        let stage = serde_json::to_value(self.stage)
            .ok()
            .and_then(|value| value.as_str().map(str::to_string))
            .unwrap_or_else(|| "unknown".to_string());
        let reason = serde_json::to_value(self.reason)
            .ok()
            .and_then(|value| value.as_str().map(str::to_string))
            .unwrap_or_else(|| "unknown".to_string());
        if self.detail.is_empty() {
            format!("{stage}/{reason}")
        } else {
            format!("{stage}/{reason}: {}", self.detail)
        }
    }
}

/// Durable pointers to the evidence for one attempt.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrialDiagnosticRef {
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub attempt_id: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub manifest_path: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub log_path: String,
}

/// One measured trial: the throughput numbers and health flags the scores read.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Trial {
    /// The server started and answered.
    pub startup_ok: bool,
    /// The trial hit an out-of-memory failure.
    pub oom: bool,
    /// Why startup failed, when it did — `None` for a success or a non-startup
    /// failure (e.g. degenerate output). Serde-defaulted so trials cached before
    /// this field existed still load, and skipped when absent so a trial's JSON
    /// is unchanged unless a startup diagnosis is actually recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub startup_failure: Option<StartupFailure>,
    /// True only when startup, protocol, timings, and visible-content gates all
    /// completed successfully.
    #[serde(default)]
    pub measurement_usable: bool,
    /// Typed failure for any unusable trial.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure: Option<TrialFailure>,
    /// Exact evidence pointers, populated by the live/cached runner.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diagnostic: Option<TrialDiagnosticRef>,
    /// Child exit code when one was observed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub process_status: Option<i32>,
    /// Sanitized, bounded runtime evidence associated with the outcome.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub diagnostic_excerpt: String,
    /// Authoritative runtime fields observed from the engine. Requested values
    /// are kept separately in the run manifest.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub observed_configuration: BTreeMap<String, serde_json::Value>,
    /// Final argv (without the binary) used for the candidate launch.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub launch_args: Vec<String>,
    /// Bounded log-derived observations that may explain an engine adjustment.
    /// These are explicitly advisory and never treated as effective settings.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub advisory_observations: Vec<String>,
    /// Prompt-processing tokens/sec (search prompt).
    pub pp_tps: f64,
    /// Decode tokens/sec (search prompt).
    pub tg_tps: f64,
    /// Prompt-processing tokens/sec at the long-context probe, when measured.
    pub long_ctx_pp_tps: Option<f64>,
    /// Decode tokens/sec at the long-context probe, when measured.
    pub long_ctx_tg_tps: Option<f64>,
    /// The long-context probe's prompt size, when measured.
    pub long_ctx_target_tokens: Option<u32>,
    /// Run-to-run throughput variance for this trial, when measured.
    pub variance: Option<f64>,
    /// Host telemetry sampled during the trial.
    pub telemetry: Telemetry,
}

impl Trial {
    /// Whether this trial is safe to score, rank, verify, cache as a success,
    /// or export.
    #[must_use]
    pub fn is_measurement_usable(&self) -> bool {
        self.startup_ok
            && !self.oom
            && self.measurement_usable
            && self.failure.is_none()
            && self.pp_tps.is_finite()
            && self.tg_tps.is_finite()
            && self.pp_tps > 0.0
            && self.tg_tps > 0.0
    }

    /// Whether this is genuine startup/fit evidence that may enter a memory
    /// recovery ladder.
    #[must_use]
    pub fn needs_memory_recovery(&self) -> bool {
        self.oom
            || self
                .failure
                .as_ref()
                .is_some_and(|failure| failure.stage == TrialFailureStage::Readiness)
    }

    /// Whether a different KV cache pair could plausibly fix this trial: the
    /// server started, stayed inside memory, and still returned a reply the
    /// content gates rejected.
    ///
    /// Deliberately separate from [`Self::needs_memory_recovery`] rather than
    /// folded into it. That predicate routes to a memory ladder, which for a
    /// MoE model pins the KV pair and sweeps only the expert lever — the wrong
    /// axis entirely here — and it would have a content failure scored,
    /// logged, and explained as memory pressure.
    #[must_use]
    pub fn needs_kv_recovery(&self) -> bool {
        self.startup_ok
            && !self.oom
            && self
                .failure
                .as_ref()
                .is_some_and(|failure| failure.stage == TrialFailureStage::Content)
    }
}

/// Host telemetry sampled during a trial. Every field optional — an absent
/// signal never penalizes.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Telemetry {
    /// Average CPU utilisation percent across the trial.
    pub cpu_avg_pct: Option<f64>,
    /// Minimum available system RAM (GB) observed.
    pub ram_available_gb_min: Option<f64>,
    /// Minimum free VRAM (GB) observed.
    pub gpu_vram_free_gb_min: Option<f64>,
    /// Standard deviation of the free-VRAM samples (GB).
    pub gpu_vram_free_gb_std: Option<f64>,
    /// Number of free-VRAM samples taken.
    pub gpu_vram_free_gb_samples: Option<u32>,
    /// Total VRAM on the card (GB).
    pub gpu_vram_total_gb: Option<f64>,
}

/// Telemetry inputs consumed by [`balanced_score`]. The live producer tests
/// this contract so a newly scored signal cannot silently remain unmeasured.
pub const BALANCED_TELEMETRY_FIELDS: [&str; 6] = [
    "cpu_avg_pct",
    "ram_available_gb_min",
    "gpu_vram_free_gb_min",
    "gpu_vram_free_gb_std",
    "gpu_vram_free_gb_samples",
    "gpu_vram_total_gb",
];

impl Telemetry {
    /// Balanced-score inputs that were unavailable for this trial.
    #[must_use]
    pub fn missing_balanced_fields(&self) -> Vec<&'static str> {
        BALANCED_TELEMETRY_FIELDS
            .into_iter()
            .zip([
                self.cpu_avg_pct.is_some(),
                self.ram_available_gb_min.is_some(),
                self.gpu_vram_free_gb_min.is_some(),
                self.gpu_vram_free_gb_std.is_some(),
                self.gpu_vram_free_gb_samples.is_some(),
                self.gpu_vram_total_gb.is_some(),
            ])
            .filter_map(|(field, present)| (!present).then_some(field))
            .collect()
    }
}

/// Config overrides as arbitrary key/value pairs (the tuner's config axes,
/// e.g. `NCpuMoe`, `UbatchSize`, `Threads`, `Mlock`).
pub type Overrides = BTreeMap<String, serde_json::Value>;

/// The pure (unrounded) score for a trial under an optimization target. A trial
/// that OOMed or failed to start scores `0.0` — health gates the score.
///
/// The long-context probe's pp/tg are preferred when present: the search-time
/// measurement uses a small prompt that does not expose the KV-cache pressure
/// which crushes marginal configs at the configured workload size. Anchoring in
/// long-context truth lets the pure score reject configs that look fast at 2k
/// tokens but thrash at 64k+.
#[must_use]
pub fn trial_score(trial: &Trial, optimize: Optimize, workload: &Workload) -> f64 {
    if !trial.is_measurement_usable() {
        return 0.0;
    }

    let long_ctx = match (trial.long_ctx_pp_tps, trial.long_ctx_tg_tps) {
        (Some(pp), Some(tg)) if pp > 0.0 && tg > 0.0 => Some((pp, tg)),
        _ => None,
    };
    let (pp, tg) = long_ctx.unwrap_or((trial.pp_tps, trial.tg_tps));

    match optimize {
        Optimize::Gen => tg,
        Optimize::Prompt => pp,
        Optimize::Both => {
            if pp <= 0.0 || tg <= 0.0 {
                0.0
            } else {
                (pp * tg).sqrt()
            }
        }
        Optimize::CodingAgent => {
            if pp <= 0.0 || tg <= 0.0 {
                return 0.0;
            }
            let prompt_tokens = match (long_ctx, trial.long_ctx_target_tokens) {
                (Some(_), Some(tokens)) if tokens > 0 => f64::from(tokens),
                _ if workload.long_prompt_profile => workload.bench_prompt_tokens,
                _ => CODING_AGENT_PROMPT_TOKENS,
            };
            let gen_tokens = f64::from(workload.n_predict.max(1));
            let latency_sec = (prompt_tokens / pp) + (gen_tokens / tg);
            if latency_sec <= 0.0 {
                0.0
            } else {
                (prompt_tokens + gen_tokens) / latency_sec
            }
        }
    }
}

/// Tuning knobs for the statistical VRAM-headroom discount.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VramHeadroomParams {
    /// Absolute sigma floor (GB) for unmodeled/external risk.
    pub sigma_floor_gb: f64,
    /// With no trustworthy jitter sample, assume the margin could swing by
    /// this fraction of the card.
    pub fallback_jitter_fraction: f64,
    /// Full credit at/above this many sigmas of margin.
    pub z_full: f64,
    /// The factor bottoms out here as the margin shrinks to the noise floor.
    pub min_factor: f64,
    /// Minimum samples for the measured jitter to be trusted.
    pub min_samples: u32,
}

impl Default for VramHeadroomParams {
    fn default() -> Self {
        Self {
            sigma_floor_gb: 0.30,
            fallback_jitter_fraction: 0.05,
            z_full: 3.0,
            min_factor: 0.60,
            min_samples: 5,
        }
    }
}

/// Statistical VRAM-headroom discount for the balanced score. Instead of fixed
/// absolute GB bands (which over-penalize small cards and under-penalize large
/// ones), penalize the free-VRAM margin relative to its *uncertainty*:
/// `z = free_min / sigma_eff` (how many sigmas of safety margin remain). Full
/// credit at `z >= z_full`, bottoming at `min_factor` as the margin shrinks
/// toward the noise floor. `sigma_eff` combines measured run-to-run jitter
/// with an absolute floor; with no reliable jitter data the floor scales with
/// card size so 6GB and 48GB cards both behave.
#[must_use]
pub fn vram_headroom_factor(
    free_gb_min: f64,
    std_gb: Option<f64>,
    samples: u32,
    total_gb: f64,
    params: &VramHeadroomParams,
) -> f64 {
    if free_gb_min <= 0.0 {
        return params.min_factor;
    }
    let z_full = if params.z_full <= 0.0 {
        3.0
    } else {
        params.z_full
    };

    let sigma = match std_gb {
        // Measured jitter plus an absolute floor for unmodeled risk.
        Some(std) if std >= 0.0 && samples >= params.min_samples => {
            (std * std + params.sigma_floor_gb * params.sigma_floor_gb).sqrt()
        }
        // No trustworthy jitter sample: assume the margin could swing by a
        // small fraction of the card, never below the absolute floor.
        _ => {
            let fallback = if total_gb > 0.0 {
                params.fallback_jitter_fraction * total_gb
            } else {
                0.0
            };
            params.sigma_floor_gb.max(fallback)
        }
    };
    if sigma <= 0.0 {
        return 1.0;
    }

    let z = free_gb_min / sigma;
    let factor = params.min_factor + (1.0 - params.min_factor) * (z / z_full);
    round_dp(factor.clamp(params.min_factor, 1.0), 4)
}

/// The config axes that legitimately determine throughput, in group-key order.
/// Flash/SWA/mlock/cache flags are deliberately excluded so repeated
/// measurements of the same base config across phases land in one group and
/// reveal run-to-run instability (not config-driven variation).
const STABILITY_KEY_AXES: &[&str] = &[
    "NCpuMoe",
    "NGpuLayers",
    "UbatchSize",
    "BatchSize",
    "KvK",
    "KvV",
    "SpecType",
    "SpecDraftNMax",
];

/// Group key for cross-phase stability over the throughput-determining axes.
#[must_use]
pub fn stability_group_key(overrides: &Overrides) -> String {
    let parts: Vec<String> = STABILITY_KEY_AXES
        .iter()
        .filter_map(|axis| {
            let value = overrides.get(*axis)?;
            if value.is_null() {
                return None;
            }
            Some(format!("{axis}={}", render_value(value)))
        })
        .collect();
    parts.join(";")
}

/// Render an override value the way the group key and signatures expect:
/// bare strings (no quotes), numbers and bools as written.
fn render_value(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// One historical trial record, as the cross-phase stability index reads it.
#[derive(Debug, Clone)]
pub struct HistoryTrial {
    /// The search phase that measured it.
    pub phase: String,
    /// The config it measured.
    pub overrides: Overrides,
    /// The server started and answered.
    pub startup_ok: bool,
    /// The trial hit an out-of-memory failure.
    pub oom: bool,
    /// Decode tokens/sec measured.
    pub tg_tps: f64,
}

/// For each throughput-determining config group, the coefficient of variation
/// of decode tok/s across the search phases. A VRAM-marginal config (e.g. one
/// that measures 51 tok/s in one phase and 20 in another) shows a high CV — an
/// instability signal the balanced score penalizes. Soak/guard/recovery phases
/// are excluded (their long-context collapse is handled separately).
#[must_use]
pub fn cross_phase_stability_index(history: &[HistoryTrial]) -> BTreeMap<String, f64> {
    let mut groups: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for trial in history {
        if !trial.startup_ok || trial.oom {
            continue;
        }
        if trial.phase.starts_with("context_soak")
            || trial.phase.starts_with("stress_recovery")
            || trial.phase.starts_with("context_guard")
        {
            continue;
        }
        if trial.tg_tps <= 0.0 {
            continue;
        }
        groups
            .entry(stability_group_key(&trial.overrides))
            .or_default()
            .push(trial.tg_tps);
    }

    let mut index = BTreeMap::new();
    for (key, vals) in groups {
        if vals.len() < 2 {
            continue;
        }
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        if mean <= 0.0 {
            continue;
        }
        let sum_sq: f64 = vals.iter().map(|v| (v - mean) * (v - mean)).sum();
        let std = (sum_sq / (vals.len() as f64 - 1.0)).sqrt();
        index.insert(key, round_dp(std / mean, 4));
    }
    index
}

/// Map a throughput coefficient-of-variation to a balanced-score multiplier:
/// full credit below `low_cv`, sliding to `min_factor` at/above `high_cv`.
#[must_use]
pub fn stability_factor_from_cv(cv: f64) -> f64 {
    const LOW_CV: f64 = 0.08;
    const HIGH_CV: f64 = 0.30;
    const MIN_FACTOR: f64 = 0.80;

    if cv <= LOW_CV {
        return 1.0;
    }
    if cv >= HIGH_CV {
        return MIN_FACTOR;
    }
    let frac = (cv - LOW_CV) / (HIGH_CV - LOW_CV);
    round_dp(1.0 - frac * (1.0 - MIN_FACTOR), 4)
}

/// A config can survive the soak (no crash/OOM) yet have its long-context
/// decode collapse versus its short-prompt search measurement — a sign it is
/// thrashing at the VRAM edge and is not actually usable. Flag that so it is
/// never saved, and so edge/max-utilization modes stay safe when the hard
/// free-VRAM floor is relaxed.
#[must_use]
pub fn soak_throughput_collapsed(
    soak_tg: f64,
    search_tg: f64,
    collapse_fraction: f64,
    abs_floor_tg: f64,
) -> bool {
    if soak_tg <= 0.0 {
        return true;
    }
    if abs_floor_tg > 0.0 && soak_tg < abs_floor_tg {
        return true;
    }
    if search_tg > 0.0 && soak_tg < collapse_fraction * search_tg {
        return true;
    }
    false
}

/// The default soak-collapse fraction: long-context decode below half the
/// search measurement is a collapse.
pub const SOAK_COLLAPSE_FRACTION: f64 = 0.5;

/// The balanced score's per-factor breakdown, kept alongside the score so a
/// report can show *why* a config was discounted.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoreBreakdown {
    pub cpu_headroom_factor: f64,
    pub ram_headroom_factor: f64,
    pub vram_headroom_factor: f64,
    pub variance_factor: f64,
    pub stability_factor: f64,
    /// Confidence label carried on the breakdown (telemetry coverage is
    /// partial by construction — factors default to full credit when a signal
    /// is absent).
    pub confidence: String,
    /// Exact balanced-score telemetry inputs that were unavailable. This keeps
    /// full-credit defaults observable instead of making missing data resemble
    /// a genuinely idle host.
    pub missing_telemetry: Vec<String>,
}

impl Default for ScoreBreakdown {
    fn default() -> Self {
        Self {
            cpu_headroom_factor: 1.0,
            ram_headroom_factor: 1.0,
            vram_headroom_factor: 1.0,
            variance_factor: 1.0,
            stability_factor: 1.0,
            confidence: "partial".to_string(),
            missing_telemetry: BALANCED_TELEMETRY_FIELDS
                .iter()
                .map(ToString::to_string)
                .collect(),
        }
    }
}

/// A balanced score plus its factor breakdown.
#[derive(Debug, Clone, PartialEq)]
pub struct BalancedScore {
    /// The discounted score, rounded to 2 decimal places.
    pub score: f64,
    /// The factors that produced it.
    pub breakdown: ScoreBreakdown,
}

/// Host facts the balanced score reads (injected, never probed here, so the
/// math stays pure and testable).
#[derive(Debug, Clone, Copy, Default)]
pub struct HostSignals {
    /// Logical processor count, for the CPU-headroom factor. `0` = unknown.
    pub logical_cores: u32,
}

/// Compute the balanced score: the pure score discounted by the CPU/RAM/VRAM
/// headroom, variance, and cross-phase stability factors. A zero pure score,
/// an OOM, or a failed startup scores `0.0` with a zeroed stability factor.
#[must_use]
pub fn balanced_score(
    trial: &Trial,
    overrides: &Overrides,
    pure_score: f64,
    host: HostSignals,
    vram_params: &VramHeadroomParams,
    stability_index: &BTreeMap<String, f64>,
) -> BalancedScore {
    let missing_telemetry: Vec<String> = trial
        .telemetry
        .missing_balanced_fields()
        .into_iter()
        .map(ToString::to_string)
        .collect();
    let confidence = if missing_telemetry.is_empty() {
        "full"
    } else {
        "partial"
    };
    let mut breakdown = ScoreBreakdown {
        confidence: confidence.to_string(),
        missing_telemetry,
        ..ScoreBreakdown::default()
    };

    if pure_score <= 0.0 || !trial.is_measurement_usable() {
        breakdown.stability_factor = 0.0;
        return BalancedScore {
            score: 0.0,
            breakdown,
        };
    }

    // CPU headroom: reserving no core for the agent/OS costs a little; exactly
    // one reserved core costs less.
    let threads = overrides
        .get("Threads")
        .and_then(serde_json::Value::as_i64)
        .unwrap_or(0);
    if threads > 0 && host.logical_cores > 0 {
        let reserved = i64::from(host.logical_cores) - threads;
        if reserved <= 0 {
            breakdown.cpu_headroom_factor = 0.90;
        } else if reserved == 1 {
            breakdown.cpu_headroom_factor = 0.96;
        }
    }

    let telemetry = &trial.telemetry;
    if let Some(cpu) = telemetry.cpu_avg_pct {
        if cpu > 95.0 {
            breakdown.cpu_headroom_factor = breakdown.cpu_headroom_factor.min(0.82);
        } else if cpu > 75.0 {
            let factor = 1.0 - ((cpu - 75.0) / 20.0 * 0.12);
            breakdown.cpu_headroom_factor = breakdown.cpu_headroom_factor.min(factor.max(0.88));
        }
    }

    if let Some(ram) = telemetry.ram_available_gb_min {
        if ram < 4.0 {
            breakdown.ram_headroom_factor = 0.78;
        } else if ram < 8.0 {
            breakdown.ram_headroom_factor = 0.88;
        } else if ram < 16.0 {
            breakdown.ram_headroom_factor = 0.96;
        }
    }

    if let Some(vram_free) = telemetry.gpu_vram_free_gb_min {
        breakdown.vram_headroom_factor = vram_headroom_factor(
            vram_free,
            telemetry.gpu_vram_free_gb_std,
            telemetry.gpu_vram_free_gb_samples.unwrap_or(0),
            telemetry.gpu_vram_total_gb.unwrap_or(0.0),
            vram_params,
        );
    }

    if let Some(variance) = trial.variance {
        if variance > 0.15 {
            breakdown.variance_factor = 0.90;
        } else if variance > 0.08 {
            breakdown.variance_factor = 0.96;
        }
    }

    // mlock + no-mmap pins the model in RAM; with no RAM telemetry to clear
    // it, nudge the RAM factor conservatively.
    let flag = |name: &str| {
        overrides
            .get(name)
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
    };
    if flag("Mlock") && flag("NoMmap") && telemetry.ram_available_gb_min.is_none() {
        breakdown.ram_headroom_factor = breakdown.ram_headroom_factor.min(0.96);
    }

    // Cross-phase throughput stability: a config whose decode varies wildly
    // across phases is VRAM-marginal/unreliable even when this single
    // measurement looks fast.
    if let Some(cv) = stability_index.get(&stability_group_key(overrides)) {
        breakdown.stability_factor = stability_factor_from_cv(*cv);
    }

    let score = pure_score
        * breakdown.cpu_headroom_factor
        * breakdown.ram_headroom_factor
        * breakdown.vram_headroom_factor
        * breakdown.variance_factor
        * breakdown.stability_factor;

    BalancedScore {
        score: round_dp(score, 2),
        breakdown,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn overrides(pairs: &[(&str, serde_json::Value)]) -> Overrides {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect()
    }

    fn canned_trial(pp: f64, tg: f64, variance: f64, vram_free_min: f64, oom: bool) -> Trial {
        Trial {
            startup_ok: true,
            oom,
            measurement_usable: !oom,
            pp_tps: pp,
            tg_tps: tg,
            variance: Some(variance),
            telemetry: Telemetry {
                gpu_vram_free_gb_min: Some(vram_free_min),
                ..Telemetry::default()
            },
            ..Trial::default()
        }
    }

    /// The canned regression trio: comfortable, fast-but-tight, noisy-slow.
    fn comfortable() -> Trial {
        canned_trial(700.0, 50.0, 0.01, 1.5, false)
    }
    fn fast_but_tight() -> Trial {
        canned_trial(900.0, 40.0, 0.02, 0.4, false)
    }
    fn noisy_slow() -> Trial {
        canned_trial(400.0, 60.0, 0.3, 3.0, false)
    }

    fn pure(trial: &Trial) -> f64 {
        trial_score(trial, Optimize::CodingAgent, &Workload::default())
    }

    fn balanced(trial: &Trial, ov: &Overrides) -> BalancedScore {
        balanced_score(
            trial,
            ov,
            pure(trial),
            HostSignals::default(),
            &VramHeadroomParams::default(),
            &BTreeMap::new(),
        )
    }

    #[test]
    fn pinned_coding_agent_pure_scores() {
        // (4096 + 256) / (4096/pp + 256/tg), rounded to 2dp for the pin.
        assert_eq!(round_dp(pure(&comfortable()), 2), 396.67);
        assert_eq!(round_dp(pure(&fast_but_tight()), 2), 397.4);
        assert_eq!(round_dp(pure(&noisy_slow()), 2), 300.0);
    }

    #[test]
    fn tight_vram_headroom_is_penalized_in_the_balanced_score() {
        let ov = overrides(&[("NCpuMoe", 20.into()), ("UbatchSize", 1024.into())]);
        let b = balanced(&fast_but_tight(), &ov);
        // 0.4GB free over the 0.30GB sigma floor: z=1.33 → 0.60+0.40*(z/3).
        assert_eq!(b.breakdown.vram_headroom_factor, 0.7778);
        assert_eq!(b.score, 309.1);
    }

    #[test]
    fn high_variance_is_penalized_in_the_balanced_score() {
        let ov = overrides(&[("NCpuMoe", 30.into())]);
        let b = balanced(&noisy_slow(), &ov);
        assert_eq!(b.breakdown.variance_factor, 0.9);
        assert_eq!(b.score, 270.0);
    }

    #[test]
    fn a_comfortable_candidate_is_unpenalized() {
        let ov = overrides(&[("NCpuMoe", 10.into()), ("UbatchSize", 512.into())]);
        let b = balanced(&comfortable(), &ov);
        assert_eq!(b.breakdown.vram_headroom_factor, 1.0);
        assert_eq!(b.breakdown.variance_factor, 1.0);
        assert_eq!(b.score, round_dp(pure(&comfortable()), 2));
    }

    #[test]
    fn oom_or_failed_startup_scores_zero() {
        let ov = Overrides::new();
        let oom = canned_trial(9999.0, 999.0, 0.01, 2.0, true);
        assert_eq!(pure(&oom), 0.0);
        let b = balanced(&oom, &ov);
        assert_eq!(b.score, 0.0);
        assert_eq!(b.breakdown.stability_factor, 0.0);

        let mut dead = comfortable();
        dead.startup_ok = false;
        assert_eq!(pure(&dead), 0.0);
    }

    #[test]
    fn incomplete_or_non_finite_measurements_never_score() {
        let mut cases = Vec::new();

        let mut not_explicitly_usable = comfortable();
        not_explicitly_usable.measurement_usable = false;
        cases.push(not_explicitly_usable);

        let mut zero_prompt = comfortable();
        zero_prompt.pp_tps = 0.0;
        cases.push(zero_prompt);

        let mut zero_decode = comfortable();
        zero_decode.tg_tps = 0.0;
        cases.push(zero_decode);

        let mut nan_prompt = comfortable();
        nan_prompt.pp_tps = f64::NAN;
        cases.push(nan_prompt);

        let mut failed = comfortable();
        failed.failure = Some(TrialFailure {
            stage: TrialFailureStage::Content,
            reason: TrialFailureReason::DegenerateContent,
            detail: String::new(),
        });
        cases.push(failed);

        for trial in cases {
            assert!(!trial.is_measurement_usable());
            assert_eq!(pure(&trial), 0.0);
        }
    }

    #[test]
    fn long_context_probe_anchors_the_score_when_present() {
        // Fast at the search prompt, collapsed at long context: the score must
        // follow the long-context truth.
        let mut trial = comfortable();
        trial.long_ctx_pp_tps = Some(300.0);
        trial.long_ctx_tg_tps = Some(10.0);
        trial.long_ctx_target_tokens = Some(65536);
        let with_probe = pure(&trial);
        assert!(
            with_probe < pure(&comfortable()),
            "long-context collapse must lower the score (got {with_probe})"
        );
        // The probe's own prompt size feeds the e2e estimate.
        let expected = (65536.0 + 256.0) / (65536.0 / 300.0 + 256.0 / 10.0);
        assert!((with_probe - expected).abs() < 1e-9);
    }

    #[test]
    fn optimize_targets_read_the_right_axis() {
        let trial = comfortable();
        let w = Workload::default();
        assert_eq!(trial_score(&trial, Optimize::Gen, &w), 50.0);
        assert_eq!(trial_score(&trial, Optimize::Prompt, &w), 700.0);
        let both = trial_score(&trial, Optimize::Both, &w);
        assert!((both - (700.0f64 * 50.0).sqrt()).abs() < 1e-9);
    }

    #[test]
    fn vram_zero_jitter_still_respects_the_sigma_floor() {
        // sigma_eff = sqrt(0 + 0.3^2) = 0.3, z = 0.6/0.3 = 2.0 → 0.6 + 0.4*2/3.
        let f = vram_headroom_factor(0.6, Some(0.0), 20, 24.0, &VramHeadroomParams::default());
        assert_eq!(f, 0.8667);
    }

    #[test]
    fn vram_stable_small_card_gets_full_credit() {
        // z = 1.0 / sqrt(0.1^2 + 0.3^2) = 3.16 → full credit; absolute GB
        // bands would have penalized a 6GB card for being under 1.5GB free.
        let f = vram_headroom_factor(1.0, Some(0.10), 20, 6.0, &VramHeadroomParams::default());
        assert_eq!(f, 1.0);
    }

    #[test]
    fn vram_jitter_penalizes_and_stays_in_band() {
        let params = VramHeadroomParams::default();
        let stable = vram_headroom_factor(1.0, Some(0.10), 20, 6.0, &params);
        let jittery = vram_headroom_factor(1.0, Some(0.50), 20, 6.0, &params);
        assert!(jittery < stable);
        assert!((0.60..=0.95).contains(&jittery));
    }

    #[test]
    fn vram_factor_is_monotonic_in_free_vram() {
        let params = VramHeadroomParams::default();
        let mut prev = 0.0;
        for free in [0.3, 0.6, 1.0, 1.5, 2.0, 3.0] {
            let f = vram_headroom_factor(free, Some(0.10), 20, 6.0, &params);
            assert!(
                f >= prev,
                "must be monotonic: free={free} got {f} after {prev}"
            );
            prev = f;
        }
    }

    #[test]
    fn vram_factor_is_clamped_to_its_band() {
        let params = VramHeadroomParams::default();
        assert!(vram_headroom_factor(0.05, Some(0.40), 20, 6.0, &params) >= 0.60);
        assert_eq!(
            vram_headroom_factor(12.0, Some(0.10), 20, 24.0, &params),
            1.0
        );
        // No free VRAM at all bottoms out at the floor.
        assert_eq!(vram_headroom_factor(0.0, None, 0, 24.0, &params), 0.60);
    }

    #[test]
    fn vram_no_data_fallback_scales_with_card_size() {
        let params = VramHeadroomParams::default();
        // 6GB: sigma = max(0.30, 0.05*6=0.30) → z = 3.33 → full credit.
        assert_eq!(vram_headroom_factor(1.0, None, 0, 6.0, &params), 1.0);
        // 48GB: sigma = max(0.30, 2.4) → z = 0.42 → heavily penalized.
        let big = vram_headroom_factor(1.0, None, 0, 48.0, &params);
        assert!(
            big < 0.75,
            "48GB card with 1GB free must be penalized: {big}"
        );
    }

    #[test]
    fn vram_too_few_samples_uses_the_no_data_fallback() {
        let params = VramHeadroomParams::default();
        let few = vram_headroom_factor(1.0, Some(0.01), 2, 48.0, &params);
        let none = vram_headroom_factor(1.0, None, 0, 48.0, &params);
        assert_eq!(few, none);
    }

    #[test]
    fn stability_factor_slides_between_the_cv_bands() {
        assert_eq!(stability_factor_from_cv(0.05), 1.0);
        assert_eq!(stability_factor_from_cv(0.08), 1.0);
        assert_eq!(stability_factor_from_cv(0.30), 0.80);
        assert_eq!(stability_factor_from_cv(0.50), 0.80);
        let mid = stability_factor_from_cv(0.19);
        assert_eq!(mid, 0.90, "midpoint of the band slides halfway");
    }

    #[test]
    fn stability_group_key_reads_only_throughput_axes() {
        let ov = overrides(&[
            ("NCpuMoe", 20.into()),
            ("UbatchSize", 1024.into()),
            ("FlashAttention", true.into()),
            ("Mlock", true.into()),
        ]);
        assert_eq!(stability_group_key(&ov), "NCpuMoe=20;UbatchSize=1024");
        // Same base config with different flash/mlock flags lands in the SAME
        // group, so cross-phase noise is visible as instability.
        let ov2 = overrides(&[("NCpuMoe", 20.into()), ("UbatchSize", 1024.into())]);
        assert_eq!(stability_group_key(&ov), stability_group_key(&ov2));
    }

    #[test]
    fn cross_phase_index_flags_unstable_groups_and_skips_soak_phases() {
        let ov = overrides(&[("NCpuMoe", 20.into())]);
        let t = |phase: &str, tg: f64| HistoryTrial {
            phase: phase.to_string(),
            overrides: ov.clone(),
            startup_ok: true,
            oom: false,
            tg_tps: tg,
        };
        // 51 in one phase, 20 in another: a VRAM-marginal instability signal.
        let history = vec![
            t("seed", 51.0),
            t("beam_1", 20.0),
            t("context_soak_final", 5.0),  // excluded
            t("stress_recovery_1", 4.0),   // excluded
            t("context_guard_check", 3.0), // excluded
        ];
        let index = cross_phase_stability_index(&history);
        let cv = index.get("NCpuMoe=20").copied().expect("group indexed");
        assert!(cv > 0.30, "51 vs 20 tok/s is a high CV (got {cv})");
        assert_eq!(stability_factor_from_cv(cv), 0.80);

        // A single measurement (after exclusions) produces no CV entry.
        let single = cross_phase_stability_index(&[t("seed", 51.0), t("context_soak_x", 5.0)]);
        assert!(single.is_empty());
    }

    #[test]
    fn balanced_score_applies_the_stability_factor() {
        let ov = overrides(&[("NCpuMoe", 20.into()), ("UbatchSize", 1024.into())]);
        let mut index = BTreeMap::new();
        index.insert(stability_group_key(&ov), 0.40);
        let trial = comfortable();
        let b = balanced_score(
            &trial,
            &ov,
            pure(&trial),
            HostSignals::default(),
            &VramHeadroomParams::default(),
            &index,
        );
        assert_eq!(b.breakdown.stability_factor, 0.80);
    }

    #[test]
    fn soak_collapse_detection() {
        // Long-context decode below half the search measurement is a collapse.
        assert!(soak_throughput_collapsed(
            20.0,
            51.0,
            SOAK_COLLAPSE_FRACTION,
            0.0
        ));
        assert!(!soak_throughput_collapsed(
            30.0,
            51.0,
            SOAK_COLLAPSE_FRACTION,
            0.0
        ));
        // A dead soak is always a collapse; the absolute floor also gates.
        assert!(soak_throughput_collapsed(
            0.0,
            51.0,
            SOAK_COLLAPSE_FRACTION,
            0.0
        ));
        assert!(soak_throughput_collapsed(
            4.0,
            0.0,
            SOAK_COLLAPSE_FRACTION,
            5.0
        ));
        assert!(!soak_throughput_collapsed(
            6.0,
            0.0,
            SOAK_COLLAPSE_FRACTION,
            5.0
        ));
    }

    #[test]
    fn cpu_and_ram_headroom_factors() {
        let ov = overrides(&[("Threads", 16.into())]);
        let trial = comfortable();
        // All 16 cores consumed: reserved <= 0.
        let b = balanced_score(
            &trial,
            &ov,
            pure(&trial),
            HostSignals { logical_cores: 16 },
            &VramHeadroomParams::default(),
            &BTreeMap::new(),
        );
        assert_eq!(b.breakdown.cpu_headroom_factor, 0.90);

        // Low available RAM discounts.
        let mut tight_ram = comfortable();
        tight_ram.telemetry.ram_available_gb_min = Some(3.0);
        let b = balanced(&tight_ram, &Overrides::new());
        assert_eq!(b.breakdown.ram_headroom_factor, 0.78);

        // mlock+no-mmap with no RAM telemetry nudges conservatively.
        let pinned = overrides(&[("Mlock", true.into()), ("NoMmap", true.into())]);
        let b = balanced(&comfortable(), &pinned);
        assert_eq!(b.breakdown.ram_headroom_factor, 0.96);

        // Saturated CPU telemetry discounts hard.
        let mut hot = comfortable();
        hot.telemetry.cpu_avg_pct = Some(97.0);
        let b = balanced(&hot, &Overrides::new());
        assert_eq!(b.breakdown.cpu_headroom_factor, 0.82);
    }

    #[test]
    fn planted_cpu_and_vram_pressure_flip_balanced_ordering() {
        let complete = |cpu, free| Telemetry {
            cpu_avg_pct: Some(cpu),
            ram_available_gb_min: Some(32.0),
            gpu_vram_free_gb_min: Some(free),
            gpu_vram_free_gb_std: Some(0.0),
            gpu_vram_free_gb_samples: Some(4),
            gpu_vram_total_gb: Some(24.0),
        };
        let fast_but_hot = Trial {
            startup_ok: true,
            measurement_usable: true,
            pp_tps: 1.0,
            tg_tps: 1.0,
            telemetry: complete(99.0, 0.2),
            ..Trial::default()
        };
        let slower_with_headroom = Trial {
            startup_ok: true,
            measurement_usable: true,
            pp_tps: 1.0,
            tg_tps: 1.0,
            telemetry: complete(40.0, 8.0),
            ..Trial::default()
        };
        let params = VramHeadroomParams::default();
        let fast = balanced_score(
            &fast_but_hot,
            &Overrides::new(),
            100.0,
            HostSignals::default(),
            &params,
            &BTreeMap::new(),
        );
        let slower = balanced_score(
            &slower_with_headroom,
            &Overrides::new(),
            90.0,
            HostSignals::default(),
            &params,
            &BTreeMap::new(),
        );
        assert!(
            fast.score < slower.score,
            "{fast:?} should rank below {slower:?}"
        );
        assert_eq!(fast.breakdown.confidence, "full");
        assert!(fast.breakdown.missing_telemetry.is_empty());
    }

    #[test]
    fn missing_balanced_telemetry_is_named_in_the_breakdown() {
        let score = balanced(&comfortable(), &Overrides::new());
        assert_eq!(score.breakdown.confidence, "partial");
        assert!(score
            .breakdown
            .missing_telemetry
            .contains(&"cpu_avg_pct".to_string()));
        assert_eq!(Telemetry::default().missing_balanced_fields().len(), 6);
    }

    #[test]
    fn every_telemetry_field_belongs_to_the_balanced_producer_contract() {
        let serialized = serde_json::to_value(Telemetry::default()).unwrap();
        let mut serialized_fields: Vec<&str> = serialized
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        serialized_fields.sort_unstable();
        let mut producer_fields = BALANCED_TELEMETRY_FIELDS.to_vec();
        producer_fields.sort_unstable();
        assert_eq!(serialized_fields, producer_fields);
    }
}
