//! Live trial measurement: spawn a llama-server for a candidate config over
//! the launcher contract, measure templated chat throughput, classify
//! failures, and tear the server down.
//!
//! The [`TrialRunner`] trait is the seam the tuner drives — the live impl
//! here, a scripted mock in tests.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Child;
use std::time::{Duration, Instant};

use localbench_measure::cache::stable_json_hash;
use localbench_measure::classify::{
    is_oom_message, output_quality_ok, QUALITY_MIN_CHARS, QUALITY_MIN_WORDS,
};
use localbench_measure::prompt::coding_agent_stress_prompt;
use localbench_scoring::score::Overrides;
use localbench_scoring::score::{
    StartupFailure, Telemetry, Trial, TrialDiagnosticRef, TrialFailure, TrialFailureReason,
    TrialFailureStage,
};
use localbench_scoring::stats::median;
use localbench_search::overrides::candidate_signature;
use localbox_launcher::orchestrate::LaunchRequest;
use localbox_launcher::smoke::evaluate_smoke_text;
use localx_llama_core::args::{build_llama_server_args, LaunchParams};
use localx_llama_core::{Launcher, Mode, ModelDef};
use localx_llama_runtime::spawn::spawn_detached;
use serde::{Deserialize, Serialize};

/// Versioned public inference surface used by the live tuner.
pub const MEASUREMENT_PROTOCOL: &str = "openai-chat-completions-v1";
/// Host resource sampler whose outputs participate in balanced scoring.
pub const TELEMETRY_PROTOCOL: &str = "host-cpu-ram-nvidia-vram-v1";
/// Required response fields consumed by the tuner.
pub const RESPONSE_SCHEMA: &str =
    "choices[0].message.content+timings.prompt_per_second+timings.predicted_per_second";
const MAX_DIAGNOSTIC_CHARS: usize = 2_048;
const CHAT_PROMPT_TARGET_TOKENS: u32 = 512;

/// Measures one candidate config; the tuner is blind to how.
pub trait TrialRunner {
    /// Measure `overrides` for `phase`, returning the trial verdict.
    fn measure(&mut self, overrides: &Overrides, phase: &str) -> Trial;
}

/// What the live runner needs to know about the model under tune.
#[derive(Debug, Clone)]
pub struct TrialTarget {
    pub key: String,
    pub def: ModelDef,
    pub context_key: String,
    pub mode: Mode,
    /// The GGUF path passed as the model argument.
    pub model_arg_path: String,
    /// Samples per trial (medians are reported).
    pub runs: usize,
    /// Port search start for trial servers.
    pub port_start: u16,
    /// Where per-trial server logs land.
    pub log_dir: PathBuf,
    /// LocalBox settings overlay applied under candidate-owned values.
    pub settings_params: LaunchParams,
}

/// The live [`TrialRunner`] over the launcher contract.
pub struct LiveRunner<'a> {
    pub launcher: &'a dyn Launcher,
    pub target: TrialTarget,
    /// Startup wait per trial, seconds.
    pub startup_timeout_secs: u32,
    /// Stable identifier shared by every attempt in this run.
    pub run_id: String,
    next_ordinal: usize,
}

impl<'a> LiveRunner<'a> {
    /// Construct a live runner with a stable run id and zero attempted trials.
    #[must_use]
    pub fn new(
        launcher: &'a dyn Launcher,
        target: TrialTarget,
        startup_timeout_secs: u32,
        run_id: impl Into<String>,
    ) -> Self {
        Self {
            launcher,
            target,
            startup_timeout_secs,
            run_id: run_id.into(),
            next_ordinal: 0,
        }
    }
}

/// Fold candidate-owned launch values over LocalBox settings and the shared
/// single-session defaults.
#[must_use]
pub fn trial_launch_params(candidate: LaunchParams, settings: &LaunchParams) -> LaunchParams {
    let mut request = LaunchRequest::new("measurement", "", Mode::Native);
    request.params = candidate;
    request.apply_session_defaults(settings);
    request.params
}

/// Session-shaping values represented in the run fingerprint. Candidate-owned
/// values remain in the per-entry candidate signature and override this base.
#[must_use]
pub fn session_fingerprint(params: &LaunchParams) -> BTreeMap<String, serde_json::Value> {
    let mut values = BTreeMap::new();
    let mut insert_i64 = |key: &str, value: Option<i64>| {
        if let Some(value) = value {
            values.insert(key.to_string(), value.into());
        }
    };
    insert_i64("parallel", params.parallel);
    insert_i64("cache_reuse", params.cache_reuse);
    insert_i64("n_cpu_moe", params.n_cpu_moe);
    if let Some(value) = params.mlock {
        values.insert("mlock".to_string(), value.into());
    }
    if let Some(value) = params.no_mmap {
        values.insert("no_mmap".to_string(), value.into());
    }
    values
}

#[derive(Debug, Serialize)]
struct ChatMessage<'a> {
    role: &'static str,
    content: &'a str,
}

#[derive(Debug, Serialize)]
struct ChatMeasurementRequest<'a> {
    messages: [ChatMessage<'a>; 1],
    max_tokens: u32,
    temperature: u32,
    top_p: u32,
    seed: u32,
    stream: bool,
    cache_prompt: bool,
}

/// Build the deterministic request body whose shape participates in the cache
/// fingerprint. The full prompt is never written to diagnostics.
#[must_use]
pub fn chat_measurement_request() -> serde_json::Value {
    let prompt = coding_agent_stress_prompt(CHAT_PROMPT_TARGET_TOKENS);
    serde_json::to_value(ChatMeasurementRequest {
        messages: [ChatMessage {
            role: "user",
            content: &prompt,
        }],
        max_tokens: 256,
        temperature: 0,
        top_p: 1,
        seed: 0,
        stream: false,
        cache_prompt: false,
    })
    .unwrap_or(serde_json::Value::Null)
}

/// Stable hash of the exact embedded prompt sent by the live trial path. A
/// fixture or prompt-builder change therefore invalidates old cached trials.
#[must_use]
pub fn chat_prompt_hash() -> String {
    stable_json_hash(&serde_json::json!({
        "target_tokens": CHAT_PROMPT_TARGET_TOKENS,
        "content": coding_agent_stress_prompt(CHAT_PROMPT_TARGET_TOKENS),
    }))
}

/// Stable request-shape hash without persisting the generated prompt.
#[must_use]
pub fn chat_request_shape_hash() -> String {
    stable_json_hash(&serde_json::json!({
        "path": "/v1/chat/completions",
        "messages": [{"role": "user", "content": "stress-prompt-512"}],
        "max_tokens": 256,
        "temperature": 0,
        "top_p": 1,
        "seed": 0,
        "stream": false,
        "cache_prompt": false,
    }))
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ChatContent {
    Text(String),
    Parts(Vec<ChatContentPart>),
}

impl ChatContent {
    fn into_text(self) -> String {
        match self {
            Self::Text(text) => text,
            Self::Parts(parts) => parts
                .into_iter()
                .filter_map(|part| part.text)
                .collect::<Vec<_>>()
                .join(""),
        }
    }
}

#[derive(Debug, Deserialize)]
struct ChatContentPart {
    #[serde(default)]
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatResponseMessage {
    #[serde(default)]
    content: Option<ChatContent>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    #[serde(default)]
    message: Option<ChatResponseMessage>,
}

#[derive(Debug, Deserialize)]
struct ChatTimings {
    #[serde(default)]
    prompt_per_second: Option<serde_json::Value>,
    #[serde(default)]
    predicted_per_second: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct ChatMeasurementResponse {
    #[serde(default)]
    choices: Vec<ChatChoice>,
    #[serde(default)]
    timings: Option<ChatTimings>,
}

/// A decoded, schema-valid chat measurement.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatMeasurement {
    pub pp_tps: f64,
    pub tg_tps: f64,
    pub content: String,
}

fn failure(
    stage: TrialFailureStage,
    reason: TrialFailureReason,
    detail: impl Into<String>,
) -> TrialFailure {
    TrialFailure {
        stage,
        reason,
        detail: sanitize_excerpt(&detail.into()),
    }
}

/// Parse the supported non-streaming OpenAI-compatible chat envelope.
pub fn parse_chat_measurement(body: &str) -> Result<ChatMeasurement, TrialFailure> {
    let value: serde_json::Value = serde_json::from_str(body).map_err(|error| {
        failure(
            TrialFailureStage::Response,
            TrialFailureReason::ResponseDecode,
            error.to_string(),
        )
    })?;
    let response: ChatMeasurementResponse = serde_json::from_value(value).map_err(|error| {
        failure(
            TrialFailureStage::Response,
            TrialFailureReason::ResponseSchema,
            error.to_string(),
        )
    })?;
    let choice = response.choices.into_iter().next().ok_or_else(|| {
        failure(
            TrialFailureStage::Response,
            TrialFailureReason::ResponseSchema,
            "missing choices[0]",
        )
    })?;
    let content = choice
        .message
        .and_then(|message| message.content)
        .ok_or_else(|| {
            failure(
                TrialFailureStage::Response,
                TrialFailureReason::ResponseSchema,
                "missing choices[0].message.content",
            )
        })?
        .into_text();
    let timings = response.timings.ok_or_else(|| {
        failure(
            TrialFailureStage::Response,
            TrialFailureReason::MissingTimings,
            "missing timings",
        )
    })?;
    let read_timing = |name: &str, value: Option<serde_json::Value>| -> Result<f64, TrialFailure> {
        let value = value.ok_or_else(|| {
            failure(
                TrialFailureStage::Response,
                TrialFailureReason::MissingTimings,
                format!("missing timings.{name}"),
            )
        })?;
        let timing = value
            .as_f64()
            .or_else(|| value.as_str().and_then(|text| text.parse::<f64>().ok()))
            .ok_or_else(|| {
                failure(
                    TrialFailureStage::Response,
                    TrialFailureReason::InvalidTimings,
                    format!("timings.{name} is not numeric"),
                )
            })?;
        if !timing.is_finite() || timing <= 0.0 {
            return Err(failure(
                TrialFailureStage::Response,
                TrialFailureReason::InvalidTimings,
                format!("timings.{name} must be finite and greater than zero"),
            ));
        }
        Ok(timing)
    };
    let pp_tps = read_timing("prompt_per_second", timings.prompt_per_second)?;
    let tg_tps = read_timing("predicted_per_second", timings.predicted_per_second)?;
    Ok(ChatMeasurement {
        pp_tps,
        tg_tps,
        content,
    })
}

fn validate_measurement_content(content: &str) -> Result<(), TrialFailure> {
    let smoke = evaluate_smoke_text(content);
    if !smoke.ok {
        let reason = if content.trim().is_empty() {
            TrialFailureReason::EmptyContent
        } else if smoke.visible_text.is_empty() {
            TrialFailureReason::ThinkingOnly
        } else {
            TrialFailureReason::DegenerateContent
        };
        return Err(failure(TrialFailureStage::Content, reason, smoke.error));
    }
    if !output_quality_ok(&smoke.visible_text, QUALITY_MIN_CHARS, QUALITY_MIN_WORDS) {
        return Err(failure(
            TrialFailureStage::Content,
            TrialFailureReason::DegenerateContent,
            "visible response did not meet the tuner quality minimums",
        ));
    }
    Ok(())
}

/// Collapse whitespace, redact likely secrets, and cap diagnostic evidence.
#[must_use]
pub fn sanitize_excerpt(text: &str) -> String {
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let lower = collapsed.to_ascii_lowercase();
    if [
        "authorization",
        "api_key",
        "api-key",
        "password",
        "secret",
        "bearer ",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
    {
        return "[REDACTED]".to_string();
    }
    collapsed.chars().take(MAX_DIAGNOSTIC_CHARS).collect()
}

/// Bound launch arguments and redact values following credential-like flags
/// before they enter manifests, caches, or saved-profile provenance.
#[must_use]
pub fn sanitize_launch_args(args: &[String]) -> Vec<String> {
    let mut redact_next = false;
    args.iter()
        .take(256)
        .map(|arg| {
            if redact_next {
                redact_next = false;
                return "[REDACTED]".to_string();
            }
            let lower = arg.to_ascii_lowercase();
            if lower.starts_with('-')
                && ["api-key", "token", "password", "secret", "authorization"]
                    .iter()
                    .any(|needle| lower.contains(needle))
            {
                if !arg.contains('=') {
                    redact_next = true;
                    return arg.chars().take(128).collect();
                }
                return "[REDACTED]".to_string();
            }
            sanitize_excerpt(arg).chars().take(512).collect()
        })
        .collect()
}

/// Extract bounded, explicitly advisory engine-adjustment observations. Logs
/// are not authoritative configuration, so these strings never populate the
/// observed/effective map or influence ranking.
#[must_use]
pub fn infer_runtime_advisories(text: &str) -> Vec<String> {
    text.lines()
        .filter(|line| {
            let lower = line.to_ascii_lowercase();
            ((lower.contains("upgrade") || lower.contains("adjust"))
                && (lower.contains("kv") || lower.contains("cache") || lower.contains("turbo")))
                || lower.contains("using n_parallel")
        })
        .map(sanitize_excerpt)
        .filter(|line| !line.is_empty())
        .take(8)
        .collect()
}

fn safe_path_component(value: &str) -> String {
    let mut safe: String = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_') {
                ch
            } else {
                '-'
            }
        })
        .take(48)
        .collect();
    while safe.contains("--") {
        safe = safe.replace("--", "-");
    }
    let trimmed = safe.trim_matches('-');
    if trimmed.is_empty() {
        "unknown".to_string()
    } else {
        trimmed.to_string()
    }
}

/// A minimal failed trial used by scripted callers and compatibility tests.
#[must_use]
pub fn failed_trial(oom: bool) -> Trial {
    let reason = if oom {
        TrialFailureReason::ReadinessExitedOom
    } else {
        TrialFailureReason::SpawnFailed
    };
    Trial {
        startup_ok: false,
        oom,
        measurement_usable: false,
        failure: Some(failure(
            if oom {
                TrialFailureStage::Readiness
            } else {
                TrialFailureStage::Launch
            },
            reason,
            "",
        )),
        ..Trial::default()
    }
}

/// A startup failure that records *why* it failed. `startup_ok` stays false and
/// `oom` is set only for an OOM-tagged exit — so scoring gates it out exactly as
/// before — while the payload carries the diagnosis (`crates/.../score.rs`
/// `StartupFailure`) so an exited process and a timed-out one no longer collapse
/// into the same verdict.
#[must_use]
fn startup_failed_trial(
    startup_failure: StartupFailure,
    process_status: Option<i32>,
    diagnostic_excerpt: String,
) -> Trial {
    let (oom, reason) = match startup_failure {
        StartupFailure::ExitedOom => (true, TrialFailureReason::ReadinessExitedOom),
        StartupFailure::Exited => (false, TrialFailureReason::ReadinessExited),
        StartupFailure::TimedOut => (false, TrialFailureReason::ReadinessTimeout),
    };
    Trial {
        startup_ok: false,
        oom,
        startup_failure: Some(startup_failure),
        measurement_usable: false,
        failure: Some(failure(TrialFailureStage::Readiness, reason, "")),
        process_status,
        diagnostic_excerpt,
        ..Trial::default()
    }
}

/// How a spawned trial server finished its startup window.
enum StartupOutcome {
    /// The server answered — proceed to measure it.
    Ready,
    /// The process exited during startup; `oom` reflects its log tail.
    Exited { oom: bool, status: Option<i32> },
    /// The process was still running when the startup budget ran out.
    TimedOut,
}

/// Run-to-run relative variance of the sampled decode speeds.
#[must_use]
pub fn sample_variance(samples: &[f64]) -> Option<f64> {
    if samples.len() < 2 {
        return None;
    }
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    if mean <= 0.0 {
        return None;
    }
    let spread = samples.iter().fold(0.0_f64, |a, s| a.max((s - mean).abs()));
    Some(spread / mean)
}

#[derive(Debug, Clone, Copy, Default)]
struct ResourceSample {
    cpu_pct: Option<f64>,
    ram_available_gb: Option<f64>,
    gpu_vram_free_gb: Option<f64>,
    gpu_vram_total_gb: Option<f64>,
}

#[derive(Default)]
struct TelemetryAccumulator {
    cpu_sum: f64,
    cpu_samples: u32,
    ram_available_gb_min: Option<f64>,
    gpu_vram_free_gb: Vec<f64>,
    gpu_vram_total_gb: Option<f64>,
}

impl TelemetryAccumulator {
    fn record(&mut self, sample: ResourceSample) {
        if let Some(cpu) = sample.cpu_pct.filter(|value| value.is_finite()) {
            self.cpu_sum += cpu.clamp(0.0, 100.0);
            self.cpu_samples = self.cpu_samples.saturating_add(1);
        }
        if let Some(ram) = sample
            .ram_available_gb
            .filter(|value| value.is_finite() && *value >= 0.0)
        {
            self.ram_available_gb_min = Some(
                self.ram_available_gb_min
                    .map_or(ram, |current| current.min(ram)),
            );
        }
        if let Some(free) = sample
            .gpu_vram_free_gb
            .filter(|value| value.is_finite() && *value >= 0.0)
        {
            self.gpu_vram_free_gb.push(free);
        }
        if let Some(total) = sample
            .gpu_vram_total_gb
            .filter(|value| value.is_finite() && *value > 0.0)
        {
            self.gpu_vram_total_gb = Some(
                self.gpu_vram_total_gb
                    .map_or(total, |current| current.min(total)),
            );
        }
    }

    fn finish(self) -> Telemetry {
        let cpu_avg_pct =
            (self.cpu_samples > 0).then(|| self.cpu_sum / f64::from(self.cpu_samples));
        let gpu_vram_free_gb_min = self.gpu_vram_free_gb.iter().copied().reduce(f64::min);
        let gpu_vram_free_gb_samples = (!self.gpu_vram_free_gb.is_empty())
            .then(|| u32::try_from(self.gpu_vram_free_gb.len()).ok())
            .flatten();
        let gpu_vram_free_gb_std = if self.gpu_vram_free_gb.is_empty() {
            None
        } else {
            let mean =
                self.gpu_vram_free_gb.iter().sum::<f64>() / self.gpu_vram_free_gb.len() as f64;
            let variance = self
                .gpu_vram_free_gb
                .iter()
                .map(|value| (value - mean).powi(2))
                .sum::<f64>()
                / self.gpu_vram_free_gb.len() as f64;
            Some(variance.sqrt())
        };
        Telemetry {
            cpu_avg_pct,
            ram_available_gb_min: self.ram_available_gb_min,
            gpu_vram_free_gb_min,
            gpu_vram_free_gb_std,
            gpu_vram_free_gb_samples,
            gpu_vram_total_gb: self.gpu_vram_total_gb,
        }
    }
}

/// How often the sampler wakes while a trial runs.
const TELEMETRY_SAMPLE_INTERVAL: Duration = Duration::from_millis(250);

struct HostResourceProbe {
    system: sysinfo::System,
    cpu_ready: bool,
}

impl HostResourceProbe {
    fn new() -> Self {
        Self {
            // Memory and CPU only. `new_all()` also enumerates every process
            // on the host, which this probe never reads and which costs tens
            // of milliseconds per trial.
            system: sysinfo::System::new(),
            cpu_ready: false,
        }
    }

    fn sample(&mut self) -> ResourceSample {
        self.system.refresh_cpu_usage();
        self.system.refresh_memory();
        // sysinfo needs two refreshes separated by its minimum interval. Do
        // not turn the meaningless first value into an idle-looking sample.
        let cpu_pct = self
            .cpu_ready
            .then(|| f64::from(self.system.global_cpu_usage()));
        self.cpu_ready = true;
        let (gpu_vram_free_gb, gpu_vram_total_gb) =
            probe_nvidia_vram_gb().map_or((None, None), |(free, total)| (Some(free), Some(total)));
        ResourceSample {
            cpu_pct,
            ram_available_gb: Some(self.system.available_memory() as f64 / 1_073_741_824.0),
            gpu_vram_free_gb,
            gpu_vram_total_gb,
        }
    }
}

/// Free and total VRAM across every NVIDIA device, in GB.
///
/// This spawns a process, which costs tens of milliseconds of host CPU inside
/// the window whose throughput is being measured. Sampling it less often is
/// *not* a free trade: `VramHeadroomParams::min_samples` is 5, so a slower
/// cadence would push short trials below the threshold where measured jitter
/// is trusted and silently swap in the conservative fallback sigma. Making
/// this cheaper means making the read cheap (NVML in-process), not rarer.
fn probe_nvidia_vram_gb() -> Option<(f64, f64)> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    let mut free_mib = 0.0;
    let mut total_mib = 0.0;
    let mut devices = 0_u32;
    for line in text.lines() {
        let mut values = line.split(',').map(str::trim);
        let (Some(free), Some(total)) = (values.next(), values.next()) else {
            continue;
        };
        let (Ok(free), Ok(total)) = (free.parse::<f64>(), total.parse::<f64>()) else {
            continue;
        };
        free_mib += free;
        total_mib += total;
        devices = devices.saturating_add(1);
    }
    (devices > 0).then_some((free_mib / 1024.0, total_mib / 1024.0))
}

/// Longest the caller waits for the sampler to finish its first, expensive
/// sample. Generous, because exceeding it only means the warm-up overlaps the
/// measurement again — the behaviour before the warm-up existed.
const TELEMETRY_WARMUP_TIMEOUT: Duration = Duration::from_secs(2);

fn collect_host_telemetry<T>(operation: impl FnOnce() -> T) -> (T, Telemetry) {
    let (stop, stopped) = std::sync::mpsc::channel();
    let (warm, warmed) = std::sync::mpsc::channel();
    let worker = std::thread::Builder::new()
        .name("localbench-telemetry".to_string())
        .spawn(move || {
            let mut probe = HostResourceProbe::new();
            let mut telemetry = TelemetryAccumulator::default();
            // Take the first sample *before* the caller starts measuring. It
            // is the expensive one: sysinfo's first CPU refresh initialises
            // the host counters, and the first VRAM read spawns a process.
            // Neither belongs inside the measured window, and the sample it
            // produces is still recorded rather than thrown away.
            telemetry.record(probe.sample());
            let _ = warm.send(());
            loop {
                match stopped.recv_timeout(TELEMETRY_SAMPLE_INTERVAL) {
                    Ok(()) | Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                        telemetry.record(probe.sample());
                    }
                }
            }
            telemetry.finish()
        });
    let worker = worker.ok();
    if worker.is_some() {
        let _ = warmed.recv_timeout(TELEMETRY_WARMUP_TIMEOUT);
    }
    let result = operation();
    let _ = stop.send(());
    let telemetry = worker
        .and_then(|worker| worker.join().ok())
        .unwrap_or_default();
    (result, telemetry)
}

impl LiveRunner<'_> {
    fn measure_once(&self, port: u16) -> Result<ChatMeasurement, TrialFailure> {
        let url = format!("http://127.0.0.1:{port}/v1/chat/completions");
        let body = chat_measurement_request();
        let runtime = tokio::runtime::Runtime::new().map_err(|error| {
            failure(
                TrialFailureStage::Request,
                TrialFailureReason::Transport,
                error.to_string(),
            )
        })?;
        let raw = runtime.block_on(async {
            let client = reqwest::Client::new();
            let response = client
                .post(&url)
                .json(&body)
                .timeout(Duration::from_secs(1800))
                .send()
                .await
                .map_err(|error| {
                    failure(
                        TrialFailureStage::Request,
                        TrialFailureReason::Transport,
                        error.to_string(),
                    )
                })?;
            let status = response.status();
            let text = response.text().await.map_err(|error| {
                failure(
                    TrialFailureStage::Request,
                    TrialFailureReason::Transport,
                    error.to_string(),
                )
            })?;
            if !status.is_success() {
                return Err(failure(
                    TrialFailureStage::Response,
                    TrialFailureReason::HttpStatus,
                    format!("HTTP {status}: {}", sanitize_excerpt(&text)),
                ));
            }
            Ok(text)
        })?;
        parse_chat_measurement(&raw)
    }

    fn observed_runtime(&self, port: u16) -> BTreeMap<String, serde_json::Value> {
        let Ok(runtime) = tokio::runtime::Runtime::new() else {
            return BTreeMap::new();
        };
        runtime.block_on(async {
            let url = format!("http://127.0.0.1:{port}/props");
            let Ok(response) = reqwest::Client::new()
                .get(url)
                .timeout(Duration::from_secs(5))
                .send()
                .await
            else {
                return BTreeMap::new();
            };
            let Ok(value) = response.json::<serde_json::Value>().await else {
                return BTreeMap::new();
            };
            let mut observed = BTreeMap::new();
            for field in ["total_slots", "build_info"] {
                if let Some(value) = value.get(field) {
                    let bounded = value
                        .as_str()
                        .map(|text| serde_json::Value::from(sanitize_excerpt(text)))
                        .unwrap_or_else(|| value.clone());
                    observed.insert(field.to_string(), bounded);
                }
            }
            if let Some(template) = value.get("chat_template") {
                observed.insert(
                    "chat_template_hash".to_string(),
                    stable_json_hash(template).into(),
                );
            }
            observed
        })
    }

    fn log_tail(&self, log: &Path) -> String {
        std::fs::read_to_string(log)
            .map(|s| {
                let lines: Vec<&str> = s.lines().rev().take(80).collect();
                lines.into_iter().rev().collect::<Vec<_>>().join("\n")
            })
            .unwrap_or_default()
    }

    /// Wait for the spawned server to answer while watching the child the whole
    /// time. A process that *exits* is an unambiguous answer classified from its
    /// log the moment it happens — instead of polling a closed port to the
    /// startup ceiling — while a still-loading server keeps the full budget.
    fn await_ready(&self, port: u16, child: &mut Child, log: &Path) -> StartupOutcome {
        // One-second slices keep the process check running about once a second
        // without pushing a `Child` across the launcher's readiness contract.
        const SLICE_SECS: u32 = 1;
        let deadline = Instant::now() + Duration::from_secs(u64::from(self.startup_timeout_secs));
        loop {
            if let Ok(Some(status)) = child.try_wait() {
                return StartupOutcome::Exited {
                    oom: is_oom_message(&self.log_tail(log)),
                    status: status.code(),
                };
            }
            let now = Instant::now();
            if now >= deadline {
                return StartupOutcome::TimedOut;
            }
            // Re-derive the slice from the remaining budget so the final wait
            // cannot overrun the deadline by more than one slice.
            let slice = SLICE_SECS.min(
                u32::try_from(deadline.saturating_duration_since(now).as_secs())
                    .unwrap_or(SLICE_SECS)
                    .max(1),
            );
            if self.launcher.wait_server(port, slice).is_ok() {
                return StartupOutcome::Ready;
            }
        }
    }

    /// Everything after the server is spawned: wait for it while watching the
    /// process, then either measure a ready server or record why startup failed.
    /// `measure` delegates here exclusively, so the process-aware waiting and the
    /// outcome-to-`Trial` mapping are exercised end-to-end without a live model.
    fn measure_spawned(&self, child: &mut Child, port: u16, log: &Path) -> Trial {
        match self.await_ready(port, child, log) {
            StartupOutcome::Ready => self.sample(port, child, log),
            StartupOutcome::Exited { oom, status } => {
                // Already exited — reaping is a no-op wait, never an error.
                let _ = child.wait();
                startup_failed_trial(
                    if oom {
                        StartupFailure::ExitedOom
                    } else {
                        StartupFailure::Exited
                    },
                    status,
                    sanitize_excerpt(&self.log_tail(log)),
                )
            }
            StartupOutcome::TimedOut => {
                let _ = child.kill();
                let status = child.wait().ok().and_then(|status| status.code());
                eprintln!(
                    "trial server on port {port} never became ready within {}s and was still \
                     running; raise --startup-timeout if this machine loads models slowly",
                    self.startup_timeout_secs
                );
                startup_failed_trial(
                    StartupFailure::TimedOut,
                    status,
                    sanitize_excerpt(&self.log_tail(log)),
                )
            }
        }
    }

    /// Sample a ready server `runs` times and reduce to a rankable trial, or a
    /// failed trial when the server errors or only produces degenerate text.
    fn sample(&self, port: u16, child: &mut Child, log: &Path) -> Trial {
        let observed_configuration = self.observed_runtime(port);
        let (mut trial, telemetry) = collect_host_telemetry(|| {
            let mut pp = Vec::new();
            let mut tg = Vec::new();
            for _ in 0..self.target.runs.max(1) {
                match self.measure_once(port) {
                    Ok(measurement) => {
                        if let Err(content_failure) =
                            validate_measurement_content(&measurement.content)
                        {
                            let _ = child.kill();
                            let status = child.wait().ok().and_then(|status| status.code());
                            return Trial {
                                startup_ok: true,
                                oom: false,
                                measurement_usable: false,
                                failure: Some(content_failure),
                                process_status: status,
                                diagnostic_excerpt: sanitize_excerpt(&self.log_tail(log)),
                                observed_configuration,
                                ..Trial::default()
                            };
                        }
                        pp.push(measurement.pp_tps);
                        tg.push(measurement.tg_tps);
                    }
                    Err(measurement_failure) => {
                        let log_tail = self.log_tail(log);
                        // A response body merely mentioning OOM is not fit evidence.
                        // The candidate has its own unique engine log, so only an
                        // actual engine OOM signature may enter recovery.
                        let oom = is_oom_message(&log_tail);
                        let _ = child.kill();
                        let status = child.wait().ok().and_then(|status| status.code());
                        return Trial {
                            startup_ok: true,
                            oom,
                            measurement_usable: false,
                            failure: Some(measurement_failure),
                            process_status: status,
                            diagnostic_excerpt: sanitize_excerpt(&log_tail),
                            observed_configuration,
                            ..Trial::default()
                        };
                    }
                }
            }
            let _ = child.kill();
            let status = child.wait().ok().and_then(|status| status.code());
            Trial {
                startup_ok: true,
                oom: false,
                measurement_usable: true,
                pp_tps: median(&pp),
                tg_tps: median(&tg),
                variance: sample_variance(&tg),
                process_status: status,
                diagnostic_excerpt: sanitize_excerpt(&self.log_tail(log)),
                observed_configuration,
                ..Trial::default()
            }
        });
        trial.telemetry = telemetry;
        trial
    }
}

impl TrialRunner for LiveRunner<'_> {
    fn measure(&mut self, overrides: &Overrides, phase: &str) -> Trial {
        self.next_ordinal += 1;
        let ordinal = self.next_ordinal;
        let target = &self.target;
        let signature_hash =
            stable_json_hash(&serde_json::to_value(overrides).unwrap_or(serde_json::Value::Null));
        let attempt_id = format!(
            "{}-{ordinal:04}-{signature_hash}",
            safe_path_component(&self.run_id)
        );
        let diagnostic = |log_path: Option<&Path>| TrialDiagnosticRef {
            attempt_id: attempt_id.clone(),
            manifest_path: String::new(),
            log_path: log_path
                .map(|path| path.display().to_string())
                .unwrap_or_default(),
        };
        let launch_failed = |reason: TrialFailureReason, detail: String| Trial {
            startup_ok: false,
            oom: false,
            measurement_usable: false,
            failure: Some(failure(TrialFailureStage::Launch, reason, detail)),
            diagnostic: Some(diagnostic(None)),
            ..Trial::default()
        };
        // The map's PascalCase keys are the typed schema's wire spelling.
        let typed: localx_llama_core::tuner::Overrides = match serde_json::from_value(
            serde_json::Value::Object(overrides.clone().into_iter().collect()),
        ) {
            Ok(typed) => typed,
            Err(error) => {
                return launch_failed(TrialFailureReason::InvalidOverrides, error.to_string())
            }
        };
        let params = trial_launch_params(typed.to_launch_params(), &target.settings_params);
        let port = match self.launcher.free_port(target.port_start) {
            Ok(port) => port,
            Err(error) => {
                return launch_failed(TrialFailureReason::PortUnavailable, error.to_string())
            }
        };
        let argv = match build_llama_server_args(
            &target.def,
            &target.context_key,
            target.mode,
            &target.model_arg_path,
            i64::from(port),
            &params,
        ) {
            Ok(argv) => argv,
            Err(error) => {
                return launch_failed(TrialFailureReason::ArgumentConstruction, error.to_string())
            }
        };
        let binary = match self.launcher.server_binary(target.mode, true) {
            Ok(binary) => binary,
            Err(error) => {
                return launch_failed(TrialFailureReason::BinaryUnavailable, error.to_string())
            }
        };
        if let Err(error) = std::fs::create_dir_all(&target.log_dir) {
            return launch_failed(TrialFailureReason::SpawnFailed, error.to_string());
        }
        let log = target.log_dir.join(format!(
            "trial-{}-{ordinal:04}-{}-{signature_hash}-{port}.log",
            safe_path_component(&self.run_id),
            safe_path_component(phase),
        ));
        let mut child = match spawn_detached(
            &binary.to_string_lossy(),
            &argv,
            binary.parent(),
            Some(&log),
        ) {
            Ok(child) => child,
            Err(error) => {
                let mut trial = launch_failed(TrialFailureReason::SpawnFailed, error.to_string());
                trial.diagnostic = Some(diagnostic(Some(&log)));
                trial.launch_args = sanitize_launch_args(&argv);
                return trial;
            }
        };
        let mut trial = self.measure_spawned(&mut child, port, &log);
        trial.diagnostic = Some(diagnostic(Some(&log)));
        trial.launch_args = sanitize_launch_args(&argv);
        trial.advisory_observations = infer_runtime_advisories(&self.log_tail(&log));
        trial
    }
}

/// A [`TrialRunner`] wrapper that consults the persistent trial cache before
/// measuring and records decisive results after: a cache hit skips the whole
/// server spawn; a plain startup failure is never cached (transient), so it
/// is measured fresh next time; ineligible phases (verify, guard, soak,
/// recovery) always measure live. Each stored trial is persisted crash-safely
/// immediately, so an interrupted run keeps everything it already measured.
pub struct CachedRunner<'a> {
    pub inner: &'a mut dyn TrialRunner,
    pub cache: localbench_measure::cache::TrialCache,
    pub driver: localbench_measure::cache::Driver,
    /// Produces the `measured_at` stamp for stored entries.
    pub stamp: fn() -> String,
    /// Per-run evidence ledger. Cache semantics remain separate from it.
    pub ledger: Option<crate::diagnostics::RunLedger>,
    /// Last returned outcome, used to keep terminal CLI errors typed.
    pub last_trial: Option<Trial>,
}

impl CachedRunner<'_> {
    /// Exact manifest path shown in CLI summaries.
    #[must_use]
    pub fn manifest_path(&self) -> Option<&Path> {
        self.ledger
            .as_ref()
            .map(crate::diagnostics::RunLedger::path)
    }

    /// Mark the diagnostic run complete so retention may consider it later.
    pub fn finish(&self) -> std::io::Result<()> {
        self.ledger
            .as_ref()
            .map_or(Ok(()), crate::diagnostics::RunLedger::finish)
    }

    /// Typed summary of the final failed attempt, when there was one.
    #[must_use]
    pub fn last_failure_summary(&self) -> Option<String> {
        let trial = self.last_trial.as_ref()?;
        let failure = trial.failure.as_ref()?;
        let mut summary = failure.summary();
        if let Some(diagnostic) = &trial.diagnostic {
            let evidence = if diagnostic.log_path.is_empty() {
                &diagnostic.manifest_path
            } else {
                &diagnostic.log_path
            };
            if !evidence.is_empty() {
                summary.push_str(&format!(" (evidence: {evidence})"));
            }
        }
        Some(summary)
    }

    fn record_attempt(
        &mut self,
        overrides: &Overrides,
        phase: &str,
        source: &str,
        trial: &mut Trial,
        stamp: &str,
    ) {
        let Some(ledger) = self.ledger.as_mut() else {
            return;
        };
        if let Err(error) = ledger.record(
            phase,
            &candidate_signature(overrides),
            source,
            overrides,
            trial,
            stamp,
        ) {
            eprintln!(
                "trial diagnostic manifest could not record {}: {error}",
                ledger.path().display()
            );
        }
    }
}

impl TrialRunner for CachedRunner<'_> {
    fn measure(&mut self, overrides: &Overrides, phase: &str) -> Trial {
        if let Some(cached) = self.cache.get(self.driver, overrides, phase) {
            if let Ok(mut trial) = serde_json::from_value::<Trial>(cached) {
                // Run-scoped evidence belongs to the original live attempt, not
                // this cache hit. The new ledger line receives its own identity.
                trial.diagnostic = None;
                trial.diagnostic_excerpt.clear();
                trial.process_status = None;
                let stamp = (self.stamp)();
                self.record_attempt(overrides, phase, "cache", &mut trial, &stamp);
                self.last_trial = Some(trial.clone());
                return trial;
            }
        }
        let mut trial = self.inner.measure(overrides, phase);
        let stamp = (self.stamp)();
        let mut reusable_trial = trial.clone();
        reusable_trial.diagnostic = None;
        reusable_trial.diagnostic_excerpt.clear();
        reusable_trial.process_status = None;
        // Evidence is the first durable write for an attempt. A cache-write
        // failure or interruption must not erase the fact that it ran.
        self.record_attempt(overrides, phase, "live", &mut trial, &stamp);
        let stored = self.cache.put(&localbench_measure::cache::MeasuredTrial {
            driver: self.driver,
            overrides: overrides.clone(),
            phase: phase.to_string(),
            oom: trial.oom,
            startup_ok: trial.startup_ok,
            measurement_usable: trial.is_measurement_usable(),
            measured_at: stamp.clone(),
            trial: serde_json::to_value(&reusable_trial).unwrap_or(serde_json::Value::Null),
        });
        if stored {
            if let Err(error) = self.cache.save(&stamp) {
                eprintln!(
                    "trial cache could not be saved ({}): {error}",
                    self.cache.path().display()
                );
            }
        }
        self.last_trial = Some(trial.clone());
        trial
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use localbench_measure::cache::{Driver, Fingerprint, GgufIdentity, TrialCache};
    use localbench_scoring::score::BALANCED_TELEMETRY_FIELDS;

    #[test]
    fn chat_timings_parse_string_and_content_parts() {
        let body = r#"{"choices":[{"message":{"role":"assistant","content":"a fine long answer indeed"}}],"timings":{"prompt_per_second":123.4,"predicted_per_second":45.6},"usage":{"total_tokens":99}}"#;
        let parsed = parse_chat_measurement(body).unwrap();
        assert!((parsed.pp_tps - 123.4).abs() < 1e-9);
        assert!((parsed.tg_tps - 45.6).abs() < 1e-9);
        assert_eq!(parsed.content, "a fine long answer indeed");

        let parts = r#"{"choices":[{"message":{"content":[{"type":"text","text":"hello "},{"type":"text","text":"world"}]}}],"timings":{"prompt_per_second":1.0,"predicted_per_second":2.0}}"#;
        assert_eq!(
            parse_chat_measurement(parts).unwrap().content,
            "hello world"
        );
    }

    #[test]
    fn chat_schema_and_timing_failures_are_typed() {
        let cases = [
            ("not json", TrialFailureReason::ResponseDecode),
            ("{}", TrialFailureReason::ResponseSchema),
            (
                r#"{"choices":[{"message":{"content":"ok"}}]}"#,
                TrialFailureReason::MissingTimings,
            ),
            (
                r#"{"choices":[{"message":{"content":"ok"}}],"timings":{"prompt_per_second":0.0,"predicted_per_second":2.0}}"#,
                TrialFailureReason::InvalidTimings,
            ),
            (
                r#"{"choices":[{"message":{"content":"ok"}}],"timings":{"predicted_per_second":2.0}}"#,
                TrialFailureReason::MissingTimings,
            ),
            (
                r#"{"choices":[{"message":{"content":"ok"}}],"timings":{"prompt_per_second":"NaN","predicted_per_second":2.0}}"#,
                TrialFailureReason::InvalidTimings,
            ),
            (
                r#"{"choices":[{"message":{"content":42}}],"timings":{"prompt_per_second":1.0,"predicted_per_second":2.0}}"#,
                TrialFailureReason::ResponseSchema,
            ),
        ];
        for (body, expected) in cases {
            assert_eq!(parse_chat_measurement(body).unwrap_err().reason, expected);
        }
    }

    #[test]
    fn visible_content_failures_remain_distinct() {
        let healthy = "The configuration completed every requested check with stable throughput and enough visible words to satisfy the strict quality gate while preserving deterministic behavior across repeated samples in this isolated measurement.";
        let cases = [
            ("", Some(TrialFailureReason::EmptyContent)),
            (
                "<think>private reasoning only</think>",
                Some(TrialFailureReason::ThinkingOnly),
            ),
            (
                "loop loop loop loop loop loop loop loop loop loop",
                Some(TrialFailureReason::DegenerateContent),
            ),
            (
                "A visible answer exists but it is deliberately too short.",
                Some(TrialFailureReason::DegenerateContent),
            ),
            (healthy, None),
        ];
        for (content, expected) in cases {
            assert_eq!(
                validate_measurement_content(content)
                    .err()
                    .map(|failure| failure.reason),
                expected,
                "content={content:?}"
            );
        }
    }

    #[test]
    fn live_quality_gate_tracks_the_shared_defaults() {
        fn distinct_words_padded_to(word_count: usize, target_len: usize) -> String {
            let mut words = (0..word_count)
                .map(|index| format!("w{index}"))
                .collect::<Vec<_>>();
            let current_len = words.iter().map(String::len).sum::<usize>() + word_count - 1;
            assert!(current_len <= target_len);
            words
                .last_mut()
                .unwrap()
                .push_str(&"x".repeat(target_len - current_len));
            words.join(" ")
        }

        assert_eq!((QUALITY_MIN_CHARS, QUALITY_MIN_WORDS), (80, 20));

        let too_few_words = distinct_words_padded_to(QUALITY_MIN_WORDS - 1, QUALITY_MIN_CHARS);
        assert_eq!(
            validate_measurement_content(&too_few_words)
                .unwrap_err()
                .reason,
            TrialFailureReason::DegenerateContent
        );

        let too_few_chars = distinct_words_padded_to(QUALITY_MIN_WORDS, QUALITY_MIN_CHARS - 1);
        assert_eq!(
            validate_measurement_content(&too_few_chars)
                .unwrap_err()
                .reason,
            TrialFailureReason::DegenerateContent
        );

        let accepted = distinct_words_padded_to(QUALITY_MIN_WORDS, QUALITY_MIN_CHARS);
        assert!(validate_measurement_content(&accepted).is_ok());
    }

    #[test]
    fn runtime_adjustments_are_advisory_and_bounded() {
        let log = "engine: upgraded turbo KV cache from turbo2 to turbo3\n\
                   srv: n_parallel is set to auto, using n_parallel = 4\n\
                   ordinary startup line";
        let observations = infer_runtime_advisories(log);
        assert_eq!(observations.len(), 2);
        assert!(observations[0].contains("upgraded turbo KV"));
        assert!(observations[1].contains("using n_parallel"));
    }

    #[test]
    fn hostile_long_run_and_phase_names_become_portable_components() {
        let input = format!("../../bad:phase*?{}", "x".repeat(200));
        let safe = safe_path_component(&input);
        assert!(safe.len() <= 48);
        assert!(safe
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_')));
        assert!(!safe.contains(".."));
        assert!(!safe.contains('/') && !safe.contains('\\'));
    }

    #[test]
    fn chat_request_has_no_raw_completion_fallback() {
        let request = chat_measurement_request();
        assert!(request.get("messages").is_some());
        assert_eq!(request["max_tokens"], 256);
        assert_eq!(request["temperature"], 0);
        assert!(request.get("prompt").is_none());
        assert!(request.get("n_predict").is_none());
        let content = request["messages"][0]["content"].as_str().unwrap();
        assert!(content.contains("Support reliable llama.cpp benchmarking for a local MoE model"));
        let repeated_key_prompt = localbench_measure::prompt::stress_prompt(
            CHAT_PROMPT_TARGET_TOKENS,
            "cache-key-marker",
        );
        assert_ne!(content, repeated_key_prompt);
        assert!(!content.contains("cache-key-marker"));
    }

    #[test]
    fn prompt_fingerprint_hashes_the_exact_shipped_content() {
        let request = chat_measurement_request();
        assert_eq!(
            chat_prompt_hash(),
            stable_json_hash(&serde_json::json!({
                "target_tokens": CHAT_PROMPT_TARGET_TOKENS,
                "content": request["messages"][0]["content"],
            }))
        );
        let legacy_key_hash = stable_json_hash(&serde_json::json!({
            "target_tokens": CHAT_PROMPT_TARGET_TOKENS,
            "seed": "model-key",
        }));
        assert_ne!(chat_prompt_hash(), legacy_key_hash);
    }

    #[test]
    fn session_defaults_and_settings_are_byte_exact_in_argv() {
        let mut def = ModelDef {
            repo: "owner/model".to_string(),
            ..ModelDef::default()
        };
        def.contexts.insert(String::new(), 65_536);
        let defaults = trial_launch_params(LaunchParams::default(), &LaunchParams::default());
        let argv =
            build_llama_server_args(&def, "", Mode::Native, "/models/m.gguf", 8080, &defaults)
                .unwrap();
        assert_eq!(
            argv,
            [
                "-m",
                "/models/m.gguf",
                "-c",
                "65536",
                "--host",
                "127.0.0.1",
                "--port",
                "8080",
                "--parallel",
                "1",
                "--cache-reuse",
                "256",
                "-ngl",
                "999",
                "--cache-type-k",
                "q8_0",
                "--cache-type-v",
                "q8_0",
                "--reasoning",
                "off",
                "--reasoning-budget",
                "0",
                "--reasoning-format",
                "none",
            ]
        );

        let candidate = LaunchParams {
            parallel: Some(2),
            ..LaunchParams::default()
        };
        let settings = LaunchParams {
            parallel: Some(4),
            cache_reuse: Some(512),
            mlock: Some(true),
            ..LaunchParams::default()
        };
        let effective = trial_launch_params(candidate, &settings);
        assert_eq!(effective.parallel, Some(2));
        assert_eq!(effective.cache_reuse, Some(512));
        assert_eq!(effective.mlock, Some(true));
        let argv =
            build_llama_server_args(&def, "", Mode::Native, "/models/m.gguf", 8080, &effective)
                .unwrap();
        assert_eq!(
            argv,
            [
                "-m",
                "/models/m.gguf",
                "-c",
                "65536",
                "--host",
                "127.0.0.1",
                "--port",
                "8080",
                "--parallel",
                "2",
                "--cache-reuse",
                "512",
                "-ngl",
                "999",
                "--mlock",
                "--cache-type-k",
                "q8_0",
                "--cache-type-v",
                "q8_0",
                "--reasoning",
                "off",
                "--reasoning-budget",
                "0",
                "--reasoning-format",
                "none",
            ]
        );
    }

    #[test]
    fn loopback_raw_empty_chat_valid_is_usable_only_via_chat() {
        use std::io::{Read as _, Write as _};
        use std::sync::{Arc, Mutex};

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let captured = Arc::new(Mutex::new(Vec::<(String, String)>::new()));
        let server_captured = Arc::clone(&captured);
        let server = std::thread::spawn(move || {
            for _ in 0..3 {
                let (mut stream, _) = listener.accept().unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let mut bytes = Vec::new();
                let mut chunk = [0_u8; 4096];
                loop {
                    let count = stream.read(&mut chunk).unwrap_or(0);
                    if count == 0 {
                        break;
                    }
                    bytes.extend_from_slice(&chunk[..count]);
                    let Some(header_end) = bytes.windows(4).position(|w| w == b"\r\n\r\n") else {
                        continue;
                    };
                    let headers = String::from_utf8_lossy(&bytes[..header_end]);
                    let content_length = headers
                        .lines()
                        .find_map(|line| {
                            line.strip_prefix("content-length: ")
                                .or_else(|| line.strip_prefix("Content-Length: "))
                        })
                        .and_then(|value| value.trim().parse::<usize>().ok())
                        .unwrap_or(0);
                    if bytes.len() >= header_end + 4 + content_length {
                        break;
                    }
                }
                let header_end = bytes.windows(4).position(|w| w == b"\r\n\r\n").unwrap();
                let headers = String::from_utf8_lossy(&bytes[..header_end]);
                let path = headers
                    .lines()
                    .next()
                    .and_then(|line| line.split_whitespace().nth(1))
                    .unwrap_or("")
                    .to_string();
                let body = String::from_utf8_lossy(&bytes[header_end + 4..]).to_string();
                server_captured.lock().unwrap().push((path.clone(), body));
                let reply = if path == "/completion" {
                    r#"{"content":"","timings":{"prompt_per_second":100.0,"predicted_per_second":50.0}}"#.to_string()
                } else if path == "/props" {
                    r#"{"total_slots":1,"build_info":"loopback"}"#.to_string()
                } else {
                    serde_json::json!({
                        "id": "chatcmpl-test",
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "The configuration completed every requested check with stable throughput and enough visible words to satisfy the strict quality gate while preserving deterministic behavior across repeated samples in this isolated loopback measurement."
                            },
                            "finish_reason": "stop"
                        }],
                        "timings": {
                            "prompt_per_second": 100.0,
                            "predicted_per_second": 50.0
                        },
                        "usage": {"prompt_tokens": 512, "completion_tokens": 30}
                    })
                    .to_string()
                };
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    reply.len(),
                    reply
                );
                stream.write_all(response.as_bytes()).unwrap();
            }
        });

        let runtime = tokio::runtime::Runtime::new().unwrap();
        let raw: serde_json::Value = runtime.block_on(async {
            reqwest::Client::new()
                .post(format!("http://127.0.0.1:{port}/completion"))
                .json(&serde_json::json!({"prompt": "raw"}))
                .send()
                .await
                .unwrap()
                .json()
                .await
                .unwrap()
        });
        assert_eq!(raw["content"], "");

        let dir = tempfile::tempdir().unwrap();
        let live = live_runner(&NeverReadyLauncher, dir.path().to_path_buf(), 1);
        let (program, args) = long_lived_command();
        let log = dir.path().join("loopback.log");
        let mut child = spawn_detached(program, &args, None, Some(&log)).unwrap();
        let trial = live.sample(port, &mut child, &log);
        server.join().unwrap();

        assert!(trial.is_measurement_usable());
        assert!(trial.telemetry.ram_available_gb_min.is_some());
        assert_ne!(trial.telemetry, Telemetry::default());
        assert_eq!(trial.observed_configuration["total_slots"], 1);
        let requests = captured.lock().unwrap();
        assert_eq!(requests[0].0, "/completion");
        assert_eq!(requests[1].0, "/props");
        assert_eq!(requests[2].0, "/v1/chat/completions");
        let chat: serde_json::Value = serde_json::from_str(&requests[2].1).unwrap();
        assert_eq!(chat["messages"][0]["role"], "user");
        assert_eq!(chat["max_tokens"], 256);
        assert!(chat.get("prompt").is_none());
    }

    #[test]
    fn loopback_http_and_decode_failures_keep_their_stage() {
        use std::io::{Read as _, Write as _};

        let serve = |status: &'static str, body: &'static str| {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let port = listener.local_addr().unwrap().port();
            let thread = std::thread::spawn(move || {
                let (mut stream, _) = listener.accept().unwrap();
                let mut request = [0_u8; 16_384];
                let _ = stream.read(&mut request);
                let response = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                stream.write_all(response.as_bytes()).unwrap();
            });
            (port, thread)
        };

        let dir = tempfile::tempdir().unwrap();
        let live = live_runner(&NeverReadyLauncher, dir.path().to_path_buf(), 1);

        let (port, thread) = serve("503 Service Unavailable", "api-key: secret-value");
        let failure = live.measure_once(port).unwrap_err();
        thread.join().unwrap();
        assert_eq!(failure.stage, TrialFailureStage::Response);
        assert_eq!(failure.reason, TrialFailureReason::HttpStatus);
        assert!(!failure.detail.contains("secret-value"));

        let (port, thread) = serve("200 OK", "{broken");
        let failure = live.measure_once(port).unwrap_err();
        thread.join().unwrap();
        assert_eq!(failure.stage, TrialFailureStage::Response);
        assert_eq!(failure.reason, TrialFailureReason::ResponseDecode);
    }

    #[test]
    fn failed_trials_are_never_rankable() {
        let t = failed_trial(true);
        assert!(!t.startup_ok);
        assert!(t.oom);
        assert_eq!(t.pp_tps, 0.0);
    }

    #[test]
    fn sample_variance_is_relative_spread() {
        assert_eq!(sample_variance(&[50.0]), None);
        let v = sample_variance(&[45.0, 50.0, 55.0]).unwrap();
        assert!((v - 0.1).abs() < 1e-9);
    }

    /// A VRAM read that fails mid-trial must leave the readings around it
    /// alone instead of erasing them.
    #[test]
    fn a_tick_without_a_vram_reading_keeps_the_readings_around_it() {
        let mut accumulator = TelemetryAccumulator::default();
        accumulator.record(ResourceSample {
            cpu_pct: Some(50.0),
            ram_available_gb: Some(16.0),
            gpu_vram_free_gb: Some(3.0),
            gpu_vram_total_gb: Some(24.0),
        });
        accumulator.record(ResourceSample {
            cpu_pct: Some(60.0),
            ram_available_gb: Some(15.0),
            gpu_vram_free_gb: None,
            gpu_vram_total_gb: None,
        });
        let telemetry = accumulator.finish();
        assert_eq!(telemetry.gpu_vram_free_gb_min, Some(3.0));
        assert_eq!(telemetry.gpu_vram_total_gb, Some(24.0));
        assert_eq!(telemetry.gpu_vram_free_gb_samples, Some(1));
        assert_eq!(telemetry.cpu_avg_pct, Some(55.0));
        assert_eq!(telemetry.ram_available_gb_min, Some(15.0));
    }

    /// The first sample is taken before the measured operation starts, so an
    /// operation that returns immediately still carries host telemetry — and
    /// the expensive first CPU refresh and VRAM spawn stay out of the window.
    #[test]
    fn the_first_sample_is_taken_before_the_measured_operation() {
        let (result, telemetry) = collect_host_telemetry(|| 7);
        assert_eq!(result, 7);
        assert!(
            telemetry.ram_available_gb_min.is_some(),
            "a warmed sampler reports the sample it already took: {telemetry:?}"
        );
    }

    #[test]
    fn telemetry_accumulator_produces_every_balanced_score_input() {
        let mut accumulator = TelemetryAccumulator::default();
        accumulator.record(ResourceSample {
            cpu_pct: Some(80.0),
            ram_available_gb: Some(12.0),
            gpu_vram_free_gb: Some(2.0),
            gpu_vram_total_gb: Some(24.0),
        });
        accumulator.record(ResourceSample {
            cpu_pct: Some(100.0),
            ram_available_gb: Some(8.0),
            gpu_vram_free_gb: Some(1.0),
            gpu_vram_total_gb: Some(24.0),
        });
        let telemetry = accumulator.finish();
        assert_eq!(BALANCED_TELEMETRY_FIELDS.len(), 6);
        assert!(telemetry.missing_balanced_fields().is_empty());
        assert_eq!(telemetry.cpu_avg_pct, Some(90.0));
        assert_eq!(telemetry.ram_available_gb_min, Some(8.0));
        assert_eq!(telemetry.gpu_vram_free_gb_min, Some(1.0));
        assert_eq!(telemetry.gpu_vram_free_gb_samples, Some(2));
        assert_eq!(telemetry.gpu_vram_total_gb, Some(24.0));
    }

    struct ScriptedRunner {
        calls: usize,
        next: Trial,
    }

    impl TrialRunner for ScriptedRunner {
        fn measure(&mut self, _overrides: &Overrides, _phase: &str) -> Trial {
            self.calls += 1;
            self.next.clone()
        }
    }

    fn fingerprint() -> Fingerprint {
        Fingerprint {
            schema: 1,
            tuner_version: "test".to_string(),
            measurement_protocol: "chat".to_string(),
            telemetry_protocol: "telemetry".to_string(),
            request_shape_hash: "request".to_string(),
            chat_template_hash: "template".to_string(),
            model_definition_hash: "model-definition".to_string(),
            response_schema: "response".to_string(),
            session: BTreeMap::new(),
            key: "model-x".to_string(),
            context_key: "64k".to_string(),
            context_tokens: 65_536,
            mode: "native".to_string(),
            quant: "q".to_string(),
            prompt_length: "short".to_string(),
            prompt_hash: "p".to_string(),
            optimize: "coding-agent".to_string(),
            profile: "pure".to_string(),
            search_strategy: "findbest".to_string(),
            beam_width: 0,
            runs: 3,
            vram_gb: 24,
            gpu_names: vec![],
            llamacpp_build: String::new(),
            gguf: GgufIdentity::default(),
            allowed_kv_types: vec![],
            stress_targets: vec![],
            stress_min_free_vram_gb: 0.0,
            skip_mtp: false,
            skip_stress_test: false,
        }
    }

    fn measured() -> Trial {
        Trial {
            startup_ok: true,
            oom: false,
            measurement_usable: true,
            pp_tps: 100.0,
            tg_tps: 50.0,
            ..Trial::default()
        }
    }

    fn stamp() -> String {
        "2026-01-01T00:00:00Z".to_string()
    }

    #[test]
    fn a_cache_hit_skips_the_live_measurement_and_survives_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trial-cache-model-x-64k.json");
        let overrides = localbench_search::overrides::overrides_of(&[("NCpuMoe", 20.into())]);

        let mut inner = ScriptedRunner {
            calls: 0,
            next: measured(),
        };
        {
            let (cache, _) = TrialCache::open(&path, &fingerprint(), true);
            let mut runner = CachedRunner {
                inner: &mut inner,
                cache,
                driver: Driver::Server,
                stamp,
                ledger: None,
                last_trial: None,
            };
            let first = runner.measure(&overrides, "baseline");
            assert_eq!(first.tg_tps, 50.0);
            let second = runner.measure(&overrides, "batching");
            assert_eq!(second.tg_tps, 50.0);
        }
        assert_eq!(inner.calls, 1, "the second measurement came from the cache");

        // A fresh process with the same fingerprint reuses the persisted entry.
        let (cache, outcome) = TrialCache::open(&path, &fingerprint(), true);
        assert!(matches!(
            outcome,
            localbench_measure::cache::LoadOutcome::Loaded(1)
        ));
        let mut inner = ScriptedRunner {
            calls: 0,
            next: measured(),
        };
        let mut runner = CachedRunner {
            inner: &mut inner,
            cache,
            driver: Driver::Server,
            stamp,
            ledger: None,
            last_trial: None,
        };
        let cached = runner.measure(&overrides, "baseline");
        assert_eq!(cached.tg_tps, 50.0);
        assert_eq!(inner.calls, 0);
    }

    #[test]
    fn live_and_cached_attempts_receive_distinct_manifest_records() {
        let dir = tempfile::tempdir().unwrap();
        let cache_path = dir.path().join("trial-cache.json");
        let overrides = localbench_search::overrides::overrides_of(&[("NCpuMoe", 20.into())]);
        let mut inner = ScriptedRunner {
            calls: 0,
            next: measured(),
        };
        let (cache, _) = TrialCache::open(&cache_path, &fingerprint(), true);
        let ledger = crate::diagnostics::RunLedger::create(dir.path(), "cache-run", 5).unwrap();
        let manifest = ledger.path().to_path_buf();
        let mut runner = CachedRunner {
            inner: &mut inner,
            cache,
            driver: Driver::Server,
            stamp,
            ledger: Some(ledger),
            last_trial: None,
        };

        let live = runner.measure(&overrides, "baseline");
        let cached = runner.measure(&overrides, "batching");
        runner.finish().unwrap();

        assert_eq!(inner.calls, 1);
        assert_ne!(
            live.diagnostic.as_ref().unwrap().attempt_id,
            cached.diagnostic.as_ref().unwrap().attempt_id
        );
        let records = crate::diagnostics::read_manifest(&manifest).unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].cache_source, "live");
        assert_eq!(records[1].cache_source, "cache");
        assert_ne!(records[0].attempt_id, records[1].attempt_id);
    }

    #[test]
    fn transient_startup_failures_are_measured_fresh_but_oom_is_cached() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trial-cache-model-x.json");
        let overrides = localbench_search::overrides::overrides_of(&[("KvK", "q8_0".into())]);

        let mut inner = ScriptedRunner {
            calls: 0,
            next: failed_trial(false),
        };
        {
            let (cache, _) = TrialCache::open(&path, &fingerprint(), true);
            let mut runner = CachedRunner {
                inner: &mut inner,
                cache,
                driver: Driver::Server,
                stamp,
                ledger: None,
                last_trial: None,
            };
            let _ = runner.measure(&overrides, "baseline");
            assert!(runner
                .last_failure_summary()
                .is_some_and(|summary| summary.starts_with("launch/spawn_failed")));
            let _ = runner.measure(&overrides, "baseline");
        }
        assert_eq!(
            inner.calls, 2,
            "a plain startup failure is transient and re-measured"
        );

        inner.next = failed_trial(true);
        {
            let (cache, _) = TrialCache::open(&path, &fingerprint(), true);
            let mut runner = CachedRunner {
                inner: &mut inner,
                cache,
                driver: Driver::Server,
                stamp,
                ledger: None,
                last_trial: None,
            };
            let _ = runner.measure(&overrides, "baseline");
            let again = runner.measure(&overrides, "baseline");
            assert!(again.oom, "the definite OOM answer came from the cache");
        }
        assert_eq!(inner.calls, 3);
    }

    #[test]
    fn ineligible_phases_always_measure_live() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trial-cache-model-x.json");
        let overrides = localbench_search::overrides::overrides_of(&[("KvK", "q8_0".into())]);
        let mut inner = ScriptedRunner {
            calls: 0,
            next: measured(),
        };
        {
            let (cache, _) = TrialCache::open(&path, &fingerprint(), true);
            let mut runner = CachedRunner {
                inner: &mut inner,
                cache,
                driver: Driver::Server,
                stamp,
                ledger: None,
                last_trial: None,
            };
            let _ = runner.measure(&overrides, "verify");
            let _ = runner.measure(&overrides, "verify");
        }
        assert_eq!(inner.calls, 2, "verify never trusts the cache");
    }

    // --- process-aware startup wait (LocalHub#77) ---------------------------

    use localx_llama_core::{BackendSession, KvTypes, LauncherError, LauncherVersion};

    /// A launcher that never reports a ready server: `wait_server` consumes its
    /// slice and returns not-ready, exactly like polling a port with nothing
    /// behind it. Every other method is unreachable on the post-spawn path
    /// `measure_spawned` drives, so they stay `unimplemented!`.
    struct NeverReadyLauncher;

    impl Launcher for NeverReadyLauncher {
        fn wait_server(&self, _port: u16, timeout_secs: u32) -> Result<(), LauncherError> {
            std::thread::sleep(Duration::from_secs(u64::from(timeout_secs)));
            Err(LauncherError::Unavailable("stub: never ready".to_string()))
        }

        fn version(&self) -> LauncherVersion {
            unimplemented!("launcher not used past spawn")
        }
        fn model_def(&self, _key: &str) -> Result<ModelDef, LauncherError> {
            unimplemented!("launcher not used past spawn")
        }
        fn gguf_path(
            &self,
            _def: &ModelDef,
            _quant: Option<&str>,
        ) -> Result<PathBuf, LauncherError> {
            unimplemented!("launcher not used past spawn")
        }
        fn context_value(&self, _def: &ModelDef, _context_key: &str) -> Result<u32, LauncherError> {
            unimplemented!("launcher not used past spawn")
        }
        fn resolve_context_key(
            &self,
            _def: &ModelDef,
            _context_key: &str,
        ) -> Result<String, LauncherError> {
            unimplemented!("launcher not used past spawn")
        }
        fn vision_module_path(&self, _key: &str, _def: &ModelDef) -> Option<PathBuf> {
            unimplemented!("launcher not used past spawn")
        }
        fn resolve_quant_key(
            &self,
            _def: &ModelDef,
            _quant: &str,
        ) -> Result<String, LauncherError> {
            unimplemented!("launcher not used past spawn")
        }
        fn vram_gb(&self) -> u32 {
            unimplemented!("launcher not used past spawn")
        }
        fn server_binary(
            &self,
            _mode: Mode,
            _non_interactive: bool,
        ) -> Result<PathBuf, LauncherError> {
            unimplemented!("launcher not used past spawn")
        }
        fn bench_binary(&self, _non_interactive: bool) -> Option<PathBuf> {
            unimplemented!("launcher not used past spawn")
        }
        fn perplexity_binary(&self, _non_interactive: bool, _mode: Mode) -> Option<PathBuf> {
            unimplemented!("launcher not used past spawn")
        }
        fn install_root(&self, _mode: Mode) -> PathBuf {
            unimplemented!("launcher not used past spawn")
        }
        fn kv_types(&self, _def: &ModelDef) -> KvTypes {
            unimplemented!("launcher not used past spawn")
        }
        fn kv_type_supported(&self, _kv_type: &str, _mode: Mode) -> bool {
            unimplemented!("launcher not used past spawn")
        }
        fn free_port(&self, _start: u16) -> Result<u16, LauncherError> {
            unimplemented!("launcher not used past spawn")
        }
        fn stop_server(&self, _quiet: bool) {
            unimplemented!("launcher not used past spawn")
        }
        fn set_backend_session(&self, _session: &BackendSession) {
            unimplemented!("launcher not used past spawn")
        }
        fn expand_path(&self, _path: &str) -> PathBuf {
            unimplemented!("launcher not used past spawn")
        }
    }

    fn live_runner(
        launcher: &dyn Launcher,
        log_dir: PathBuf,
        startup_timeout_secs: u32,
    ) -> LiveRunner<'_> {
        LiveRunner::new(
            launcher,
            TrialTarget {
                key: "stub".to_string(),
                def: ModelDef::default(),
                context_key: "64k".to_string(),
                mode: Mode::Native,
                model_arg_path: String::new(),
                runs: 1,
                port_start: 0,
                log_dir,
                settings_params: LaunchParams::default(),
            },
            startup_timeout_secs,
            "test-run",
        )
    }

    /// A process that writes an OOM signature to its (redirected) log, then exits.
    fn oom_exit_command() -> (&'static str, Vec<String>) {
        #[cfg(windows)]
        return (
            "cmd",
            vec![
                "/c".to_string(),
                "echo failed to allocate backend buffer".to_string(),
            ],
        );
        #[cfg(not(windows))]
        return (
            "sh",
            vec![
                "-c".to_string(),
                "echo 'failed to allocate backend buffer'".to_string(),
            ],
        );
    }

    /// A process that exits immediately writing nothing to its log.
    fn plain_exit_command() -> (&'static str, Vec<String>) {
        #[cfg(windows)]
        return ("cmd", vec!["/c".to_string(), "exit 1".to_string()]);
        #[cfg(not(windows))]
        return ("sh", vec!["-c".to_string(), "exit 1".to_string()]);
    }

    /// A process that stays alive for several seconds without answering.
    /// Spawned directly (not via a shell) so `Child::kill` targets the sleeper
    /// itself — a shell wrapper would leave the real `ping`/`sleep` running as a
    /// surviving grandchild, and on Windows holding the log handle.
    fn long_lived_command() -> (&'static str, Vec<String>) {
        #[cfg(windows)]
        return (
            "ping",
            vec!["-n".to_string(), "5".to_string(), "127.0.0.1".to_string()],
        );
        #[cfg(not(windows))]
        return ("sleep", vec!["4".to_string()]);
    }

    #[test]
    fn a_crashed_server_is_classified_from_its_log_without_waiting_the_budget() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("trial.log");
        let (prog, args) = oom_exit_command();
        let mut child = spawn_detached(prog, &args, None, Some(&log)).unwrap();

        let live = live_runner(&NeverReadyLauncher, dir.path().to_path_buf(), 30);
        let start = Instant::now();
        let trial = live.measure_spawned(&mut child, 1, &log);

        assert!(!trial.startup_ok);
        assert!(trial.oom);
        assert_eq!(trial.startup_failure, Some(StartupFailure::ExitedOom));
        assert!(
            start.elapsed() < Duration::from_secs(10),
            "the exited process is recognized promptly, not after the 30s budget"
        );
        assert!(
            child.try_wait().unwrap().is_some(),
            "the child is reaped, not left running"
        );
    }

    #[test]
    fn a_crash_without_an_oom_signature_records_a_plain_exit() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("trial.log");
        let (prog, args) = plain_exit_command();
        let mut child = spawn_detached(prog, &args, None, Some(&log)).unwrap();

        let live = live_runner(&NeverReadyLauncher, dir.path().to_path_buf(), 30);
        let trial = live.measure_spawned(&mut child, 1, &log);

        assert!(!trial.startup_ok);
        assert!(!trial.oom);
        assert_eq!(trial.startup_failure, Some(StartupFailure::Exited));
        assert!(
            child.try_wait().unwrap().is_some(),
            "the child is reaped, not left running"
        );
    }

    #[test]
    fn a_server_that_never_answers_waits_the_budget_then_reaps_the_child() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("trial.log");
        let (prog, args) = long_lived_command();
        let mut child = spawn_detached(prog, &args, None, Some(&log)).unwrap();

        let live = live_runner(&NeverReadyLauncher, dir.path().to_path_buf(), 1);
        let start = Instant::now();
        let trial = live.measure_spawned(&mut child, 1, &log);
        let elapsed = start.elapsed();

        assert!(!trial.startup_ok);
        assert!(!trial.oom);
        assert_eq!(trial.startup_failure, Some(StartupFailure::TimedOut));
        // The full 1s budget is spent — the process stayed alive throughout...
        assert!(elapsed >= Duration::from_millis(900));
        // ...but the child's own ~4s lifetime is not: it is killed at the
        // deadline rather than waited out.
        assert!(
            elapsed < Duration::from_secs(4),
            "a still-alive server is reaped at the deadline, not waited out"
        );
        // The directly-spawned sleeper is actually dead: this is the leak-free
        // check — with the kill/wait removed it would still be running here.
        assert!(
            child.try_wait().unwrap().is_some(),
            "the still-alive server is killed and reaped, not leaked"
        );
    }
}
