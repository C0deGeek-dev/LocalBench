//! `llama-bench` screening: order a phase's candidates by throughput measured
//! in one `llama-bench` process, so only the most promising get a full server
//! measurement.
//!
//! A screen result is never a measurement. It is not scored, cached, ranked
//! against server trials, verified, or saved — it only decides which
//! candidates are worth a real `/v1/chat/completions` trial. `llama-bench`
//! runs no chat template, no prompt cache, and a context of only
//! prompt + generation tokens, so its numbers are comparable with each other
//! and nothing else.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use localbench_scoring::score::{trial_score, Optimize, Overrides, Trial, Workload};
use localx_llama_core::args::{resolve_kv_types, LaunchParams};
use localx_llama_core::capabilities::LoadFlags;
use localx_llama_core::Launcher;
use localx_llama_runtime::tool::run_tool;
use serde_json::Value;

use crate::trial::{candidate_launch_params, TrialTarget};
use crate::tuner::BenchScreen;

/// Flag/value pairs, sorted: one side of a `llama-bench` configuration.
type Flags = Vec<(String, String)>;
/// Prompt and generation throughput, tokens per second.
type Rates = (f64, f64);

/// Prompt tokens per screening test: large enough that prefill speed, which
/// dominates a coding agent's latency, shows up.
pub const SCREEN_PROMPT_TOKENS: u32 = 2048;
/// Generated tokens per screening test.
pub const SCREEN_GEN_TOKENS: u32 = 128;
/// Repetitions per screening test (the server verify samples again).
pub const SCREEN_REPETITIONS: u32 = 2;
/// One screening process may take this long (model load included).
const SCREEN_TIMEOUT: Duration = Duration::from_secs(1800);

/// The `llama-bench` executable name for this platform.
#[must_use]
pub fn bench_exe_name() -> &'static str {
    if cfg!(windows) {
        "llama-bench.exe"
    } else {
        "llama-bench"
    }
}

/// The live screen over one tuning target, using the `llama-bench` that
/// ships beside the engine's `llama-server`.
pub struct LiveBenchScreen<'a> {
    launcher: &'a dyn Launcher,
    target: TrialTarget,
    binary: PathBuf,
    optimize: Optimize,
}

impl<'a> LiveBenchScreen<'a> {
    /// A screen for `target`, when its engine ships `llama-bench`.
    #[must_use]
    pub fn new(
        launcher: &'a dyn Launcher,
        target: TrialTarget,
        optimize: Optimize,
    ) -> Option<Self> {
        let server = launcher.server_binary(target.mode, true).ok()?;
        let binary = server.with_file_name(bench_exe_name());
        binary.is_file().then_some(Self {
            launcher,
            target,
            binary,
            optimize,
        })
    }
}

/// One candidate as `llama-bench` sees it.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct BenchShape {
    /// Values that reload the model when they change.
    model: Vec<(String, String)>,
    /// Values `llama-bench` sweeps in-process, as (flag, value).
    context: Vec<(String, String)>,
}

/// Translate a candidate's launch parameters into `llama-bench` flags.
fn bench_shape(def_params: &LaunchParams, kv: (String, String), load: LoadFlags) -> BenchShape {
    let mut model = vec![
        (
            "-ngl".to_string(),
            def_params
                .n_gpu_layers
                .filter(|n| *n > 0)
                .unwrap_or(99)
                .to_string(),
        ),
        (
            "-ncmoe".to_string(),
            def_params.n_cpu_moe.unwrap_or(0).to_string(),
        ),
    ];
    let mlock = def_params.mlock.unwrap_or(false);
    let no_mmap = def_params.no_mmap.unwrap_or(false);
    match load {
        LoadFlags::LoadMode => {
            let mode = match (mlock, no_mmap) {
                (false, false) => None,
                (false, true) => Some("none"),
                (true, false) => Some("mmap+mlock"),
                (true, true) => Some("mlock"),
            };
            if let Some(mode) = mode {
                model.push(("-lm".to_string(), mode.to_string()));
            }
        }
        LoadFlags::Legacy => {
            if no_mmap {
                model.push(("-mmp".to_string(), "0".to_string()));
            }
        }
    }
    let mut context = vec![
        ("-ctk".to_string(), kv.0.to_ascii_lowercase()),
        ("-ctv".to_string(), kv.1.to_ascii_lowercase()),
    ];
    if let Some(ub) = def_params.ubatch_size.filter(|n| *n > 0) {
        context.push(("-ub".to_string(), ub.to_string()));
    }
    if let Some(b) = def_params.batch_size.filter(|n| *n > 0) {
        context.push(("-b".to_string(), b.to_string()));
    }
    if let Some(fa) = def_params.flash_attn {
        context.push(("-fa".to_string(), if fa { "on" } else { "off" }.to_string()));
    }
    if let Some(threads) = def_params.threads.filter(|n| *n > 0) {
        context.push(("-t".to_string(), threads.to_string()));
    }
    BenchShape { model, context }
}

/// The `llama-bench` argv for a group of candidates sharing model values:
/// each context flag carries the comma list of every value in the group.
fn bench_args(model_path: &str, model: &[(String, String)], group: &[&BenchShape]) -> Vec<String> {
    let mut args = vec!["-m".to_string(), model_path.to_string()];
    for (flag, value) in model {
        args.push(flag.clone());
        args.push(value.clone());
    }
    let mut lists: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for shape in group {
        for (flag, value) in &shape.context {
            let values = lists.entry(flag.clone()).or_default();
            if !values.contains(value) {
                values.push(value.clone());
            }
        }
    }
    for (flag, values) in lists {
        args.push(flag);
        args.push(values.join(","));
    }
    args.extend(
        [
            "-p",
            &SCREEN_PROMPT_TOKENS.to_string(),
            "-n",
            &SCREEN_GEN_TOKENS.to_string(),
            "-r",
            &SCREEN_REPETITIONS.to_string(),
            "-o",
            "json",
        ]
        .map(str::to_string),
    );
    args
}

/// Throughput per context shape from `llama-bench -o json`: (prompt t/s,
/// generation t/s), keyed by the context flags the row reports.
fn parse_bench_json(stdout: &str) -> BTreeMap<Flags, Rates> {
    let rows: Vec<Value> = serde_json::from_str(stdout.trim()).unwrap_or_default();
    let mut out: BTreeMap<Flags, Rates> = BTreeMap::new();
    for row in rows {
        let Some(key) = row_key(&row) else {
            continue;
        };
        let rate = row.get("avg_ts").and_then(Value::as_f64).unwrap_or(0.0);
        let prompt = row.get("n_prompt").and_then(Value::as_i64).unwrap_or(0);
        let entry = out.entry(key).or_insert((0.0, 0.0));
        if prompt > 0 {
            entry.0 = rate;
        } else {
            entry.1 = rate;
        }
    }
    out
}

/// A row's context flags in the same spelling [`bench_shape`] produces.
fn row_key(row: &Value) -> Option<Flags> {
    let int = |field: &str| row.get(field).and_then(Value::as_i64);
    let text = |field: &str| row.get(field).and_then(Value::as_str).map(str::to_string);
    let flash = match row.get("flash_attn") {
        Some(Value::Bool(on)) => Some(*on),
        Some(Value::Number(n)) => n.as_i64().map(|n| n > 0),
        _ => None,
    };
    let mut key = vec![
        ("-ctk".to_string(), text("type_k")?),
        ("-ctv".to_string(), text("type_v")?),
        ("-ub".to_string(), int("n_ubatch")?.to_string()),
        ("-b".to_string(), int("n_batch")?.to_string()),
        ("-t".to_string(), int("n_threads")?.to_string()),
    ];
    if let Some(on) = flash {
        key.push(("-fa".to_string(), if on { "on" } else { "off" }.to_string()));
    }
    key.sort();
    Some(key)
}

/// Whether a row key matches a shape's context flags (a shape leaves unset
/// flags at the bench default, which any row value satisfies).
fn row_matches(row: &[(String, String)], shape: &BenchShape) -> bool {
    shape
        .context
        .iter()
        .all(|(flag, value)| row.iter().any(|(f, v)| f == flag && v == value))
}

impl LiveBenchScreen<'_> {
    fn run_group(
        &self,
        model: &[(String, String)],
        group: &[&BenchShape],
    ) -> BTreeMap<Flags, Rates> {
        let args = bench_args(&self.target.model_arg_path, model, group);
        run_tool(Path::new(&self.binary), &args, SCREEN_TIMEOUT)
            .filter(|output| output.success)
            .map(|output| parse_bench_json(&output.stdout))
            .unwrap_or_default()
    }
}

impl BenchScreen for LiveBenchScreen<'_> {
    fn rank(&mut self, candidates: &[Overrides]) -> Option<Vec<usize>> {
        let load = self
            .launcher
            .server_capabilities(self.target.mode)
            .load_flags();
        let shapes: Vec<Option<BenchShape>> = candidates
            .iter()
            .map(|overrides| {
                let params =
                    candidate_launch_params(self.launcher, &self.target, overrides).ok()?;
                let kv = resolve_kv_types(
                    &self.target.def,
                    params.kv_k.as_deref(),
                    params.kv_v.as_deref(),
                );
                Some(bench_shape(&params, kv, load))
            })
            .collect();
        // One llama-bench process per set of model-reloading values.
        let mut groups: BTreeMap<Flags, Vec<&BenchShape>> = BTreeMap::new();
        for shape in shapes.iter().flatten() {
            groups.entry(shape.model.clone()).or_default().push(shape);
        }
        let mut results: Vec<(Flags, BenchShape, Rates)> = Vec::new();
        for (model, group) in &groups {
            for (row, rates) in self.run_group(model, group) {
                if let Some(shape) = group.iter().find(|shape| row_matches(&row, shape)) {
                    results.push((row, (*shape).clone(), rates));
                }
            }
        }
        if results.is_empty() {
            return None;
        }
        let workload = Workload::default();
        let score = |shape: &Option<BenchShape>| -> f64 {
            let Some(shape) = shape else { return -1.0 };
            results
                .iter()
                .filter(|(_, s, _)| s == shape)
                .map(|(_, _, (pp, tg))| {
                    let screen = Trial {
                        startup_ok: true,
                        measurement_usable: true,
                        pp_tps: *pp,
                        tg_tps: *tg,
                        ..Trial::default()
                    };
                    trial_score(&screen, self.optimize, &workload)
                })
                .fold(-1.0, f64::max)
        };
        let mut order: Vec<(usize, f64)> = shapes
            .iter()
            .enumerate()
            .map(|(i, shape)| (i, score(shape)))
            .collect();
        order.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Some(order.into_iter().map(|(i, _)| i).collect())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn params(ub: i64, fa: bool, threads: i64) -> LaunchParams {
        LaunchParams {
            n_cpu_moe: Some(36),
            ubatch_size: Some(ub),
            batch_size: Some(2048),
            flash_attn: Some(fa),
            threads: Some(threads),
            no_mmap: Some(true),
            ..LaunchParams::default()
        }
    }

    fn kv() -> (String, String) {
        ("q8_0".to_string(), "q8_0".to_string())
    }

    #[test]
    fn a_group_sweeps_its_context_values_in_one_process() {
        let a = bench_shape(&params(512, true, 12), kv(), LoadFlags::LoadMode);
        let b = bench_shape(&params(1024, true, 16), kv(), LoadFlags::LoadMode);
        assert_eq!(a.model, b.model);
        let args = bench_args("m.gguf", &a.model, &[&a, &b]).join(" ");
        assert!(
            args.starts_with("-m m.gguf -ngl 99 -ncmoe 36 -lm none "),
            "{args}"
        );
        assert!(args.contains("-ub 512,1024"), "{args}");
        assert!(args.contains("-t 12,16"), "{args}");
        assert!(args.contains("-fa on "), "{args}");
        assert!(args.ends_with("-p 2048 -n 128 -r 2 -o json"), "{args}");
    }

    #[test]
    fn legacy_builds_screen_no_mmap_with_the_mmap_axis() {
        let shape = bench_shape(&params(512, true, 12), kv(), LoadFlags::Legacy);
        assert!(shape.model.contains(&("-mmp".to_string(), "0".to_string())));
        assert!(!shape.model.iter().any(|(flag, _)| flag == "-lm"));
    }

    #[test]
    fn bench_rows_map_back_to_their_shape() {
        let json = r#"[
          {"n_prompt":2048,"n_gen":0,"avg_ts":200.0,"type_k":"q8_0","type_v":"q8_0","n_ubatch":512,"n_batch":2048,"n_threads":12,"flash_attn":1},
          {"n_prompt":0,"n_gen":128,"avg_ts":26.5,"type_k":"q8_0","type_v":"q8_0","n_ubatch":512,"n_batch":2048,"n_threads":12,"flash_attn":1},
          {"n_prompt":2048,"n_gen":0,"avg_ts":210.0,"type_k":"q8_0","type_v":"q8_0","n_ubatch":1024,"n_batch":2048,"n_threads":16,"flash_attn":1}
        ]"#;
        let rows = parse_bench_json(json);
        let a = bench_shape(&params(512, true, 12), kv(), LoadFlags::LoadMode);
        let (row, rates) = rows.iter().find(|(row, _)| row_matches(row, &a)).unwrap();
        assert_eq!(*rates, (200.0, 26.5));
        assert!(row.contains(&("-t".to_string(), "12".to_string())));
        let b = bench_shape(&params(1024, true, 12), kv(), LoadFlags::LoadMode);
        assert!(
            !rows.keys().any(|row| row_matches(row, &b)),
            "threads differ"
        );
    }
}
