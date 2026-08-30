//! The `localbench` binary: benchmark CLI over the library layer.
//!
//! Structured results print to stdout (JSON by default when stdout is not a
//! terminal); human logs go to stderr. The command surface covers the live
//! tuner (`findbest`), the harness-capability matrix (`arms`) with its offline
//! `rescore`, the lesson-uplift A/B (`uplift`), and the deterministic `version`
//! and instrument self-tests.

use std::io::IsTerminal;
use std::process::ExitCode;

use localbench::consumer;
use localbench::output::{resolve_format, OutputFormat};
use localbench_measure::runner;

const VERSION: &str = env!("CARGO_PKG_VERSION");

const USAGE: &str = "\
usage: localbench <command> [options]

commands:
  version                    print the version envelope (JSON)
  instruments                run the deterministic instrument self-tests
  findbest --model <key> [--context <k>] [--mode native|turboquant|mtpturbo|prism]
           [--quant <q>] [--profile pure|balanced|both] [--budget <n>]
           [--beam-width <1..100>]
           [--runs <n>] [--optimize gen|prompt|both|coding-agent] [--no-save]
           [--no-cache] [--startup-timeout <secs>]
                             tune the model live and save the best config;
                             decisive measurements persist in the trial cache
  arms --spec <run-spec.json> [--cells-dir <dir>] [--grade docker|none]
       [--localpilot <bin>] [--solver-timeout <s>] [--grade-timeout <s>]
                             drive the solver across every arm x task cell,
                             grade in a network-isolated container, keep cells
  uplift --report <file>     render a saved lesson-uplift report
  uplift --task-set <file> --workspace <dir> --model <key> [--trials <n>]
         [--localpilot <bin>] [--timeout <s>] [--intended a,b]
                             run the live lesson-on/off A/B. The operator stages
                             per-arm memory FIRST (the CLI does not): the
                             baseline needs a clean store (learning off) and the
                             lesson arm needs the seed pack staged with learning
                             on (localbench uplift --emit-seed-pack ->
                             localpilot learning seed). A mis-staged run VOIDs.
  rescore --dir <cells-dir> [--corpus first-party|external]
                             recompute the comparative report from kept cells
options:
  --format text|json         output format (default: json when stdout is piped)
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(code) => code,
        Err(message) => {
            eprintln!("localbench: {message}");
            ExitCode::FAILURE
        }
    }
}

fn flag_value(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn parse_mode(value: Option<&str>) -> Result<Option<localx_llama_core::Mode>, String> {
    let mode = match value {
        None => return Ok(None),
        Some("native") => localx_llama_core::Mode::Native,
        Some("turboquant") => localx_llama_core::Mode::Turboquant,
        Some("mtpturbo") => localx_llama_core::Mode::Mtpturbo,
        Some("prism" | "prismml") => localx_llama_core::Mode::PrismMl,
        Some(other) => {
            return Err(format!(
                "unknown --mode '{other}' (use native|turboquant|mtpturbo|prism)"
            ))
        }
    };
    Ok(Some(mode))
}

fn cli_mode_name(mode: localx_llama_core::Mode) -> &'static str {
    match mode {
        localx_llama_core::Mode::PrismMl => "prism",
        _ => mode.as_str(),
    }
}

fn resolve_mode(
    model: &str,
    requested: Option<localx_llama_core::Mode>,
    required: Option<localx_llama_core::Mode>,
) -> Result<localx_llama_core::Mode, String> {
    if let Some(required) = required {
        if let Some(requested) = requested {
            if requested != required {
                return Err(format!(
                    "{model} requires --mode {}; '{}' is incompatible",
                    cli_mode_name(required),
                    cli_mode_name(requested)
                ));
            }
        }
        return Ok(required);
    }
    Ok(requested.unwrap_or(localx_llama_core::Mode::Native))
}

fn output_format(args: &[String]) -> Result<OutputFormat, String> {
    let explicit = match flag_value(args, "--format").as_deref() {
        Some("text") => Some(OutputFormat::Text),
        Some("json") => Some(OutputFormat::Json),
        Some(other) => return Err(format!("unknown --format '{other}' (use text|json)")),
        None => None,
    };
    Ok(resolve_format(explicit, std::io::stdout().is_terminal()))
}

fn print_json(value: &serde_json::Value) -> Result<(), String> {
    let json = serde_json::to_string_pretty(value).map_err(|e| e.to_string())?;
    println!("{json}");
    Ok(())
}

fn run(args: &[String]) -> Result<ExitCode, String> {
    let Some(command) = args.first() else {
        eprint!("{USAGE}");
        return Ok(ExitCode::FAILURE);
    };

    match command.as_str() {
        "version" => {
            let envelope = consumer::own_version(VERSION);
            print_json(&serde_json::to_value(envelope).map_err(|e| e.to_string())?)?;
            Ok(ExitCode::SUCCESS)
        }
        "instruments" => {
            let checks = runner::instrument_checks();
            let ready = runner::instruments_ready(&checks);
            match output_format(args)? {
                OutputFormat::Json => print_json(&serde_json::json!({
                    "ready": ready,
                    "checks": checks,
                }))?,
                OutputFormat::Text => {
                    for check in &checks {
                        eprintln!(
                            "  {:<24} {}",
                            check.name,
                            if check.ok { "ok" } else { "BROKEN" }
                        );
                    }
                    println!("{}", if ready { "ready" } else { "broken" });
                }
            }
            Ok(if ready {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            })
        }
        "findbest" => cmd_findbest(args),
        "arms" => cmd_arms(args),
        "uplift" => cmd_uplift(args),
        "rescore" => {
            let dir = flag_value(args, "--dir").ok_or("rescore needs --dir <cells-dir>")?;
            let corpus = flag_value(args, "--corpus").unwrap_or_else(|| "external".to_string());
            if corpus != "external" && corpus != "first-party" {
                return Err(format!(
                    "unknown --corpus '{corpus}' (use first-party|external)"
                ));
            }
            let report =
                runner::rescore(std::path::Path::new(&dir), &corpus).map_err(|e| e.to_string())?;
            print_json(&serde_json::to_value(report).map_err(|e| e.to_string())?)?;
            Ok(ExitCode::SUCCESS)
        }
        "--help" | "-h" | "help" => {
            eprint!("{USAGE}");
            Ok(ExitCode::SUCCESS)
        }
        other => Err(format!("unknown command '{other}'\n{USAGE}")),
    }
}

fn cmd_arms(args: &[String]) -> Result<ExitCode, String> {
    use localbench::matrix::{
        load_run_spec, os_exec, render_capability_report, run_matrix, ContainerGrader, Grader,
        NoGrader,
    };
    use localbench::output::{MachineOutput, RunEvent};
    use localbench::solver::LocalPilotSolver;

    let spec_path = flag_value(args, "--spec").ok_or("arms needs --spec <run-spec.json>")?;
    let spec = load_run_spec(std::path::Path::new(&spec_path))?;
    let cells_dir = flag_value(args, "--cells-dir")
        .map(std::path::PathBuf::from)
        .or_else(|| {
            home_dir().map(|h| {
                // A per-run subfolder so a new run's cells never mix with a prior
                // run's — mixing would break `rescore` ≡ live. The stamp is
                // filesystem-safe (no ':', which is illegal in a Windows path).
                let stamp = now_iso().replace(':', "");
                h.join(".localbench")
                    .join("runs")
                    .join(format!("arms-{}-{stamp}", spec.model))
            })
        })
        .ok_or("could not resolve a cells directory; pass --cells-dir")?;
    let solver_timeout = flag_value(args, "--solver-timeout")
        .map(|v| v.parse().map_err(|_| format!("bad --solver-timeout '{v}'")))
        .transpose()?
        .unwrap_or(600_u64);
    let grade_timeout = flag_value(args, "--grade-timeout")
        .map(|v| v.parse().map_err(|_| format!("bad --grade-timeout '{v}'")))
        .transpose()?
        .unwrap_or(900_u64);
    let bin = flag_value(args, "--localpilot").unwrap_or_else(|| "localpilot".to_string());
    let grade_mode = flag_value(args, "--grade").unwrap_or_else(|| "docker".to_string());

    let mut solver = LocalPilotSolver {
        bin,
        timeout: std::time::Duration::from_secs(solver_timeout),
    };

    let format = output_format(args)?;
    let mut output = MachineOutput::new(std::io::stdout(), std::io::stderr());
    // run_matrix owns the JSONL protocol (started → result per cell →
    // completed/error); this command just puts it on the wire in JSON mode.
    let mut on_event = |event: &RunEvent| {
        if format == OutputFormat::Json {
            let _ = output.event(event);
        }
    };
    let mut run = |grader: &mut dyn Grader| {
        run_matrix(
            &mut solver,
            grader,
            &spec,
            &cells_dir,
            &mut |line| {
                eprintln!("{line}");
            },
            &mut on_event,
        )
    };
    let outcome = match grade_mode.as_str() {
        "docker" => {
            let mut grader =
                ContainerGrader::new(os_exec, std::time::Duration::from_secs(grade_timeout));
            run(&mut grader)?
        }
        "none" => run(&mut NoGrader)?,
        other => return Err(format!("unknown --grade '{other}' (use docker|none)")),
    };
    for caveat in &outcome.caveats {
        eprintln!("{caveat}");
    }
    match format {
        OutputFormat::Json => {}
        OutputFormat::Text => {
            println!("{}", render_capability_report(&outcome.report));
            if let Some(reason) = &outcome.aborted {
                println!("\nRun yielded early: {reason}");
            }
            println!(
                "\nCells kept in {} (rescore with `localbench rescore --dir`).",
                cells_dir.display()
            );
        }
    }
    Ok(if outcome.aborted.is_none() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}

fn cmd_uplift(args: &[String]) -> Result<ExitCode, String> {
    use localbench::upliftrun::{
        load_task_set, render_uplift_report, run_uplift, seed_pack, PrintDriver, UpliftReport,
    };

    // Render mode: a saved report becomes Markdown.
    if let Some(report_path) = flag_value(args, "--report") {
        let raw = std::fs::read_to_string(&report_path)
            .map_err(|e| format!("uplift report not found: {report_path}: {e}"))?;
        let report: UpliftReport =
            serde_json::from_str(&raw).map_err(|e| format!("{report_path} does not parse: {e}"))?;
        println!("{}", render_uplift_report(&report));
        return Ok(ExitCode::SUCCESS);
    }

    let set_path = flag_value(args, "--task-set")
        .ok_or("uplift needs --report <file> or --task-set <file>")?;
    let set = load_task_set(std::path::Path::new(&set_path))?;

    // Seed-pack projection mode: print the pack the lesson arm seeds.
    if args.iter().any(|a| a == "--emit-seed-pack") {
        let pack = serde_json::to_string_pretty(&seed_pack(&set)).map_err(|e| e.to_string())?;
        println!("{pack}");
        return Ok(ExitCode::SUCCESS);
    }

    let workspace = flag_value(args, "--workspace").ok_or("uplift needs --workspace <dir>")?;
    let model = flag_value(args, "--model").ok_or("uplift needs --model <key>")?;
    let trials: u32 = flag_value(args, "--trials")
        .map(|v| v.parse().map_err(|_| format!("bad --trials '{v}'")))
        .transpose()?
        .unwrap_or(5);
    let timeout = flag_value(args, "--timeout")
        .map(|v| v.parse().map_err(|_| format!("bad --timeout '{v}'")))
        .transpose()?
        .unwrap_or(600_u64);
    let bin = flag_value(args, "--localpilot").unwrap_or_else(|| "localpilot".to_string());
    // The intended lesson ids come from the seeding step (the engine's ids are
    // never recomputed here); default to the task set's declared ids.
    let intended: Vec<String> = match flag_value(args, "--intended") {
        Some(list) => list.split(',').map(|s| s.trim().to_string()).collect(),
        None => set.lessons.iter().map(|l| l.id.clone()).collect(),
    };

    let workspace = std::path::PathBuf::from(workspace);
    let mut baseline = PrintDriver {
        bin: bin.clone(),
        workspace: workspace.clone(),
        model: model.clone(),
        timeout: std::time::Duration::from_secs(timeout),
    };
    let mut lessons = PrintDriver {
        bin,
        workspace,
        model: model.clone(),
        timeout: std::time::Duration::from_secs(timeout),
    };
    eprintln!(
        "Running the lesson-on/off A/B in one workspace. Stage per-arm memory \
         FIRST — the baseline arm needs a clean store (learning off, no seeded \
         memory) and the lesson arm needs the seed pack staged with learning on \
         (localpilot learning seed). This CLI does not stage per-arm memory; a \
         mis-staged run (baseline retrieved memory, or the lesson arm never \
         injected) voids the result. See the operator scripts."
    );
    let report = run_uplift(&set, &mut baseline, &mut lessons, &intended, trials, &model)?;
    match output_format(args)? {
        OutputFormat::Json => {
            print_json(&serde_json::to_value(&report).map_err(|e| e.to_string())?)?;
        }
        OutputFormat::Text => println!("{}", render_uplift_report(&report)),
    }
    Ok(ExitCode::SUCCESS)
}

/// A UTC RFC3339 timestamp from the system clock (no external time crate).
fn now_iso() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    iso_from_secs(secs)
}

/// A UTC RFC3339 timestamp from Unix seconds.
fn iso_from_secs(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let (h, m, s) = ((secs % 86_400) / 3_600, (secs % 3_600) / 60, secs % 60);
    // Civil-from-days (the standard era-based calendar conversion).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mth = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mth <= 2 { y + 1 } else { y };
    format!("{y:04}-{mth:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
}

fn home_dir() -> Option<std::path::PathBuf> {
    let var = if cfg!(windows) { "USERPROFILE" } else { "HOME" };
    std::env::var(var)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .map(std::path::PathBuf::from)
}

fn has_no_save(args: &[String]) -> bool {
    args.iter().any(|a| a == "--no-save")
}

fn probe_vram_gb() -> u32 {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .output();
    match output {
        Ok(out) if out.status.success() => localx_llama_runtime::probe::parse_nvidia_smi_vram_gb(
            &String::from_utf8_lossy(&out.stdout),
        )
        .and_then(|gb| u32::try_from(gb).ok())
        .unwrap_or(0),
        _ => 0,
    }
}

fn probe_available_ram_gb() -> f64 {
    let mut system = sysinfo::System::new_all();
    system.refresh_memory();
    system.available_memory() as f64 / 1_073_741_824.0
}

/// GPU names for the trial-cache fingerprint — a GPU swap (same VRAM GB) must
/// still invalidate reused measurements.
fn probe_gpu_names() -> Vec<String> {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output();
    match output {
        Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout)
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(str::to_string)
            .collect(),
        _ => Vec::new(),
    }
}

/// A build signature for the resolved `llama-server` binary — its variant dir
/// (e.g. `win-cuda-13.3`) plus file size, so an `llm-update` that swaps the
/// binary invalidates measurements tuned against the old one.
fn probe_llamacpp_build(
    launcher: &localbox_launcher::launcher::LlamaLauncher,
    mode: localx_llama_core::Mode,
) -> String {
    use localx_llama_core::Launcher;
    let Ok(bin) = launcher.server_binary(mode, true) else {
        return String::new();
    };
    let variant = bin
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|s| s.to_str())
        .unwrap_or("");
    let size = std::fs::metadata(&bin).map(|m| m.len()).unwrap_or(0);
    if variant.is_empty() && size == 0 {
        String::new()
    } else {
        format!("{variant}:{size}")
    }
}

fn cmd_findbest(args: &[String]) -> Result<ExitCode, String> {
    use localbench::export::{export_best_config, ExportProvenance, ExportWinner, StorePaths};
    use localbench::trial::{
        chat_prompt_hash, chat_request_shape_hash, session_fingerprint, trial_launch_params,
        LiveRunner, TrialTarget, MEASUREMENT_PROTOCOL, RESPONSE_SCHEMA, TELEMETRY_PROTOCOL,
    };
    use localbench::tuner::{run_tuner, TunerParams, DEFAULT_BEAM_WIDTH};
    use localbench_scoring::score::{HostSignals, Optimize, Workload};
    use localbench_search::candidate::{ScoreProfile, ScoringContext};
    use localbench_search::space::{resolve_search_space, KvPair, ModelAxes};
    use localx_llama_core::tuner::{Profile, PromptLength, SearchStrategy, TunerEntry};
    use localx_llama_core::Launcher;
    use localx_llama_core::CURRENT_TUNER_VERSION;

    let key = flag_value(args, "--model").ok_or("findbest needs --model <key>")?;
    let context_key = flag_value(args, "--context").unwrap_or_default();
    let requested_mode = parse_mode(flag_value(args, "--mode").as_deref())?;
    let profile = match flag_value(args, "--profile").as_deref() {
        None | Some("pure") => ScoreProfile::Pure,
        Some("balanced") => ScoreProfile::Balanced,
        Some("both") => ScoreProfile::Both,
        Some(other) => return Err(format!("unknown --profile '{other}'")),
    };
    let optimize = match flag_value(args, "--optimize").as_deref() {
        Some("gen") => Optimize::Gen,
        Some("prompt") => Optimize::Prompt,
        Some("both") => Optimize::Both,
        None | Some("coding-agent") => Optimize::CodingAgent,
        Some(other) => return Err(format!("unknown --optimize '{other}'")),
    };
    let budget: i64 = flag_value(args, "--budget")
        .map(|v| v.parse().map_err(|_| format!("bad --budget '{v}'")))
        .transpose()?
        .unwrap_or(30);
    let beam_width: usize = flag_value(args, "--beam-width")
        .map(|value| {
            value
                .parse()
                .map_err(|_| format!("bad --beam-width '{value}'"))
        })
        .transpose()?
        .unwrap_or(DEFAULT_BEAM_WIDTH);
    if !(1..=100).contains(&beam_width) {
        return Err(format!(
            "--beam-width must be between 1 and 100 (got {beam_width})"
        ));
    }
    let runs: usize = flag_value(args, "--runs")
        .map(|v| v.parse().map_err(|_| format!("bad --runs '{v}'")))
        .transpose()?
        .unwrap_or(3);
    // The per-trial startup budget. The default suits a large GGUF off a cold
    // disk; a fast NVMe can lower it to cut the cost of a genuinely stuck trial,
    // and a slow disk can raise it. A crashed process short-circuits this budget
    // regardless (LocalHub#77).
    let startup_timeout_secs: u32 = flag_value(args, "--startup-timeout")
        .map(|v| {
            v.parse()
                .map_err(|_| format!("bad --startup-timeout '{v}'"))
        })
        .transpose()?
        .unwrap_or(300);

    // Bind the real launcher and gate on the contract before trusting it.
    let home = home_dir().ok_or("could not determine the user home directory")?;
    let local_llm = home.join(".local-llm");
    let catalog =
        localbox_launcher::catalog::Catalog::load(&local_llm).map_err(|e| e.to_string())?;
    let launcher = localbox_launcher::launcher::LlamaLauncher::new(
        catalog,
        localbox_launcher::product_version(),
        &home,
        probe_vram_gb(),
    );
    consumer::assert_launcher_usable(&launcher.version(), "1.0.0").map_err(|e| e.to_string())?;

    let mode = resolve_mode(&key, requested_mode, launcher.required_mode(&key))?;
    let def = launcher.model_def(&key).map_err(|e| e.to_string())?;
    let resolved_context = localx_llama_core::model::resolve_context_key(&def, &context_key)
        .map_err(|e| e.to_string())?;
    let context_tokens = localx_llama_core::model::context_value(&def, &resolved_context)
        .map_err(|e| e.to_string())?
        .unwrap_or(0);
    let quant = match flag_value(args, "--quant") {
        Some(q) => {
            localx_llama_core::model::resolve_quant_key(&def, &q).map_err(|e| e.to_string())?
        }
        None => def.quant.clone().unwrap_or_default(),
    };
    // A model that has never been launched has no GGUF on disk yet. Tuning is
    // the step that makes a model usable, so fetch it here — through the
    // launcher's own resumable download, the one a LocalBox launch performs —
    // rather than sending the user to another tool.
    let mut stderr = std::io::stderr();
    let gguf = consumer::ensure_gguf_on_disk(
        &launcher,
        &key,
        (!quant.is_empty()).then_some(quant.as_str()),
        &mut stderr,
    )?;
    let gguf = launcher
        .gguf_path(&def, (!quant.is_empty()).then_some(quant.as_str()))
        .unwrap_or(gguf);

    // Read the GGUF header for the two facts that decide the search: whether the
    // model has experts (dense vs MoE) and how many layers it has. Without this
    // a dense model is misclassified as MoE and swept on the no-op `--n-cpu-moe`
    // axis until it runs out of candidates with no winner (LocalHub#76).
    let shape = localbench::gguf::read_model_shape(&gguf);
    let space = resolve_search_space(
        &ModelAxes {
            n_cpu_moe: def.n_cpu_moe,
            config_n_cpu_moe: None,
            n_gpu_layers: def.n_gpu_layers,
            moe_expert_layers: None,
            spec_type: def.spec_type.clone(),
            spec_draft_n_max: None,
            skip_phases: vec![],
        },
        shape.expert_count,
        shape.block_count,
    );
    let baseline_kv = KvPair {
        k: def.kv_cache_k.clone().unwrap_or_else(|| "q8_0".to_string()),
        v: def.kv_cache_v.clone().unwrap_or_else(|| "q8_0".to_string()),
    };
    let cores = u32::try_from(
        std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(0),
    )
    .unwrap_or(0);
    let ctx = ScoringContext {
        workload: Workload::default(),
        host: HostSignals {
            logical_cores: cores,
        },
        vram_params: Default::default(),
        stability_index: Default::default(),
    };
    let params = TunerParams {
        profile,
        optimize,
        budget,
        baseline_kv,
        mode,
        logical_cores: cores,
        beam_width,
    };
    let settings_params = launcher.settings_launch_params();
    let session_defaults = trial_launch_params(Default::default(), &settings_params);
    let run_id = format!("{}-{}", now_iso(), std::process::id());
    let log_dir = local_llm.join("logs").join("tuner");
    let mut live = LiveRunner::new(
        &launcher,
        TrialTarget {
            key: key.clone(),
            def: def.clone(),
            context_key: resolved_context.clone(),
            mode,
            model_arg_path: gguf.to_string_lossy().to_string(),
            runs,
            port_start: 8091,
            log_dir: log_dir.clone(),
            settings_params,
        },
        startup_timeout_secs,
        &run_id,
    );

    // The persistent trial cache: an interrupted or repeated findbest reuses
    // every decisive measurement, keyed by config signature and invalidated
    // whole when anything shaping a measurement changes.
    use localbench::trial::CachedRunner;
    use localbench_measure::cache::{
        stable_json_hash, Driver, Fingerprint, GgufIdentity, LoadOutcome, TrialCache,
    };
    let paths = StorePaths::new(&local_llm);
    let gguf_meta = std::fs::metadata(&gguf).ok();
    let fingerprint = Fingerprint {
        schema: 2,
        tuner_version: CURRENT_TUNER_VERSION.to_string(),
        measurement_protocol: MEASUREMENT_PROTOCOL.to_string(),
        telemetry_protocol: TELEMETRY_PROTOCOL.to_string(),
        request_shape_hash: chat_request_shape_hash(),
        chat_template_hash: stable_json_hash(&serde_json::json!({
            "parser": def.parser,
            "chat_template": def.chat_template,
            "thinking_policy": def.thinking_policy,
        })),
        model_definition_hash: stable_json_hash(
            &serde_json::to_value(&def).unwrap_or(serde_json::Value::Null),
        ),
        response_schema: RESPONSE_SCHEMA.to_string(),
        session: session_fingerprint(&session_defaults),
        key: key.clone(),
        context_key: resolved_context.clone(),
        context_tokens: u32::try_from(context_tokens).unwrap_or(0),
        mode: mode.as_str().to_string(),
        quant: quant.clone(),
        prompt_length: "short".to_string(),
        prompt_hash: chat_prompt_hash(),
        optimize: flag_value(args, "--optimize").unwrap_or_else(|| "coding-agent".to_string()),
        profile: flag_value(args, "--profile").unwrap_or_else(|| "pure".to_string()),
        search_strategy: "beam".to_string(),
        beam_width: u32::try_from(beam_width).unwrap_or(u32::MAX),
        runs: u32::try_from(runs).unwrap_or(0),
        vram_gb: probe_vram_gb(),
        gpu_names: probe_gpu_names(),
        llamacpp_build: probe_llamacpp_build(&launcher, mode),
        gguf: GgufIdentity {
            path: gguf.to_string_lossy().to_string(),
            size_bytes: gguf_meta.as_ref().map(std::fs::Metadata::len).unwrap_or(0),
            last_write_utc: gguf_meta
                .and_then(|m| m.modified().ok())
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| iso_from_secs(d.as_secs()))
                .unwrap_or_default(),
        },
        allowed_kv_types: vec![],
        stress_targets: vec![],
        stress_min_free_vram_gb: 0.0,
        skip_mtp: false,
        skip_stress_test: true,
    };
    let cache_enabled = !args.iter().any(|a| a == "--no-cache");
    let (cache, cache_outcome) = TrialCache::open(
        paths.trial_cache(&key, &resolved_context),
        &fingerprint,
        cache_enabled,
    );
    match &cache_outcome {
        LoadOutcome::Loaded(count) => eprintln!("trial cache: {count} reusable measurement(s)"),
        LoadOutcome::RecoveredFromBackup(count) => {
            eprintln!("trial cache: {count} measurement(s) recovered from the backup copy");
        }
        LoadOutcome::FingerprintMismatch(fields) => eprintln!(
            "trial cache: starting fresh — the run differs in {}",
            fields.join(", ")
        ),
        LoadOutcome::LoadFailed(error) => {
            eprintln!("trial cache: starting fresh — the file did not parse ({error})");
        }
        LoadOutcome::NoReusableEntries | LoadOutcome::NotFound => {}
    }
    let ledger = localbench::diagnostics::RunLedger::create(
        &log_dir,
        &run_id,
        localbench::diagnostics::DEFAULT_RETAINED_RUNS,
    )
    .map_err(|error| format!("could not create tuner diagnostics: {error}"))?;
    let mut runner = CachedRunner {
        inner: &mut live,
        cache,
        driver: Driver::Server,
        stamp: now_iso,
        ledger: Some(ledger),
        last_trial: None,
    };

    let seeds = localbench_search::seeds::resolve_smart_seeds(
        &space,
        localbench_search::seeds::HostFacts {
            vram_gb: probe_vram_gb(),
            logical_cores: cores,
            available_ram_gb: probe_available_ram_gb(),
            gguf_size_gb: std::fs::metadata(&gguf)
                .map(|m| m.len() as f64 / 1_073_741_824.0)
                .unwrap_or(0.0),
        },
        localbench::tuner::rank_profile(profile),
    );
    let outcome = run_tuner(&mut runner, &space, &seeds, &ctx, &params, &mut |line| {
        eprintln!("{line}");
    });
    let manifest_path = runner
        .manifest_path()
        .map(|path| path.display().to_string())
        .unwrap_or_default();
    let terminal_failure = runner.last_failure_summary();
    if let Err(error) = runner.finish() {
        eprintln!("trial diagnostic run could not be marked complete: {error}");
    }
    let outcome = outcome.ok_or_else(|| {
        let reason = terminal_failure
            .map(|failure| format!("; last outcome: {failure}"))
            .unwrap_or_default();
        format!("no candidate produced and passed a usable measurement{reason}; diagnostics: {manifest_path}")
    })?;

    let mut result = serde_json::json!({
        "model": key,
        "context": resolved_context,
        "mode": mode.as_str(),
        "score": outcome.winner.selected_score,
        "signature": outcome.winner.signature,
        "trials": outcome.trials,
        "verified": outcome.verified,
        "search_strategy": "beam",
        "beam_width": outcome.beam_width,
        "diagnostics": manifest_path,
        // How much of the balanced discount actually fired: "full" only when
        // host telemetry backed every factor; unavailable probes are named in
        // `missing_telemetry` instead of silently receiving full credit.
        "confidence": outcome.winner.score_breakdown.confidence.clone(),
        "missing_telemetry": outcome.winner.score_breakdown.missing_telemetry.clone(),
    });
    if !has_no_save(args) {
        let vram = i64::from(probe_vram_gb());
        let entry = TunerEntry {
            quant: quant.clone(),
            context_key: resolved_context.clone(),
            context_tokens: Some(context_tokens),
            mode,
            vram_gb: vram,
            prompt_length: PromptLength::Short,
            profile: match profile {
                ScoreProfile::Balanced => Profile::Balanced,
                _ => Profile::Pure,
            },
            search_strategy: Some(SearchStrategy::Beam),
            beam_width: Some(i64::try_from(outcome.beam_width).unwrap_or(i64::MAX)),
            score: outcome.winner.selected_score,
            score_unit: "tps".to_string(),
            pure_score: Some(outcome.winner.pure_score),
            args: outcome
                .winner
                .trial
                .as_ref()
                .map(|trial| trial.launch_args.clone())
                .unwrap_or_default(),
            overrides: serde_json::from_value(serde_json::Value::Object(
                outcome.winner.overrides.clone().into_iter().collect(),
            ))
            .unwrap_or_default(),
            measured_at: now_iso(),
            tuner_version: CURRENT_TUNER_VERSION,
            trial_count: Some(outcome.trials as i64),
            gpu_names: None,
            llamacpp_build: None,
        };
        let path = export_best_config(
            &paths.best_config(&key),
            &key,
            &ExportWinner {
                entry,
                vision: false,
            },
            &ExportProvenance {
                localbench_version: VERSION.to_string(),
                launcher_export_version: consumer::LAUNCHER_EXPORT_VERSION,
                measured_at: now_iso(),
            },
        )
        .map_err(|e| e.to_string())?;
        result["saved"] = serde_json::json!(path.display().to_string());
    }
    print_json(&result)?;
    Ok(ExitCode::SUCCESS)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::{iso_from_secs, parse_mode, probe_available_ram_gb, resolve_mode};
    use localx_llama_core::Mode;

    #[test]
    fn prism_mode_aliases_and_required_mode_resolve() {
        assert_eq!(parse_mode(Some("prism")).unwrap(), Some(Mode::PrismMl));
        assert_eq!(parse_mode(Some("prismml")).unwrap(), Some(Mode::PrismMl));
        assert_eq!(
            resolve_mode("bonsai", None, Some(Mode::PrismMl)).unwrap(),
            Mode::PrismMl
        );
        assert_eq!(resolve_mode("ordinary", None, None).unwrap(), Mode::Native);
    }

    #[test]
    fn a_conflicting_explicit_mode_is_rejected() {
        let error =
            resolve_mode("tbonsai27b", Some(Mode::Native), Some(Mode::PrismMl)).unwrap_err();
        assert!(error.contains("requires --mode prism"));
        assert!(error.contains("native"));
    }

    #[test]
    fn iso_from_secs_matches_known_epoch_vectors() {
        // Hand-rolled civil-from-days math gets pinned against known instants
        // (it stamps every cell ledger and best-config export).
        assert_eq!(iso_from_secs(0), "1970-01-01T00:00:00Z");
        assert_eq!(iso_from_secs(86_399), "1970-01-01T23:59:59Z");
        assert_eq!(iso_from_secs(86_400), "1970-01-02T00:00:00Z");
        // Leap day 2000 (a divisible-by-400 century year).
        assert_eq!(iso_from_secs(951_782_400), "2000-02-29T00:00:00Z");
        // 2100 is a non-leap century year: this instant is Feb 28, and one
        // day later must skip Feb 29 entirely.
        assert_eq!(iso_from_secs(4_107_456_000), "2100-02-28T00:00:00Z");
        assert_eq!(
            iso_from_secs(4_107_456_000 + 86_400),
            "2100-03-01T00:00:00Z"
        );
        // A recent reference instant.
        assert_eq!(iso_from_secs(1_735_689_600), "2025-01-01T00:00:00Z");
        // End-of-year rollover.
        assert_eq!(iso_from_secs(1_735_689_599), "2024-12-31T23:59:59Z");
    }

    #[test]
    fn shipped_host_probe_reports_available_ram() {
        let available_ram_gb = probe_available_ram_gb();
        assert!(
            available_ram_gb > 0.0,
            "the seed resolver must receive measured RAM, got {available_ram_gb}GB"
        );
    }
}
