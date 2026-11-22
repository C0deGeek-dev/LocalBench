//! Model-specific benchmark runtime supplied by the installed LocalBox.

use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

use localx_llama_core::{ModelDef, ServerCapabilities};
use localx_llama_runtime::{help::read_help_output, tool::run_tool};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Handoff {
    schema: u32,
    server: PathBuf,
    model: ModelDef,
    batch_size: i64,
    ubatch_size: i64,
}

/// Runtime and launch defaults bound to a single tuning target.
#[derive(Debug, Clone)]
pub struct BenchmarkRuntime {
    pub server: PathBuf,
    pub capabilities: ServerCapabilities,
    pub build_identity: String,
    pub batch_size: i64,
    pub ubatch_size: i64,
}

impl BenchmarkRuntime {
    pub fn apply_batch_defaults(&self, params: &mut localx_llama_core::args::LaunchParams) {
        params.batch_size.get_or_insert(self.batch_size);
        params.ubatch_size.get_or_insert(self.ubatch_size);
    }

    /// Ask LocalBox to provision the same runtime it would use to serve this model.
    pub fn resolve(key: &str) -> Result<(Self, ModelDef), String> {
        let executable = format!("localbox{}", std::env::consts::EXE_SUFFIX);
        let program = std::env::current_exe()
            .ok()
            .map(|path| path.with_file_name(&executable))
            .filter(|path| path.is_file())
            .unwrap_or_else(|| PathBuf::from(executable));
        let output = Command::new(program)
            .args(["bench-runtime", key])
            // Stream installer progress and errors without mixing them into JSON.
            .stderr(std::process::Stdio::inherit())
            .output()
            .map_err(|error| {
                format!("could not ask LocalBox for the benchmark runtime: {error}")
            })?;
        if !output.status.success() {
            return Err(format!(
                "LocalBox did not prepare a benchmark runtime for '{key}' ({}). A LocalBox build with bench-runtime support is required",
                output.status
            ));
        }
        let handoff: Handoff = serde_json::from_slice(&output.stdout)
            .map_err(|error| format!("invalid LocalBox benchmark runtime response: {error}"))?;
        if handoff.schema != 1 || !handoff.server.is_absolute() || !handoff.server.is_file() {
            return Err("unsupported LocalBox benchmark runtime response".to_string());
        }
        let timeout = Duration::from_secs(10);
        let help = read_help_output(&handoff.server, timeout).ok_or_else(|| {
            format!(
                "could not read benchmark flags from {}",
                handoff.server.display()
            )
        })?;
        let version = run_tool(&handoff.server, &["--version".into()], timeout)
            .filter(|output| output.success)
            .map(|output| format!("{}\n{}", output.stdout.trim(), output.stderr.trim()))
            .ok_or_else(|| format!("could not identify {}", handoff.server.display()))?;
        let build_identity = format!("{}:{}", handoff.server.display(), version.trim());
        Ok((
            Self {
                server: handoff.server,
                capabilities: ServerCapabilities::from_help(&help),
                build_identity,
                batch_size: handoff.batch_size,
                ubatch_size: handoff.ubatch_size,
            },
            handoff.model,
        ))
    }
}
