//! llama.cpp's memory fitter as the tuner's [`FitOracle`].
//!
//! For each candidate memory shape the oracle builds the exact server argv a
//! trial would launch, hands the memory-relevant part to the build's own
//! `llama-fit-params`, and caches the placement per shape. A placement only
//! bounds the search; every result the tuner keeps still comes from a server
//! measurement.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

use localbench_scoring::score::Overrides;
use localbench_search::overrides::candidate_signature;
use localx_llama_core::args::build_llama_server_args;
use localx_llama_core::fit::{fit_params_args, FitError, FitPlacement};
use localx_llama_core::Launcher;
use localx_llama_runtime::fit::{run_fit_params, FitRunError};

use crate::trial::{candidate_launch_params, TrialTarget};
use crate::tuner::FitOracle;

/// Free VRAM the fitter keeps per device: llama.cpp's own `--fit` default.
/// The VRAM-fit phase probes past it, so this sets where the search starts,
/// not a hard limit.
pub const DEFAULT_FIT_MARGIN_MIB: u32 = 1024;

/// How long one fitter call may take. It reads metadata only and answers in
/// seconds even for 100 GB models; this bounds a stuck build.
const FIT_TIMEOUT: Duration = Duration::from_secs(120);

/// Keys that choose a placement or a CPU thread count — they never change
/// where the model fits, so they are left out of the cache key.
const NON_SHAPE_KEYS: &[&str] = &["NCpuMoe", "NGpuLayers", "Threads", "ThreadsBatch"];

/// The live oracle over one tuning target.
pub struct LiveFitOracle<'a> {
    launcher: &'a dyn Launcher,
    target: TrialTarget,
    binary: PathBuf,
    margin_mib: u32,
    cache: HashMap<String, Result<FitPlacement, FitRunError>>,
}

impl<'a> LiveFitOracle<'a> {
    /// An oracle for `target`, when its engine ships `llama-fit-params`.
    #[must_use]
    pub fn new(launcher: &'a dyn Launcher, target: TrialTarget, margin_mib: u32) -> Option<Self> {
        let binary = launcher.fit_params_binary(target.mode)?;
        Some(Self {
            launcher,
            target,
            binary,
            margin_mib,
            cache: HashMap::new(),
        })
    }
}

/// The part of a candidate that decides how much memory it needs.
fn shape_key(overrides: &Overrides) -> String {
    let mut shape = overrides.clone();
    for key in NON_SHAPE_KEYS {
        shape.remove(*key);
    }
    candidate_signature(&shape)
}

impl LiveFitOracle<'_> {
    /// The fitter's answer for a candidate's memory shape, asked once per shape.
    fn answer(&mut self, overrides: &Overrides) -> Result<FitPlacement, FitRunError> {
        let key = shape_key(overrides);
        if let Some(known) = self.cache.get(&key) {
            return known.clone();
        }
        let answer = candidate_launch_params(self.launcher, &self.target, overrides)
            .ok()
            .and_then(|params| {
                build_llama_server_args(
                    &self.target.def,
                    &self.target.context_key,
                    self.target.mode,
                    &self.target.model_arg_path,
                    0,
                    &params,
                )
                .ok()
            })
            .map_or(Err(FitRunError::DidNotRun), |argv| {
                let args = fit_params_args(&argv, self.margin_mib);
                run_fit_params(&self.binary, &args, FIT_TIMEOUT)
            });
        self.cache.insert(key, answer.clone());
        answer
    }
}

impl FitOracle for LiveFitOracle<'_> {
    fn placement(&mut self, overrides: &Overrides) -> Option<FitPlacement> {
        self.answer(overrides).ok()
    }

    fn rejection(&mut self, overrides: &Overrides) -> Option<String> {
        match self.answer(overrides) {
            Err(FitRunError::Fit(FitError::Failed(reason))) => Some(reason),
            _ => None,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use localbench_search::overrides::overrides_of;
    use serde_json::json;

    #[test]
    fn placement_and_threads_do_not_change_the_memory_shape() {
        let a = overrides_of(&[
            ("KvK", json!("q8_0")),
            ("NCpuMoe", json!(35)),
            ("Threads", json!(12)),
        ]);
        let b = overrides_of(&[("KvK", json!("q8_0")), ("NCpuMoe", json!(38))]);
        let c = overrides_of(&[("KvK", json!("f16")), ("NCpuMoe", json!(35))]);
        assert_eq!(shape_key(&a), shape_key(&b));
        assert_ne!(shape_key(&a), shape_key(&c));
    }
}
