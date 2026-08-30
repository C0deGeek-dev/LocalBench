//! The recorded arm matrix and the arm-isolation contract.
//!
//! The model is pinned across every arm, so the whole delta is harness
//! quality. An arm's behaviour is therefore a **recorded config, not a label**:
//! these emitters produce the canonical `.localpilot.toml` / `.localmind.toml`
//! that *defines* each arm — the exact config an operator stages before a run.
//! The live driver differentiates arms by the `--arm`/`--verify`/`--learn`
//! flags it hands to `localpilot eval` (the harness side owns applying an arm's
//! effective config); these emitters are not auto-applied by it. What the
//! isolation contract enforces is that a baseline arm's *declared* effective
//! config carries no harness behaviour through an ambient channel — a silent
//! leak would collapse every harness-vs-baseline delta toward zero.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// The LocalPilot-driven arms whose configs are emitted here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HarnessArm {
    /// The original agentic harness: no explicit rails or verify (built-in
    /// safety defaults still apply) — the confounded arm.
    Full,
    /// Rails matched to Claude Code's turn cap plus the verify-before-done
    /// gate, so the head-to-head is fair rather than unbounded + unverified.
    Fair,
    /// `full` plus only the verify gate — the +verify side of the ablation.
    Verify,
    /// `fair` plus persistent machine-wide learning shared across exercises.
    Warm,
}

impl HarnessArm {
    /// The arm's config label.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            HarnessArm::Full => "full",
            HarnessArm::Fair => "fair",
            HarnessArm::Verify => "verify",
            HarnessArm::Warm => "warm",
        }
    }

    fn wants_rails(self) -> bool {
        matches!(self, HarnessArm::Fair | HarnessArm::Warm)
    }

    fn wants_verify(self) -> bool {
        matches!(
            self,
            HarnessArm::Fair | HarnessArm::Verify | HarnessArm::Warm
        )
    }

    /// Only the warm arm closes out into accumulated memory.
    #[must_use]
    pub fn learns(self) -> bool {
        matches!(self, HarnessArm::Warm)
    }
}

/// Knobs for the emitted `.localpilot.toml`.
#[derive(Debug, Clone)]
pub struct ArmConfigParams {
    pub model: String,
    pub base_url: String,
    pub context_window: u32,
    /// Fairness anchor: ~ Claude Code `--max-turns 40` × a few tool calls.
    pub tool_call_budget_max: u32,
    pub turn_timeout_secs: u32,
    /// Per-language verify command override (else the gate detects the stack).
    pub verify_command: Option<String>,
}

impl ArmConfigParams {
    /// Defaults for a pinned local model.
    #[must_use]
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            base_url: "http://localhost:11435/v1".to_string(),
            context_window: 32_000,
            tool_call_budget_max: 160,
            turn_timeout_secs: 600,
            verify_command: None,
        }
    }
}

/// Emit the `.localpilot.toml` text for one arm. Pure, so the exact config an
/// arm runs with is unit-testable without launching a binary.
#[must_use]
pub fn localpilot_arm_config(arm: HarnessArm, params: &ArmConfigParams) -> String {
    let mut lines = vec![
        "[provider]".to_string(),
        "default = \"local\"".to_string(),
        "[providers.local]".to_string(),
        "kind = \"openai-compatible\"".to_string(),
        format!("base_url = \"{}\"", params.base_url),
        format!("model = \"{}\"", params.model),
        format!("context_window = {}", params.context_window),
    ];
    // `full` ships no explicit `[harness]` block (the original arm).
    if arm.wants_rails() || arm.wants_verify() {
        lines.push(String::new());
        lines.push("[harness]".to_string());
        if arm.wants_rails() {
            lines.push(format!(
                "tool_call_budget_max = {}",
                params.tool_call_budget_max
            ));
            lines.push(format!("turn_timeout_secs = {}", params.turn_timeout_secs));
        }
        if arm.wants_verify() {
            lines.push("verify_before_done = true".to_string());
            if let Some(command) = &params.verify_command {
                lines.push(format!("verify_command = \"{command}\""));
            }
        }
    }
    lines.join("\n") + "\n"
}

/// The `.localmind.toml` a clean-room MEASUREMENT arm runs with: learning
/// **explicitly OFF**. Learning is on by default, so a capability-measurement
/// arm (baseline / full / verify / fair) must disable it — otherwise the
/// solver reads accumulated machine-wide memory and the deltas are no longer
/// clean-room. Only the warm arm opts back in.
#[must_use]
pub fn localmind_measurement_config() -> String {
    "[learning]\nenabled = false\n".to_string()
}

/// Knobs for the warm arm's `.localmind.toml`.
#[derive(Debug, Clone)]
pub struct WarmConfigParams {
    /// The persistent store shared across every exercise. Must be absolute —
    /// per-exercise workspaces are wiped, so a relative root would silently
    /// scatter the "accumulated" memory.
    pub global_memory_root: String,
    pub model: String,
    /// The model's OpenAI-compatible endpoint for closeout extraction (the
    /// chat server, NOT the no-think proxy the solver uses).
    pub chat_base_url: String,
    /// The CPU embedding server on its own port, so it adds zero GPU VRAM and
    /// the chat model stays byte-identical across arms.
    pub embed_base_url: String,
    pub embed_model: String,
}

impl WarmConfigParams {
    /// Defaults for a pinned local model over the standard local endpoints.
    #[must_use]
    pub fn new(global_memory_root: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            global_memory_root: global_memory_root.into(),
            model: model.into(),
            chat_base_url: "http://127.0.0.1:8080".to_string(),
            embed_base_url: "http://127.0.0.1:8090".to_string(),
            embed_model: "qwen3-embedding-0.6b".to_string(),
        }
    }
}

/// An arm-configuration failure.
#[derive(Debug, thiserror::Error)]
pub enum ArmConfigError {
    /// The warm arm's shared store must be an absolute path.
    #[error("the warm global memory root must be an absolute path (got '{0}')")]
    RelativeGlobalRoot(String),
    /// A baseline arm carries harness behaviour; names the leaked channels.
    #[error(
        "baseline arm '{arm}' is contaminated by harness behaviour via: {channels}. \
         A baseline must run with the harness OFF (no LOCALPILOT_* env, no ambient \
         config file, no enabled plugins/skills, empty system prompt, retrieval off) \
         or every harness-vs-baseline delta is invalid."
    )]
    ContaminatedBaseline { arm: String, channels: String },
}

/// Emit the `.localmind.toml` the warm/teaching arm runs with: learning on,
/// the machine-wide global scope opted in at a persistent shared root,
/// auto-accepted review (so candidates become injectable memory unattended),
/// model-backed closeout extraction, embedding-backed semantic dedup, and the
/// retrieval rerank opt-in. The warm arm is the AUTO-ACCEPT upper bound of
/// "smarter as you use it" — real use is human-reviewed, so warm reads as the
/// optimistic ceiling, not the realistic curve.
///
/// # Errors
/// Returns [`ArmConfigError::RelativeGlobalRoot`] when the shared store path
/// is not absolute.
pub fn localmind_warm_config(params: &WarmConfigParams) -> Result<String, ArmConfigError> {
    let root = std::path::Path::new(&params.global_memory_root);
    if !root.is_absolute() {
        return Err(ArmConfigError::RelativeGlobalRoot(
            params.global_memory_root.clone(),
        ));
    }
    Ok(format!(
        "[learning]\n\
         enabled = true\n\
         allowed_scopes = [\"project\", \"global_user\"]\n\
         global_memory_root = '{}'\n\
         \n\
         [inference]\n\
         chat_base_url = \"{}\"\n\
         chat_model = \"{}\"\n\
         embedding_base_url = \"{}\"\n\
         embedding_model = \"{}\"\n\
         \n\
         [inference.features]\n\
         embeddings = true\n\
         \n\
         [review]\n\
         mode = \"automatic\"\n\
         semantic_dedup = true\n\
         \n\
         [retrieval]\n\
         rerank = true\n",
        params.global_memory_root,
        params.chat_base_url,
        params.model,
        params.embed_base_url,
        params.embed_model,
    ))
}

/// One row of the recorded arm matrix: the config knobs that distinguish an
/// arm, so a published number can always be traced to the exact configuration
/// it was produced under.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArmDefinition {
    pub arm: String,
    pub driver: String,
    pub rails: bool,
    pub verify: bool,
    pub learning: bool,
    pub note: String,
}

/// The recorded arm matrix for the harness sweep.
#[must_use]
pub fn harness_arm_definitions() -> Vec<ArmDefinition> {
    let def = |arm: &str, driver: &str, rails: bool, verify: bool, learning: bool, note: &str| {
        ArmDefinition {
            arm: arm.to_string(),
            driver: driver.to_string(),
            rails,
            verify,
            learning,
            note: note.to_string(),
        }
    };
    vec![
        def(
            "baseline",
            "raw-model",
            false,
            false,
            false,
            "single-shot model, no harness/tools",
        ),
        def(
            "full",
            "localpilot",
            false,
            false,
            false,
            "agentic harness, no explicit rails or verify (built-in safety defaults apply)",
        ),
        def(
            "fair",
            "localpilot",
            true,
            true,
            false,
            "rails matched to CC --max-turns + verify on (the fair LP-vs-CC arm)",
        ),
        def(
            "verify",
            "localpilot",
            false,
            true,
            false,
            "full + verify only (the +verify side of the ablation)",
        ),
        def(
            "warm",
            "localpilot",
            true,
            true,
            true,
            "fair + persistent global learning shared across exercises",
        ),
        def(
            "claude-code",
            "claude-code",
            true,
            false,
            false,
            "Claude Code harness driving the same pinned model (--max-turns 40)",
        ),
    ]
}

/// A raw arm config as supplied by a runner.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RawArmConfig {
    /// Explicit baseline flag; when absent, an arm literally named `baseline`
    /// is treated as the baseline.
    pub is_baseline: Option<bool>,
    pub env: BTreeMap<String, String>,
    pub config_file: Option<String>,
    pub plugins: Vec<String>,
    pub system_prompt: Option<String>,
    pub retrieval: bool,
}

/// The normalized effective configuration the isolation contract reasons over.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArmEffectiveConfig {
    pub arm: String,
    pub is_baseline: bool,
    pub env: BTreeMap<String, String>,
    pub config_file: String,
    pub plugins: Vec<String>,
    pub system_prompt: String,
    pub retrieval: bool,
}

/// Normalize a raw arm config into its inspectable effective config.
#[must_use]
pub fn arm_effective_config(arm: &str, config: &RawArmConfig) -> ArmEffectiveConfig {
    ArmEffectiveConfig {
        arm: arm.to_string(),
        is_baseline: config
            .is_baseline
            .unwrap_or_else(|| arm.eq_ignore_ascii_case("baseline")),
        env: config.env.clone(),
        config_file: config.config_file.clone().unwrap_or_default(),
        plugins: config
            .plugins
            .iter()
            .filter(|p| !p.trim().is_empty())
            .cloned()
            .collect(),
        system_prompt: config.system_prompt.clone().unwrap_or_default(),
        retrieval: config.retrieval,
    }
}

/// The harness channels that leak into an effective config (empty when
/// clean). Meaningful only for a baseline arm; a harness arm is *meant* to
/// carry these.
#[must_use]
pub fn arm_isolation_leaks(config: &ArmEffectiveConfig) -> Vec<&'static str> {
    let mut leaks = Vec::new();
    if config
        .env
        .keys()
        .any(|k| k.to_ascii_uppercase().starts_with("LOCALPILOT_"))
    {
        leaks.push("env");
    }
    if !config.config_file.trim().is_empty() {
        leaks.push("configFile");
    }
    if !config.plugins.is_empty() {
        leaks.push("plugins");
    }
    if !config.system_prompt.trim().is_empty() {
        leaks.push("systemPrompt");
    }
    if config.retrieval {
        leaks.push("retrieval");
    }
    leaks
}

/// One-token isolation provenance for a report row: `clean`,
/// `CONTAMINATED: <channels>`, or `n/a (harness arm)`.
#[must_use]
pub fn arm_isolation_summary(config: &ArmEffectiveConfig) -> String {
    if !config.is_baseline {
        return "n/a (harness arm)".to_string();
    }
    let leaks = arm_isolation_leaks(config);
    if leaks.is_empty() {
        "clean".to_string()
    } else {
        format!("CONTAMINATED: {}", leaks.join(","))
    }
}

/// Refuse a baseline arm that carries any harness behaviour, naming the exact
/// leaked channel(s) — a contaminated baseline fails fast at setup rather than
/// producing a quietly-invalid delta. A harness arm always passes.
///
/// # Errors
/// Returns [`ArmConfigError::ContaminatedBaseline`] naming the channels.
pub fn assert_arm_isolation(
    arm: &str,
    config: &RawArmConfig,
) -> Result<ArmEffectiveConfig, ArmConfigError> {
    let effective = arm_effective_config(arm, config);
    if !effective.is_baseline {
        return Ok(effective);
    }
    let leaks = arm_isolation_leaks(&effective);
    if leaks.is_empty() {
        Ok(effective)
    } else {
        Err(ArmConfigError::ContaminatedBaseline {
            arm: arm.to_string(),
            channels: leaks.join(", "),
        })
    }
}

/// The cheap-prompt control arm: a one-line system-prompt nudge, included as a
/// distinct arm so the harness must beat a cheap instruction, not only an
/// empty baseline. It is NOT the baseline (it carries a prompt), so it passes
/// the isolation contract as a harness arm.
#[must_use]
pub fn control_arm() -> ArmEffectiveConfig {
    ArmEffectiveConfig {
        arm: "control-prompt".to_string(),
        is_baseline: false,
        env: BTreeMap::new(),
        config_file: String::new(),
        plugins: Vec::new(),
        system_prompt: "Prefer the smallest correct change that fully solves the task; \
                        do not add abstraction, configuration, or features the task did \
                        not ask for."
            .to_string(),
        retrieval: false,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn parse(text: &str) -> toml::Value {
        text.parse::<toml::Value>().expect("valid TOML")
    }

    #[test]
    fn arm_configs_round_trip_with_the_right_knobs() {
        let params = ArmConfigParams::new("apex-i-quality");
        // full: provider only, no [harness] block.
        let full = parse(&localpilot_arm_config(HarnessArm::Full, &params));
        assert_eq!(
            full["providers"]["local"]["model"].as_str(),
            Some("apex-i-quality")
        );
        assert!(full.get("harness").is_none(), "full ships no rails/verify");
        // fair: rails AND verify.
        let fair = parse(&localpilot_arm_config(HarnessArm::Fair, &params));
        assert_eq!(
            fair["harness"]["tool_call_budget_max"].as_integer(),
            Some(160)
        );
        assert_eq!(fair["harness"]["turn_timeout_secs"].as_integer(), Some(600));
        assert_eq!(fair["harness"]["verify_before_done"].as_bool(), Some(true));
        // verify: the gate only, no rails.
        let verify = parse(&localpilot_arm_config(HarnessArm::Verify, &params));
        assert_eq!(
            verify["harness"]["verify_before_done"].as_bool(),
            Some(true)
        );
        assert!(verify["harness"].get("tool_call_budget_max").is_none());
        // warm matches fair's harness block.
        let warm = parse(&localpilot_arm_config(HarnessArm::Warm, &params));
        assert_eq!(
            warm["harness"]["tool_call_budget_max"].as_integer(),
            Some(160)
        );
        assert!(HarnessArm::Warm.learns() && !HarnessArm::Fair.learns());
    }

    #[test]
    fn a_verify_command_override_is_carried() {
        let mut params = ArmConfigParams::new("m");
        params.verify_command = Some("ctest --output-on-failure".to_string());
        let verify = parse(&localpilot_arm_config(HarnessArm::Verify, &params));
        assert_eq!(
            verify["harness"]["verify_command"].as_str(),
            Some("ctest --output-on-failure")
        );
    }

    #[test]
    fn measurement_arms_explicitly_disable_learning() {
        let config = parse(&localmind_measurement_config());
        assert_eq!(config["learning"]["enabled"].as_bool(), Some(false));
    }

    #[test]
    fn warm_config_carries_the_full_learning_stack() {
        let root = if cfg!(windows) {
            r"C:\bench\warm-global\memory"
        } else {
            "/bench/warm-global/memory"
        };
        let text = localmind_warm_config(&WarmConfigParams::new(root, "apex")).expect("emit");
        let config = parse(&text);
        assert_eq!(config["learning"]["enabled"].as_bool(), Some(true));
        let scopes: Vec<&str> = config["learning"]["allowed_scopes"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(toml::Value::as_str)
            .collect();
        assert!(scopes.contains(&"global_user"));
        assert_eq!(
            config["learning"]["global_memory_root"].as_str(),
            Some(root)
        );
        assert_eq!(config["review"]["mode"].as_str(), Some("automatic"));
        assert_eq!(config["review"]["semantic_dedup"].as_bool(), Some(true));
        assert_eq!(config["retrieval"]["rerank"].as_bool(), Some(true));
        // The embed server rides its own port so the chat model stays pinned.
        assert_eq!(
            config["inference"]["embedding_base_url"].as_str(),
            Some("http://127.0.0.1:8090")
        );
        assert_eq!(config["inference"]["chat_model"].as_str(), Some("apex"));
    }

    #[test]
    fn warm_config_refuses_a_relative_global_root() {
        assert!(matches!(
            localmind_warm_config(&WarmConfigParams::new("warm-global/memory", "m")),
            Err(ArmConfigError::RelativeGlobalRoot(_))
        ));
    }

    #[test]
    fn the_arm_matrix_is_recorded() {
        let defs = harness_arm_definitions();
        let warm = defs.iter().find(|d| d.arm == "warm").unwrap();
        assert!(warm.rails && warm.verify && warm.learning);
        let baseline = defs.iter().find(|d| d.arm == "baseline").unwrap();
        assert!(!baseline.rails && !baseline.verify && !baseline.learning);
        assert_eq!(defs.len(), 6);
    }

    #[test]
    fn a_clean_baseline_passes_isolation() {
        let effective = assert_arm_isolation("baseline", &RawArmConfig::default()).expect("clean");
        assert!(effective.is_baseline);
        assert_eq!(arm_isolation_summary(&effective), "clean");
    }

    #[test]
    fn a_contaminated_baseline_is_refused_naming_the_channels() {
        let mut config = RawArmConfig {
            retrieval: true,
            ..RawArmConfig::default()
        };
        config
            .env
            .insert("LOCALPILOT_CONFIG".to_string(), "x".to_string());
        let err = assert_arm_isolation("baseline", &config).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("env") && message.contains("retrieval"));
        // The summary renders the contamination for a report row.
        let effective = arm_effective_config("baseline", &config);
        assert_eq!(
            arm_isolation_summary(&effective),
            "CONTAMINATED: env,retrieval"
        );
    }

    #[test]
    fn a_harness_arm_may_carry_harness_behaviour() {
        let config = RawArmConfig {
            retrieval: true,
            system_prompt: Some("be terse".to_string()),
            ..RawArmConfig::default()
        };
        let effective = assert_arm_isolation("full", &config).expect("harness arm");
        assert!(!effective.is_baseline);
        assert_eq!(arm_isolation_summary(&effective), "n/a (harness arm)");
        // An empty-plugins list is not a leak (the absent-config footgun).
        let sparse = RawArmConfig {
            plugins: vec![String::new()],
            ..RawArmConfig::default()
        };
        assert!(assert_arm_isolation("baseline", &sparse).is_ok());
    }

    #[test]
    fn the_control_arm_is_a_harness_arm_with_only_a_prompt() {
        let control = control_arm();
        assert!(!control.is_baseline);
        assert!(!control.system_prompt.is_empty());
        assert!(control.plugins.is_empty() && !control.retrieval);
        assert_eq!(arm_isolation_summary(&control), "n/a (harness arm)");
    }
}
