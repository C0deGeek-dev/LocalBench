//! The synthetic long-context stress prompt: deterministic filler that reads
//! like accumulated coding-agent tool context, sized to a token target.

/// Approximate characters per token for sizing the prompt body.
const CHARS_PER_TOKEN: usize = 4;

/// Minimum prompt body size, so tiny targets still exercise the path.
const MIN_TARGET_CHARS: usize = 4096;

/// Realistic accumulated coding-agent context embedded at compile time so
/// measurement never depends on the process working directory.
pub const LONG_CODING_AGENT_FIXTURE: &str =
    include_str!("../../../data/long-coding-agent-fixture.txt");

/// Hermetic fallback used if an explicitly supplied seed or the embedded
/// fixture is ever empty.
pub const SHORT_BENCHMARK_PROMPT: &str =
    "Inspect the repository state, identify the likely runtime failure, and propose a small safe patch with validation.";

/// Build the stress prompt for a token target from a seed text (the long
/// prompt fixture, or the short benchmark prompt as a fallback): a header,
/// repeated context blocks until ~4 chars/token of the target, and a final
/// task line.
#[must_use]
pub fn stress_prompt(target_tokens: u32, seed: &str) -> String {
    let seed = match seed.trim() {
        "" => SHORT_BENCHMARK_PROMPT,
        seed => seed,
    };
    let target_chars = MIN_TARGET_CHARS.max(target_tokens as usize * CHARS_PER_TOKEN);
    let mut prompt = String::with_capacity(target_chars + 4096);
    prompt.push_str("You are running a LocalBench long-context coding-agent soak test.\n");
    prompt.push_str(
        "Analyze the accumulated repository context and return a concise stability diagnosis.\n\n",
    );

    let mut block = 0usize;
    while prompt.len() < target_chars {
        block += 1;
        prompt.push_str(&format!("## Accumulated tool context block {block}\n"));
        prompt.push_str(seed);
        prompt.push_str("\n\nSynthetic log tail:\n");
        prompt.push_str("slot update_slots: prompt processing progress, n_tokens += 1024\n");
        prompt.push_str(
            "context checkpoint created during processing; cache reuse and prompt rebuild remain active\n\n",
        );
    }

    prompt.push_str(
        "Final task: identify runtime stability risks, VRAM headroom risks, and whether this \
         profile should be saved for AutoBest.\n",
    );
    prompt
}

/// Build the shipped coding-agent measurement prompt from the embedded real
/// fixture (and the short fallback enforced by [`stress_prompt`]).
#[must_use]
pub fn coding_agent_stress_prompt(target_tokens: u32) -> String {
    stress_prompt(target_tokens, LONG_CODING_AGENT_FIXTURE)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEED: &str = "fn main() { println!(\"hello\"); } // repository excerpt";

    #[test]
    fn prompt_scales_with_the_token_target() {
        let small = stress_prompt(1_024, SEED);
        let big = stress_prompt(32_768, SEED);
        assert!(small.len() >= MIN_TARGET_CHARS);
        assert!(big.len() >= 32_768 * CHARS_PER_TOKEN);
        assert!(big.len() > small.len() * 4);
    }

    #[test]
    fn prompt_is_deterministic_and_framed() {
        let a = coding_agent_stress_prompt(8_192);
        let b = coding_agent_stress_prompt(8_192);
        assert_eq!(a, b, "same target + seed → identical prompt");
        assert!(a.starts_with("You are running a LocalBench long-context coding-agent soak test."));
        assert!(a.contains("## Accumulated tool context block 1"));
        assert!(a.contains("Support reliable llama.cpp benchmarking for a local MoE model"));
        assert!(a.trim_end().ends_with("saved for AutoBest."));
    }

    #[test]
    fn an_empty_seed_uses_the_short_benchmark_fallback() {
        let prompt = stress_prompt(512, "  ");
        assert!(prompt.contains(SHORT_BENCHMARK_PROMPT));
    }
}
