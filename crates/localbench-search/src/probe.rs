//! Probe and stress-target policy: where the long-context probe and the
//! soak/guard validations measure.

use localbench_scoring::stats::round_dp;

/// The probe floor: below this context, the regular search prompt is already
/// at-or-near probe scale, so no probe runs.
const PROBE_FLOOR_TOKENS: u32 = 16_384;

/// The probe cap (96k): probing above this costs more than it informs.
const PROBE_CAP_TOKENS: u32 = 98_304;

/// Pick the prompt-token target for the long-context probe whose pp/tg anchor
/// the pure score. Drives the search away from configs that look fast at 2k
/// tokens but thrash at the configured context. Returns 0 when probing makes
/// no sense (very small ctx).
///
/// Policy: **ctx/2, capped at 96k, floored at 16k; ctx ≤ 16k → no probe.**
#[must_use]
pub fn long_context_probe_target(context_tokens: u32) -> u32 {
    if context_tokens <= PROBE_FLOOR_TOKENS {
        return 0;
    }
    (context_tokens / 2).clamp(PROBE_FLOOR_TOKENS, PROBE_CAP_TOKENS)
}

/// The prompt-token targets for the long-context stress soak. Explicit
/// requests win; otherwise the soak **auto-enables** only for the long-prompt
/// coding-agent workload at 64k+ context, tiered so a bigger context soaks at
/// a bigger prompt.
#[must_use]
pub fn stress_prompt_token_targets(
    context_tokens: u32,
    long_prompt_profile: bool,
    coding_agent: bool,
    requested: &[u32],
) -> Vec<u32> {
    if !requested.is_empty() {
        let mut targets: Vec<u32> = requested.iter().copied().filter(|t| *t > 0).collect();
        targets.sort_unstable();
        targets.dedup();
        return targets;
    }

    if !long_prompt_profile || !coding_agent || context_tokens < 65_536 {
        return Vec::new();
    }

    if context_tokens >= 196_608 {
        vec![65_536]
    } else if context_tokens >= 131_072 {
        vec![32_768]
    } else {
        vec![16_384]
    }
}

/// The guard-phase targets: a cheap re-check at the smallest soak target,
/// capped at 16k so the guard stays fast.
#[must_use]
pub fn stress_guard_token_targets(stress_targets: &[u32]) -> Vec<u32> {
    let Some(first) = stress_targets.iter().copied().filter(|t| *t > 0).min() else {
        return Vec::new();
    };
    vec![first.min(16_384)]
}

/// The default minimum-free-VRAM floor for the stress soak: 5% of the card,
/// clamped to `[0.25, 1.0]` GB; `0.5` GB when the card size is unknown.
#[must_use]
pub fn default_stress_min_free_vram_gb(vram_gb: u32) -> f64 {
    if vram_gb == 0 {
        return 0.5;
    }
    round_dp((f64::from(vram_gb) * 0.05).clamp(0.25, 1.0), 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_target_is_half_ctx_with_floor_and_cap() {
        // ctx at or under 16k: no probe.
        assert_eq!(long_context_probe_target(16_384), 0);
        assert_eq!(long_context_probe_target(8_192), 0);
        // Just above the floor: half would undershoot, so the floor holds.
        assert_eq!(long_context_probe_target(16_385), 16_384);
        assert_eq!(long_context_probe_target(24_576), 16_384);
        // Mid range: ctx/2.
        assert_eq!(long_context_probe_target(65_536), 32_768);
        assert_eq!(long_context_probe_target(131_072), 65_536);
        // Huge ctx: capped at 96k.
        assert_eq!(long_context_probe_target(262_144), 98_304);
    }

    #[test]
    fn stress_targets_auto_enable_only_for_long_coding_agent_at_64k_plus() {
        assert!(stress_prompt_token_targets(65_536, false, true, &[]).is_empty());
        assert!(stress_prompt_token_targets(65_536, true, false, &[]).is_empty());
        assert!(stress_prompt_token_targets(65_535, true, true, &[]).is_empty());
        assert_eq!(
            stress_prompt_token_targets(65_536, true, true, &[]),
            vec![16_384]
        );
        assert_eq!(
            stress_prompt_token_targets(131_072, true, true, &[]),
            vec![32_768]
        );
        assert_eq!(
            stress_prompt_token_targets(262_144, true, true, &[]),
            vec![65_536]
        );
    }

    #[test]
    fn explicit_stress_targets_win_and_are_sorted_deduped() {
        assert_eq!(
            stress_prompt_token_targets(4_096, false, false, &[32_768, 0, 16_384, 32_768]),
            vec![16_384, 32_768]
        );
    }

    #[test]
    fn guard_targets_take_the_smallest_soak_capped_at_16k() {
        assert!(stress_guard_token_targets(&[]).is_empty());
        assert_eq!(stress_guard_token_targets(&[32_768, 65_536]), vec![16_384]);
        assert_eq!(stress_guard_token_targets(&[8_192, 32_768]), vec![8_192]);
    }

    #[test]
    fn stress_vram_floor_scales_with_the_card() {
        assert_eq!(default_stress_min_free_vram_gb(0), 0.5);
        assert_eq!(default_stress_min_free_vram_gb(4), 0.25);
        assert_eq!(default_stress_min_free_vram_gb(12), 0.6);
        assert_eq!(default_stress_min_free_vram_gb(24), 1.0);
        assert_eq!(default_stress_min_free_vram_gb(48), 1.0);
    }
}
