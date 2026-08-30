//! Search strategy for the LocalBench auto-tuner: how candidates are
//! generated, scored into a beam retained between live tuner phases,
//! validated, and adopted.
//!
//! Deterministic and I/O-free — measurement happens elsewhere; this crate owns
//! the *decisions*: which configs to try (candidate spaces, smart seeds, edge
//! and recovery ladders), which to keep (beam selection with its tie-breaks),
//! and which to adopt (cross-profile validated-soak adoption, MTP-regime
//! merge). The following rules are pinned by golden tests but are not called
//! by the current tuner:
//!
//! - **Library-only / unwired:** an MTP tie keeps MTP on — the non-MTP regime
//!   wins only by strictly beating it ([`regime::merge_mtp_regimes`]).
//! - **Library-only / unwired:** a passed-soak sibling beats a higher-scoring
//!   degraded winner
//!   ([`candidate::cross_profile_winner`]).
//! - **Library-only / unwired:** the long-context probe targets ctx/2, floored
//!   at 16k and capped at 96k
//!   ([`probe::long_context_probe_target`]).
//! - **Library-only / unwired:** stress soaks would auto-enable for long-prompt
//!   coding-agent runs at 64k+ context
//!   ([`probe::stress_prompt_token_targets`]).

#![forbid(unsafe_code)]

pub mod candidate;
pub mod overrides;
pub mod probe;
pub mod regime;
pub mod seeds;
pub mod space;

#[cfg(test)]
mod tests {
    #[test]
    fn crate_docs_disclose_unwired_search_rules_consistently_with_tuning_guide() {
        let source = include_str!("lib.rs");
        let header = source
            .lines()
            .take_while(|line| line.starts_with("//!") || line.is_empty())
            .collect::<Vec<_>>()
            .join("\n");

        for symbol in [
            "regime::merge_mtp_regimes",
            "candidate::cross_profile_winner",
            "probe::long_context_probe_target",
            "probe::stress_prompt_token_targets",
        ] {
            let bullet = header
                .split("\n//! - ")
                .find(|bullet| bullet.contains(symbol))
                .unwrap_or_else(|| panic!("crate docs must describe {symbol}"));
            assert!(
                bullet.contains("**Library-only / unwired:**"),
                "crate docs must qualify {symbol} as library-only and unwired"
            );
        }

        let tuning = include_str!("../../../docs/tuning.md");
        assert!(tuning.contains("not part of `findbest` in this release"));
        assert!(tuning.contains("ship as tested library behaviour only"));
    }
}
