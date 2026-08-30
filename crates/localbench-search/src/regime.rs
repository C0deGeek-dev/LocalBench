//! MTP-regime merge: pick the better of two full-search regimes per profile.

use serde::{Deserialize, Serialize};

/// One regime's finished result for a profile — the minimal shape the merge
/// reads (the app layer carries the full result payload alongside).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegimeResult {
    /// The scoring profile this result belongs to (`pure` / `balanced`).
    pub profile: String,
    /// The winner's score under that profile.
    pub score: f64,
}

/// Which regime the merge picked for a profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RegimePick {
    MtpOn,
    MtpOff,
}

/// One merged profile outcome.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MergedRegime {
    pub profile: String,
    pub pick: RegimePick,
    pub score: f64,
    /// The losing regime's score, when both regimes measured.
    pub other_score: Option<f64>,
}

/// Pick the better of two full-search regimes (MTP-on vs MTP-off) per profile.
/// MTP costs VRAM and is frequently net-negative on consumer GPUs, so the
/// winner is whichever actually measured higher — not an assumption. **MTP-off
/// must strictly beat MTP-on: a tie keeps MTP on** (the on-regime already paid
/// for its draft head; an equal score means it is not net-negative).
#[must_use]
pub fn merge_mtp_regimes(
    with_mtp: &[RegimeResult],
    without_mtp: &[RegimeResult],
) -> Vec<MergedRegime> {
    // Profiles in first-seen order across both regimes.
    let mut profiles: Vec<&str> = Vec::new();
    for r in with_mtp.iter().chain(without_mtp) {
        if !profiles.contains(&r.profile.as_str()) {
            profiles.push(&r.profile);
        }
    }

    let mut merged = Vec::new();
    for profile in profiles {
        let on = with_mtp.iter().find(|r| r.profile == profile);
        let off = without_mtp.iter().find(|r| r.profile == profile);
        let outcome = match (on, off) {
            (Some(on), Some(off)) => {
                if off.score > on.score {
                    MergedRegime {
                        profile: profile.to_string(),
                        pick: RegimePick::MtpOff,
                        score: off.score,
                        other_score: Some(on.score),
                    }
                } else {
                    MergedRegime {
                        profile: profile.to_string(),
                        pick: RegimePick::MtpOn,
                        score: on.score,
                        other_score: Some(off.score),
                    }
                }
            }
            (Some(on), None) => MergedRegime {
                profile: profile.to_string(),
                pick: RegimePick::MtpOn,
                score: on.score,
                other_score: None,
            },
            (None, Some(off)) => MergedRegime {
                profile: profile.to_string(),
                pick: RegimePick::MtpOff,
                score: off.score,
                other_score: None,
            },
            (None, None) => continue,
        };
        merged.push(outcome);
    }
    merged
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn r(profile: &str, score: f64) -> RegimeResult {
        RegimeResult {
            profile: profile.to_string(),
            score,
        }
    }

    #[test]
    fn a_tie_keeps_mtp_on() {
        // Equal scores: MTP is not net-negative, so the on-regime survives.
        let merged = merge_mtp_regimes(&[r("pure", 300.0)], &[r("pure", 300.0)]);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].pick, RegimePick::MtpOn);
    }

    #[test]
    fn off_wins_only_by_strictly_beating_on() {
        let merged = merge_mtp_regimes(&[r("pure", 300.0)], &[r("pure", 300.01)]);
        assert_eq!(merged[0].pick, RegimePick::MtpOff);
        assert_eq!(merged[0].score, 300.01);
        assert_eq!(merged[0].other_score, Some(300.0));

        let merged = merge_mtp_regimes(&[r("pure", 301.0)], &[r("pure", 300.0)]);
        assert_eq!(merged[0].pick, RegimePick::MtpOn);
    }

    #[test]
    fn merge_is_per_profile_and_tolerates_one_sided_regimes() {
        let merged = merge_mtp_regimes(
            &[r("pure", 310.0)],
            &[r("pure", 305.0), r("balanced", 280.0)],
        );
        assert_eq!(merged.len(), 2);
        let pure = merged.iter().find(|m| m.profile == "pure").unwrap();
        assert_eq!(pure.pick, RegimePick::MtpOn);
        let balanced = merged.iter().find(|m| m.profile == "balanced").unwrap();
        assert_eq!(balanced.pick, RegimePick::MtpOff);
        assert_eq!(balanced.other_score, None);
    }
}
