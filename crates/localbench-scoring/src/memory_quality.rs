//! The memory-quality evaluation report and its pass gate.
//!
//! The scorer and fixtures live in LocalMind (the engine owns extraction and
//! retrieval); LocalBench invokes `localmind eval --json` and gates the
//! normalized result. Parsing and gating are kept separate from the invocation
//! so they are testable against a canned payload without the LocalMind binary.

use serde::{Deserialize, Serialize};

/// Default pass bar for the memory-quality gate, mirroring the LocalMind
/// regression threshold.
pub const MEMORY_QUALITY_THRESHOLD: f64 = 0.9;

/// One fixture's scores in the eval report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FixtureScore {
    pub name: String,
    pub candidate_count: u32,
    pub extraction_precision: f64,
    pub extraction_recall: f64,
    pub retrieval_recall_at_k: f64,
}

/// The model-extraction lift over the deterministic baseline, present only
/// when the eval ran with lift measurement. A zero lift (the offline case, no
/// local model) is the honest signal that model extraction is not yet
/// justified as a default.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Lift {
    pub extraction_precision_delta: f64,
    pub extraction_recall_delta: f64,
    pub retrieval_recall_at_k_delta: f64,
}

/// The memory-quality eval report (`localmind eval --json`). Every required
/// field must be present — a missing field is a deserialization error, never a
/// silent zero.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalReport {
    /// The retrieval cutoff (recall@k).
    pub k: u32,
    pub scores: Vec<FixtureScore>,
    pub mean_extraction_precision: f64,
    pub mean_extraction_recall: f64,
    pub mean_retrieval_recall_at_k: f64,
    #[serde(default)]
    pub lift: Option<Lift>,
}

impl EvalReport {
    /// Parse the eval JSON payload.
    ///
    /// # Errors
    /// Returns the `serde_json` error when the payload is malformed or a
    /// required field is missing.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Whether the report passes the gate: all three means at or above the
    /// threshold.
    #[must_use]
    pub fn passes(&self, threshold: f64) -> bool {
        self.mean_extraction_precision >= threshold
            && self.mean_extraction_recall >= threshold
            && self.mean_retrieval_recall_at_k >= threshold
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    const CANNED: &str = r#"{
        "k": 5,
        "scores": [
            { "name": "meeting-notes", "candidate_count": 12,
              "extraction_precision": 0.95, "extraction_recall": 0.92,
              "retrieval_recall_at_k": 1.0 }
        ],
        "mean_extraction_precision": 0.95,
        "mean_extraction_recall": 0.92,
        "mean_retrieval_recall_at_k": 1.0
    }"#;

    #[test]
    fn parses_a_canned_payload_and_passes_the_gate() {
        let report = EvalReport::from_json(CANNED).expect("parse");
        assert_eq!(report.k, 5);
        assert_eq!(report.scores.len(), 1);
        assert!(report.lift.is_none());
        assert!(report.passes(MEMORY_QUALITY_THRESHOLD));
    }

    #[test]
    fn any_mean_below_the_threshold_fails_the_gate() {
        let mut report = EvalReport::from_json(CANNED).expect("parse");
        report.mean_extraction_recall = 0.89;
        assert!(!report.passes(MEMORY_QUALITY_THRESHOLD));
    }

    #[test]
    fn a_missing_required_field_is_an_error_not_a_zero() {
        let truncated = r#"{ "k": 5, "scores": [] }"#;
        assert!(EvalReport::from_json(truncated).is_err());
    }

    #[test]
    fn lift_block_round_trips_when_present() {
        let with_lift = r#"{
            "k": 5, "scores": [],
            "mean_extraction_precision": 1.0,
            "mean_extraction_recall": 1.0,
            "mean_retrieval_recall_at_k": 1.0,
            "lift": { "extraction_precision_delta": 0.0,
                      "extraction_recall_delta": 0.02,
                      "retrieval_recall_at_k_delta": -0.01 }
        }"#;
        let report = EvalReport::from_json(with_lift).expect("parse");
        let lift = report.lift.expect("lift present");
        assert!((lift.extraction_recall_delta - 0.02).abs() < 1e-12);
    }
}
