//! The LocalBench application layer.
//!
//! - [`consumer`] — the launcher-contract consumer: the version-triple gate a
//!   benchmark applies before trusting a launcher implementation, plus the
//!   mockable conformance seam.
//! - [`export`] — the tuner best-config export: merge a winner into the
//!   launcher-readable `best-<key>.json` store (never overwriting across
//!   distinct quant/context/mode/VRAM/prompt/profile/vision combinations) and
//!   the store/profile path conventions.
//! - [`output`] — machine-output discipline: JSON to stdout clean, logs to
//!   stderr, non-TTY defaults to JSON, and the JSONL event stream
//!   (`started|result|completed|error`) for long runs.

#![forbid(unsafe_code)]

pub mod coach;
pub mod consumer;
pub mod diagnostics;
pub mod export;
pub mod gguf;
pub mod matrix;
pub mod output;
pub mod solver;
pub mod trial;
pub mod tuner;
pub mod upliftrun;
