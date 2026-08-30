//! Pure scoring math for the LocalBench benchmark/auto-tune stack.
//!
//! Everything here is deterministic and I/O-free — the numbers ARE this tool's
//! value, so each formula is pinned by golden tests against known
//! inputs/outputs before anything consumes it:
//!
//! - [`stats`] — median / mean / sample standard deviation.
//! - [`score`] — the trial ("pure") score per optimization target, and the
//!   balanced score that discounts it by CPU/RAM/VRAM headroom, run variance,
//!   and cross-phase throughput stability.
//! - [`tds`] — parsing the tool-discipline scorecard line into metrics, the
//!   safety gates (gates, never averaged terms), and A/B delta rows.
//!   **Library-only / unwired:** no binary command emits a TDS report today
//!   (its schema is marked retired in `schemas/README.md`); kept for history.
//! - [`uplift`] — the deterministic answer grader and the trial-level A/B
//!   statistics (aggregate, pooled-stddev significance, injection contract).
//! - [`memory_quality`] — the memory-quality eval report and its pass gate.
//!   **Library-only / unwired:** there is no `memory-quality` binary command or
//!   schema; the scorer/fixtures live in LocalMind (`localmind eval`), and this
//!   parses/gates its payload. `reports/memory-quality-sample.md` is an
//!   illustrative sample, not a report this binary produces.

#![forbid(unsafe_code)]

pub mod memory_quality;
pub mod score;
pub mod stats;
pub mod tds;
pub mod uplift;
