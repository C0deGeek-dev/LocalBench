//! Measurement support for the LocalBench auto-tuner: everything around a
//! trial that is *not* the live server run itself.
//!
//! - [`classify`] — OOM-signature classification (the pinned llama.cpp
//!   failure-pattern list) and the output-quality sanity check.
//! - [`soak`] — context-soak validation evaluation (pure: it judges recorded
//!   per-target outcomes) and the stress-recovery override ladder.
//! - [`prompt`] — the synthetic long-context stress prompt.
//! - [`cache`] — the fingerprinted trial cache: stable keys (FNV-1a, never the
//!   std hasher, so a persisted cache stays valid across builds), phase
//!   eligibility, fingerprint diffing, and crash-safe persistence.
//! - [`grade`] — the offline vendored cargo cache for the `--network=none`
//!   grade, and the language-aware test counting over the shared grade table.
//! - [`container`] — the network-isolated container grade plan and the
//!   Docker-wedge circuit breaker.
//! - [`arms`] — the recorded arm matrix (an arm is a config, not a label) and
//!   the arm-isolation contract that refuses a contaminated baseline.
//! - [`runner`] — capability-scorecard parsing, the comparative report with
//!   anti-gaming paired-metric verdicts, the instrument self-test gate, and
//!   cell persistence with deterministic offline rescore.

#![forbid(unsafe_code)]

pub mod arms;
pub mod cache;
pub mod classify;
pub mod container;
pub mod grade;
pub mod prompt;
pub mod runner;
pub mod soak;
