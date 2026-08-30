# Plan-Template Overrides — LocalBench

Project-specific content spliced into a copy of the canonical plan template
(the `plan-from-template` skill in the c0degeek-ai plugin). The canonical
template is generic; everything LocalBench-specific lives here. Never fork the
template — generic improvements go upstream to c0degeek-ai instead.

LocalBench has no dedicated planning skill, so the c0degeek `plan-from-template`
skill auto-splices this file from its conventional path
(`.claude/plan-template-overrides.md`). Each section below names the extension
point in the copied plan where its content lands.

> **LocalX workspace note.** Plans, tasks, and work tracking live in the private
> LocalHub repo (`LocalHub/plans/localbench/`), never in this repo. This repo
> keeps only its `docs/`, README, and CHANGELOG. See `LocalX/CLAUDE.md`.

## §2 Verification-commands rows (repo defaults, mirror CI)

> LocalBench is a native **Rust** workspace since the v2.0.0 native-stack rewrite
> (the PowerShell module and .NET TUI are retired). Confirm/correct in subject 00.

| Purpose | Command | Notes |
|---|---|---|
| Build | `cargo check --workspace` | from `.github/workflows/ci.yml` |
| Test | `cargo test --workspace` | golden vectors, instrument self-tests, contract tests |
| Lint/format | `cargo fmt --check` then `cargo clippy --workspace --all-targets -- -D warnings` | both clean |
| Docs link-check | `lychee --no-progress --offline docs docs/wiki README.md` | org-pinned |

## §6 plan-specific principles (slot for §6.18+)

- **Tier-1 parity (Windows / Linux / macOS).** Native Rust behind cross-platform
  traits (ADR-0007); a box that only works on one OS is not done.
- **Rust engineering rules hold.** MSRV 1.82, exact-pinned workspace deps,
  `#![forbid(unsafe_code)]`, `unwrap`/`expect`/`todo`/`dbg` denied outside
  `#[cfg(test)]`, typed errors.
- **Launcher-contract boundary is load-bearing.** `docs/launcher-contract.md`
  plus the versioned `LauncherVersion` envelope define the LocalBox launcher
  surface this tuner consumes; the Rust conformance test
  (`crates/localbench/tests/launcher_contract_localbox.rs`) checks LocalBox's
  committed envelope fixture cross-repo in CI. Any doc change describing that
  surface must match the contract doc and the conformance test.
- **Shared crate tier is rev-pinned.** `localx-llama-core`/`-runtime`/
  `localx-eval-core` and `localbox-launcher` are rev-pinned git deps; they must
  move in lockstep — advance the rev at a checkpoint and re-run the suite.
- **Measurement integrity is the product.** Instrument gate before spend,
  contaminated-baseline refusal, paired-metric anti-gaming verdicts, timeouts as
  caveats (never capability results), crash-safe fingerprinted trial cache. A
  change that could silently deflate/inflate a number must add or update the
  test that pins it.
- **Doc-ownership map (which doc owns which area).** Match a change to its owning
  doc; do not restate an area in two places.
  - `README.md` — lean overview, install entry point, ecosystem links.
  - `docs/launcher-contract.md` — the LocalBox launcher contract (canonical).
  - `docs/` owned topics: running a benchmark, reading a report, defining a
    profile, tuning.
  - `schemas/` — report/profile schemas; `reports/` + `examples/` — sample output.
  - `docs/wiki/` — wiki source (see below).
  - `CHANGELOG.md` — every user-facing change, under an Unreleased/next heading.
- **Wiki source of truth is in-repo.** `docs/wiki/` is authoritative and
  PR-reviewed; the published GitHub Wiki is a one-way generated mirror — never
  hand-edited on github.com. Wiki Reference pages link the owned `docs/`.
- **VERSION discipline, both directions.** No README/doc/wiki claim may exceed
  the current `VERSION` (read the `VERSION` file, never hardcode a literal),
  **and** no doc may describe the retired PowerShell/.NET stack (or its deleted
  Pester suites) as current.

## §7 plan-specific gates

- [ ] `cargo fmt --check`, clippy `-D warnings`, and `cargo test --workspace`
      pass or blockers recorded.
- [ ] Any doc describing the launcher surface still matches
      `docs/launcher-contract.md` and the Rust conformance test.
- [ ] No README/doc/wiki claim exceeds the current `VERSION`, and none describes
      the retired PowerShell/.NET stack as current.

## Captain Hindsight prompt — extra "Check specifically for" lines

- Any `README.md`/`docs/`/`docs/wiki/` claim that does not match shipped
  behaviour at the current `VERSION` (in either direction), or a wiki page
  hand-edited on github.com instead of the in-repo `docs/wiki/` source.
- Drift between `docs/launcher-contract.md`, the committed envelope fixture, and
  the cross-repo Rust conformance test.
- OS-specific assumptions that break tier-1 parity (Windows/Linux/macOS).
