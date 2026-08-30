# Schema status

These JSON schemas describe LocalBench's report/store artifacts. Since the
v2.0.0 native-stack rewrite the emitters are **Rust** (the PowerShell cmdlets and
the `*.tests.ps1` conformance suites named in earlier revisions of this file no
longer exist). Not every schema still has a live emitter — treat this table as
the source of truth for which are current.

| Schema | Status | Emitter / consumer |
|---|---|---|
| `tuner-best-config.schema.json` | **current** | Written by `localbench findbest` (`crates/localbench/src/export.rs`) to `~/.local-llm/tuner/best-<key>.json`; the export round-trips through the typed `TunerEntry` and is re-validated against this schema on write. LocalBox's guided launcher reads the same store for AutoBest replay. |
| `localbox-autobest-v1.schema.json` | **current** | The AutoBest launcher-profile shape LocalBox consumes. |
| `localbench-capability-v1.schema.json` | **current** | The cross-arm harness-capability report `localbench arms`/`rescore` render. |
| `localbench-uplift-v1.schema.json` | **current** | The lesson-on/off uplift A/B report `localbench uplift` renders (mean ± stddev, significance verdict, memories-used injection assertion). |
| `hardware-profile.schema.json` | **retired** | No current emitter — the hardware-profile report is not produced by the Rust binary. Kept for history; do not treat as a live contract. |
| `localbench-tds-v1.schema.json` | **retired** | No current emitter — the Tool Discipline Score report is not produced by the Rust binary (the `scoring::tds` module is library-only/unwired). Kept for history. |

Every current schema carries an explicit version field (`schema_version`, plus
`tuner_version` / `launcher_export_version` where applicable) so producer and
consumer can gate on it.

The best-config document schema remains `1`; its independent per-entry
measurement compatibility is currently `tuner_version: 5`. Older entries may
remain in a store but current readers do not rank or replay them.

**Cross-repo conformance.** The launcher best-config store is a cross-project
contract with LocalBox. It is conformance-tested in Rust:
`crates/localbench/tests/launcher_contract_localbox.rs` checks LocalBox's
committed launcher-envelope fixture cross-repo in CI (both directions). A
Rust conformance test that validates the emitted capability/uplift/best-config
artifacts against these schema files is a follow-up; the best-config export is
type-round-trip + schema re-validated at write today.
