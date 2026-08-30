# LocalBox ↔ LocalBench contract

**Contract version: 3** (bump on any add/remove/rename of a trait capability,
a required envelope field, a shared artefact field, or a version-gate rule
below; additions are minor-compatible for implementors, removals/renames are
breaking).

This document is the formal contract between the launcher (LocalBox by
default) and LocalBench. It covers three surfaces: the launcher trait
LocalBench consumes, the version envelope both sides gate on, and the
artefacts/settings the two sides exchange. LocalBench owns this document and
the schemas; the launcher links here rather than restating it.

## 1. The launcher trait

The authoritative, machine-checked surface is the `Launcher` trait in the
shared `localx-llama-core` crate. Parameter obligations are the type system's
job — any implementation that compiles against the trait (including a test
mock) can drive the tuner. The capability groups:

| Group | Obligation |
|---|---|
| Model resolution | model key → definition; definition + quant → resolved GGUF path; context key resolution and token values; vision-module path when the model has one |
| Hardware | total VRAM in GB |
| Binary resolution | path to the `llama-server` binary per mode (native / turboquant / mtpturbo / PrismML), installing when sanctioned; per-mode install roots |
| Argument / KV | the `llama-server` argv for a trial config; available KV cache types per mode; unsupported-type rejection |
| Server lifecycle | a free TCP port; block until the server responds; stop; record the active backend session |
| Paths | expand `~` and `%VAR%` style path spellings |

## 2. The version envelope

A launcher presents a `LauncherVersion` envelope:

```jsonc
{
  "version": "1.2.1",              // the product version (suffix allowed)
  "api_version": 1,                // launcher API generation
  "launcher_export_version": 1,    // export schema generation
  "supported_targets": ["LocalBox"],
  "supported_runtimes": ["llamacpp"]
}
```

The consumer gate (`assert_launcher_usable`) refuses a launcher whose
`api_version` or `launcher_export_version` is below 1, whose declared
targets/runtimes don't include the `LocalBox`/`llamacpp` pairing (a blank list
means no preference), or whose numeric product version is below the consumer's
floor. The floor comparison is **suffix-free numeric dotted** — `1.2.1-beta.3`
satisfies a `1.2.1` floor; the human train string never carries the gate.

## 3. Shared artefacts and settings

| Artefact | Writer → Reader | Format authority |
|---|---|---|
| Tuner best-config store `~/.local-llm/tuner/best-<key>.json` | LocalBench tuner → launcher AutoBest load (both sides ship an implementation) | `schemas/tuner-best-config.schema.json` (binding); readers must ignore records whose `tuner_version` they do not understand (currently 5); every LocalBench write re-validates against the schema |
| Trial cache `~/.local-llm/tuner/trial-cache-<key>[-<context>].json` | LocalBench internal | fingerprinted; invalidates whole when protocol, request/prompt/template shape, response schema, effective session settings, engine/build, model identity, or another measurement-shaping input differs |
| Trial manifest `~/.local-llm/logs/tuner/run-<run-id>.jsonl` | LocalBench internal/operator evidence | append-only per-attempt diagnostics; requested settings are separate from authoritative observations and log-only advisories; newest 20 completed runs retained while active markers are protected |
| Hardware profile | LocalBench internal | `schemas/hardware-profile.schema.json` (binding) |
| Capability scorecard | LocalPilot `eval` → LocalBench `arms`/`rescore` | the shared `localx-eval-core` wire contract (schema 1) |

Settings / discovery:

- LocalBench binds the real launcher **in-process** (the `localbox-launcher`
  crate) and gates it with the envelope before any trial.
- LocalBench honors a catalog-required engine (such as PrismML for Ternary
  Bonsai) and refuses an incompatible explicit mode before starting a trial.
- Tuner candidates use LocalBox's `settings_launch_params` and
  `apply_session_defaults` precedence: candidate values, then settings, then the
  single-session `parallel=1` / `cache_reuse=256` defaults.
- Root discovery order for the launcher's installed tree: environment
  (`LOCALBOX_ROOT`), then the launcher's own setting, then a sibling checkout.

## 4. Conformance

- LocalBench's consumer gate is pinned by unit tests against a mock launcher
  (the trait seam) and by an integration test against **the real LocalBox
  envelope fixture** (`LOCALBOX_ROOT` or a CI checkout) — a launcher change
  that would break the tuner fails LocalBench's build instead of a user's run.
- LocalBox commits its envelope as a fixture and pins the live
  `LlamaLauncher::version()` against it and against `VERSION`, and its CI runs
  LocalBench's consumer gate against the checkout — a breaking envelope change
  fails at the source.
