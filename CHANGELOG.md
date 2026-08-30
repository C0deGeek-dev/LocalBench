# Changelog

Past-tense record of shipped changes, newest first.

## Unreleased

## v5.0.0 - 2026-08-30

- Began a new public Git history under PolyForm Noncommercial 1.0.0. Versions
  through v4.0.0 remain available to existing recipients under their original
  MIT grants; commercial use of v5+ requires a separate written license.

## v4.0.0 - 2026-08-29

Coordinated LocalX release.

- Wall-clock-bounded solver and grader commands now capture output through
  regular temporary files and own a process group/tree, so a descendant that
  inherits stdout or stderr cannot keep LocalBench blocked after the deadline.
  Timeout cleanup targets the whole owned tree while unrelated processes
  survive; any failed tree cleanup is reported explicitly with a bounded,
  potentially truncated output snapshot.
- Every Docker health, Cargo-warm, and grade run now has a valid unique name.
  A timeout or ambiguous Docker CLI error triggers a bounded
  `docker rm -f <exact-name>` compensation. Cleanup failures remain secondary
  infrastructure caveats: the original timeout/grade result, wedge-breaker
  accounting, persisted cell, and offline rescore semantics do not change.
- Corrected the `localbench-search` crate documentation to mark its
  long-context probe, stress-soak, MTP-regime merge, and cross-profile
  soak-adoption rules as library-only and unwired from `findbest`.
- `localbench findbest` now measures deterministic templated chat through
  `/v1/chat/completions`, validates visible assistant content and required
  finite prompt/decode timings, and launches candidates through LocalBox's
  settings plus `--parallel 1` / `--cache-reuse 256` session defaults. Its
  expanded cache fingerprint invalidates raw-completion or auto-parallel data,
  and new profiles use the shared measurement version 5.
- Trial outcomes now retain typed launch/readiness/request/response/content
  stage and reason values. Only real startup/OOM evidence enters memory
  recovery; unusable responses cannot score, cache as successes, verify, or be
  saved, and verification attempts remain within the configured trial budget.
- Every live or cached candidate now has an append-only JSONL diagnostic record
  and unique live log under `~/.local-llm/logs/tuner/`. Requested values,
  authoritative `/props` observations, and bounded log-derived advisories stay
  separate; excerpts and secret-like arguments are redacted, active runs are
  protected, and completed-run retention is bounded to 20.

## v3.3.2 - 2026-08-20

Coordinated LocalX release.

## v3.3.1 - 2026-08-20

Coordinated LocalX release.

- `localbench findbest` recognizes a crashed trial the moment its server process
  exits, instead of polling the dead port for the full startup budget. The
  readiness wait now watches the child alongside the `/health` poll: a process
  that exits is classified from its log tail immediately (an OOM signature still
  reads as OOM), so a trial that used to take the whole 300 s to be marked failed
  is marked in about a second — on a measured OOM the tuner spent 97% of every
  failed trial waiting on a process that no longer existed. A server that is
  still alive and still loading keeps the full budget. The trial payload now
  records *why* startup failed — an OOM exit, a non-OOM exit, or a still-running
  timeout — where the three previously collapsed into one verdict, and a new
  `--startup-timeout <secs>` flag lets a fast NVMe lower the budget or a slow
  disk raise it (LocalHub#77).

## v3.3.0 - 2026-08-19

Coordinated LocalX release.

## v3.2.0 - 2026-08-18

Coordinated LocalX release.

- `localbench findbest` now classifies a model as dense or MoE from its GGUF
  header instead of a catalog guess. It reads only the metadata header (never the
  tensor data) for the two facts the search turns on — expert count and layer
  count — and passes them into the search space, so a dense model is tuned on the
  `--n-gpu-layers` axis it actually has rather than the no-op `--n-cpu-moe` one.
  A dense model that OOMs at its baseline now recovers: the VRAM-fit phase shrinks
  the KV cache (turbo pairs) first, then offloads layers halved from the real
  layer count, and produces a winner instead of "no candidate survived tuning". A
  dense baseline that already fits spends no VRAM-fit trials. An unreadable or
  unfamiliar header degrades to the previous catalog heuristic, so nothing that
  tunes today stops tuning (LocalHub#76).

- `localbench findbest` downloads a model that is not on disk yet instead of
  refusing with "download the model first (run localbox)": the GGUF is fetched
  through the launcher library's resumable download (the same one a LocalBox
  launch performs) with coarse progress on stderr, then tuning starts. Only the
  GGUF is fetched — a configured vision projector or draft model is never
  pulled by a tuning run. Re-pinned `localbox-launcher` to the LocalBox commit
  that moved the downloader into the library (LocalHub#75).

## v3.1.0 - 2026-08-11

Coordinated LocalX release.

## v3.0.0 - 2026-08-11

Coordinated LocalX release.

## v2.8.1 - 2026-08-07

Coordinated LocalX release.

## v2.8.0 - 2026-08-07

Coordinated LocalX release.

## v2.7.0 - 2026-08-02

Coordinated LocalX release.

## v2.6.0 - 2026-07-27

Coordinated LocalX release.

- **One-line install for the whole stack.** `localbench` is now installed alongside
  the rest of LocalX by a single command, with each archive checked against its
  published SHA-256 before it is unpacked:

  ```sh
  curl -fsSL https://raw.githubusercontent.com/C0deGeek-dev/LocalPilot/main/install/install.sh | sh
  ```

  The train cuts every tool to one version and they are only tested together, so
  they are installed together; `localpilot update --all` re-runs it. Nothing in
  this repository changed to support it — the release archives and `manifest.json`
  already published were the whole contract.

## v2.5.0 - 2026-07-27

Coordinated LocalX release.

- **Releases now ship prebuilt binaries.** A tagged release previously carried
  release notes and nothing else, so the only way to get this tool was to build
  it from source with a Rust toolchain. Each release now attaches an archive per
  platform — Linux x86-64 (glibc and a static musl build that runs anywhere),
  Linux arm64, macOS Apple Silicon, and Windows x86-64 — with a SHA-256 beside
  each archive and a `manifest.json` indexing the release.

  Publishing happens once, only when every platform built. A partial release is
  worse than a failed one: a download cannot tell the difference. The checksums
  prove an archive was not corrupted in transit; they do not prove who produced
  it, which needs signing.

## v2.4.0 - 2026-07-26

Coordinated LocalX release.

- Added PrismML tuning support. `findbest` now accepts `--mode prism` (and the
  `prismml` alias), automatically honors LocalBox catalog-required engines such
  as Ternary Bonsai 27B, and refuses incompatible explicit modes before a
  trial. Prism uses mainline-compatible KV candidates without turbo KV or MTP;
  AutoBest schemas and exports carry the shared `prismml` wire value. The
  `localx-llama` tier and `localbox-launcher` pins advanced together to the
  Prism-capable revisions.
- Added the coached arm: an arm whose spec names a `coach` script is driven
  through the solver's MCP serve surface by a deterministic scripted coach
  (steer/reply/cancel rules on session events, bounded fires, replayable),
  instead of a headless eval. The scorecard's process layer and the
  comparative report gained an interventions count (`avg interventions`
  column; its delta is gameable — fewer interventions wins only when solve
  rate held). The three-arm learning comparison (uncoached / coached /
  post-lesson) is documented in
  [docs/external-runner.md](docs/external-runner.md#the-coached-arm-scripted-coach-over-mcp);
  the promote step stays review-gated and is never automated.
- Advanced the `localx-eval-core` pin for the scorecard interventions field;
  `localx-llama-core`/`-runtime` stay on the prior rev so the dependency
  graph keeps a single `Launcher` (they move together with a
  `localbox-launcher` re-pin).

## v2.3.0 - 2026-07-07

Coordinated LocalX release.

- Both wire-format schemas (`localbench-capability-v1`, `localbench-uplift-v1`)
  are regenerated from the Rust emitters — the previous "binding" schemas
  still described the retired PowerShell emitter and rejected every real
  artifact (the shipped binary rejected its own shipped sample). The samples
  are now real emitter output, a conformance test pins emitters, schemas, and
  samples together in CI, and the PS-era provenance fields (`env`,
  `generatedAt`, `localbench_version`) are dropped from the wire contract
  (provenance lives in the cell ledger and the AutoBest export block).
- The balanced profile's cross-phase stability factor is live: the stability
  index is rebuilt from the run's own trial history at the final ranking, so
  a config that measured fast in one phase and slow in another is penalized
  as documented instead of the factor sitting permanently at 1.0.
- A JSON-mode `arms` run now streams the full JSONL protocol: `started`, one
  `result` event per persisted cell, and a terminal `completed` — or `error`
  on an abort, so a supervising harness never reads a `started` with no
  terminal event.
- `findbest` output carries the winner's balanced-score `confidence`
  (`full`/`partial`), so an operator can see whether the balanced discount
  actually saw its telemetry inputs; the README now carries the live-balanced
  caveat that previously lived only in docs/tuning.md.
- The cell-ledger/export timestamp math (`iso_from_secs`) is pinned by known
  epoch vectors, including the 2000/2100 century leap rules.
- Re-pinned the shared crate tier to its current head (picking up the
  streaming tail-recovery fixes, the stream-truncation contract test, the
  eval-core child-terminal isolation, and the proxy's constant-time key
  check) and aligned the launcher pin to the matching LocalBox revision so
  the dependency graph carries a single shared-tier copy. Full suite and the
  launcher-envelope conformance test re-run green per the re-pin ceremony.

## v2.2.0 - 2026-07-06

Coordinated LocalX release.

- **Arm isolation is now required, not opt-in, for every baseline arm.** A
  `baseline`/`is_baseline` arm with no `config` block used to skip the isolation
  check silently and anchor its harness-vs-baseline delta unguarded; the matrix
  now refuses it at setup, naming the fix. The declared clean config is also
  cross-checked against the arm's *actual* solver invocation: a baseline that
  also sets `verify`/`learn` is refused, because those flags reach `localpilot
  eval` as `--verify`/`--learn` (harness behaviour) and contradict its
  "harness off" claim.
- **An under-vendored offline cargo grade now reads as an infrastructure gap,
  not a solve failure.** A `--network=none` rust grade that fails with a cargo
  offline-fetch error (a dependency the exercise or the solver reached for was
  not pre-vendored) is detected in the container grader and surfaced as an
  infra-gap caveat — the cell counts unsolved (a floor) and the run prints a
  re-warm-the-cache caveat — instead of silently deflating the arm's solve rate
  as though the solver had failed. Mirrors the timeout-as-caveat path.
- **Docs now match the live isolation guarantee.** The external-runner and the
  arm-config module docs no longer imply LocalBench writes the exact `.toml`
  each arm runs with; they state that isolation is *checked against each arm's
  declared config* (and cross-checked against the actual `localpilot eval`
  flags), while the per-arm `.localpilot.toml`/`.localmind.toml` the emitters
  produce is the canonical config an operator stages and `localpilot eval
  --arm` applies.
- **Live `uplift` documents its per-arm memory staging.** The CLI drives both
  arms in one workspace and does not write per-arm LocalMind config or seed
  memory, so the docs, `--help`, and the run banner now state plainly that the
  operator stages each arm's memory first (baseline: clean store, learning off;
  lesson arm: seed pack staged, learning on) and that a mis-staged run VOIDs.
  Automating the staging in the CLI is deferred: it needs a live LocalMind/
  LocalPilot store, and a live local-model A/B is opportunistic while offline
  evidence is the accepted bar.
- **Disclosed the library-only memory-quality surface and removed dead path
  helpers.** `localbench-scoring::memory_quality` is now marked library-only /
  unwired in the crate docs and its sample report (no binary command or schema
  emits it; the scorer lives in LocalMind), mirroring the honest TDS
  disclosure. Removed the unused `StorePaths::native_profile`/`report` path
  helpers.
- **Refreshed the capability sample report and widened the doc-drift guard.**
  `reports/capability-sample.md` now matches the Rust `render_capability_report`
  output, including the `isolation` column, and no longer credits the retired
  `Show-*` PowerShell cmdlet. The CI doc-drift guard now scans `reports/` and
  `examples/` and flags the retired `Show-*` report cmdlets and deleted
  `*.tests.ps1` suites, while leaving legitimately-retained operator `*.ps1`
  glue alone.

## v2.1.5 - 2026-07-04

Coordinated LocalX release.

## v2.1.4 - 2026-07-04

Coordinated LocalX release.

## v2.1.3 - 2026-07-03

Coordinated LocalX release.

## v2.1.2 - 2026-07-03

Coordinated LocalX release.

## v2.1.1 - 2026-07-03

Coordinated LocalX release.

## v2.1.0 - 2026-07-03

Coordinated LocalX release.

- `rescore` now reproduces a live `arms` run exactly: a failed or hung cell is
  persisted as a synthetic unsolved cell (previously dropped, which inflated the
  rescored solve rate), and a grader-infrastructure death marks the in-flight
  cell unsolved rather than trusting the solver's self-claim.
- Each `arms` run without an explicit `--cells-dir` now writes to its own
  per-run subfolder, so one run's cells never mix with another's.
- Container grading now grades a writable copy of the solved tree under the
  toolchain PATH, so compiled-language cells build and grade correctly instead
  of reporting "0 tests ran".
- Re-pinned the shared `localx-llama` crate tier and `localbox-launcher` in
  lockstep to pick up the faithful no-think proxy and lowercased KV-type
  emission; the launcher-envelope conformance test stays green.
- Rewrote `schemas/README.md` for the Rust reality: it no longer cites the
  deleted PowerShell cmdlets and `*.tests.ps1` conformance suites as the binding
  guard; each schema is marked current or retired, and the real cross-repo
  conformance test (`launcher_contract_localbox.rs`) is named.
- **Fixed: the trial cache now persists on a fresh `~/.local-llm`.**
  `TrialCache::save` creates its parent `tuner/` directory before the crash-safe
  write, so the fingerprinted trial cache lands from the very first trial instead
  of failing every save with `No such file or directory` until a first winner had
  been exported. Not platform-specific — it only surfaced where `tuner/` did not
  already exist.
- Container grade plan now grades a **writable copy**: the solved tree is mounted
  read-only at `/src` and copied into a writable `/work` (with `-w /work`, under
  `bash -c` so the toolchain PATH survives), so compiled-language grades
  (rust/go/java/C++) can write build artifacts instead of being silently scored
  "0 tests ran → not solved". A `rust` grade can mount a warmed offline cargo
  registry via the new `cargo_cache` task field.
- Trial-cache fingerprint now records the GPU name(s) and the resolved
  `llama-server` build signature (variant dir + size), so a GPU swap or an
  `llm-update` that changes the binary invalidates stale measurements. **One-time
  effect:** existing trial caches (which recorded neither) are treated as a
  fingerprint mismatch and re-measured once.

## v2.0.2 - 2026-07-02

Coordinated LocalX release.

## v2.0.1 - 2026-07-02

Coordinated LocalX release.

## v2.0.0 - 2026-07-02

Coordinated LocalX release.

- **Retired the PowerShell module and the .NET TUI; LocalBench is now a
  single native binary.** The `src/` PowerShell module, its Pester suite, and
  `tui/LocalBench.Tui` are gone — every behaviour they carried ships in the
  Rust workspace, each hard-won invariant pinned by a golden test before its
  logic was ported. LocalBench no longer needs PowerShell or .NET at runtime
  on any platform; CI runs the tri-platform Rust gate and the cross-repo
  launcher-contract check only.
- **Live benchmark orchestration in the binary.** `localbench arms` drives
  the solver-under-test (`localpilot eval`, wall-clock bounded) across every
  harness arm × task cell behind the instrument self-test gate and the
  arm-isolation and clean-room boundaries, grades each cell in a
  network-isolated read-only container (engine pre-flight, exit 0 AND counted
  tests, timeouts stay caveats, the Docker-wedge breaker yields with the cell
  ledger intact), and persists graded cells so `localbench rescore` reproduces
  the live report offline. `localbench uplift` renders saved lesson-uplift
  reports and runs the live lesson-on/off A/B via `localpilot print`, reading
  the memories-used audit from the session log; the injection contract voids
  a result when an arm did not inject as configured.
- **`localbench findbest` reuses measurements across runs.** Decisive trials
  (healthy or definite OOM) persist in the fingerprinted trial cache
  (`tuner/trial-cache-<key>[-<context>].json`, crash-safe save after every
  measurement); a repeated or interrupted tune skips already-measured
  configs, transient startup failures are always re-measured, and the cache
  invalidates whole — naming the differing fields — when anything shaping a
  measurement changes. Opt out with `--no-cache`.
- **The `localbench` app crate: CLI, launcher-contract consumer, best-config
  export, machine-output discipline.** The launcher contract is now a shared
  Rust trait (in `localx-llama-core`) with a versioned envelope; the consumer
  gates on the product-version floor plus the version triple (`api_version≥1`,
  `launcher_export_version≥1`) and the declared LocalBox/llamacpp pairing, and
  a mock launcher pins the seam. Best-config export merges a winner into the
  launcher-readable `tuner/best-<key>.json` (one slot per
  quant/context/mode/VRAM/prompt/profile/vision combination; vision and text
  never overwrite each other; a corrupt existing store is refused, never
  clobbered; every write re-validates against the shared store schema, UTF-8
  no BOM). Machine output: structured results to stdout clean, logs to stderr,
  non-TTY stdout defaults to JSON, and the JSONL `started|result|completed|
  error` event stream for long runs — no log line can contaminate position 0.
  The binary exposes `version`, `instruments`, and `rescore` today; the
  command surface grows as the live wiring lands.
- **Arm matrix + external-runner support in Rust
  (`localbench-measure::arms`/`runner`).** An arm is a recorded config, not a
  label: the exact `.localpilot.toml` per arm (full/fair/verify/warm knobs) and
  `.localmind.toml` (measurement arms explicitly disable learning; the warm arm
  carries auto-accept review, model-backed extraction, embedding-backed
  semantic dedup on its own CPU port, the rerank opt-in, and an
  absolute-path-required shared global store), the recorded 6-arm definition
  table, and the arm-isolation contract that refuses a contaminated baseline
  naming the leaked channels. The runner side parses capability scorecards via
  the shared wire contract, builds the comparative report with isolation
  provenance, enforces the anti-gaming paired-metric verdicts (a gameable win
  needs a correctness signal that did not regress; a delta against a
  contaminated baseline is refused), gates runs on the instrument good/bad
  self-tests, persists run cells, and rescores them deterministically offline;
  wall-clock timeouts stay caveats (a floor), never capability results.
- **Grader support in Rust (`localbench-measure::grade`/`container`).** The
  language-aware test counting now rides the shared `localx-eval-core` grade
  table (upgraded to count tests that RAN — rust sums passed+failed per result
  line, go counts FAILing packages, java honors NO-SOURCE/SKIPPED, catch2/jest
  and a generic passed-sum fallback added), keeping the exit-0-with-zero-tests
  gate. The offline `--network=none` rust grade gets its vendored-cache tooling
  (declared-deps corpus scan with the curated common-model-dep list, declared
  version winning; the deterministic warm manifest; a loud under-vendored-cache
  detector) — a mis-pointed corpus fails loud instead of warming an empty
  cache. The container grade plan pins `--network=none` + a read-only workspace
  mount, and the Docker-wedge guard (bounded run-a-container health check plus
  the 3-consecutive-grade-timeout circuit breaker) keeps a wedged engine from
  burning hours.
- **`localbench-measure` crate: measurement support in Rust.** The pinned
  llama.cpp OOM/failure-pattern classifier and the output-quality sanity check
  (length + 4-gram repetition), pure context-soak validation (startup/OOM,
  free-VRAM-floor breach, and budget exhaustion all fail — an unproven soak
  never reads as a pass), the stress-recovery override ladder (smallest
  CPU-offload bumps first, then batch-safe/draft-safe/no-MTP variants), the
  deterministic synthetic stress prompt, and the fingerprinted trial cache
  (stable `driver|signature` keys, FNV-1a canonical-JSON fingerprint hash with
  a pinned vector — never the std hasher — soak/guard/recovery/verify phases
  ineligible, only decisive results cached, crash-safe atomic save with `.bak`
  recovery, and fingerprint mismatches that name the differing fields).
- **`localbench-search` crate: the tuner's search strategy in Rust.** Candidate
  construction and beam selection (signature dedup, OOM exclusion, pure/balanced
  frontier union on the `both` profile), cross-profile validated-soak adoption
  (a passed-soak sibling beats a higher-scoring degraded winner), the MTP-regime
  merge (a tie keeps MTP on — off must strictly beat), KV-type/pair spaces per
  build mode, MoE sweep values with the fine-tune grid and stride-1 edge
  refinement, the climb-higher and dense (KV-first, layer-count-anchored NGL)
  recovery ladders, smart seeds from injected host facts, and the probe/stress
  target policy (long-context probe at ctx/2 floored 16k capped 96k; soak
  auto-enable for long coding-agent runs at 64k+). 42 golden tests pin the
  tie-breaks and ladders.
- **Native Rust workspace seeded (`crates/localbench-scoring`).** The scoring
  math is being rebuilt as Rust alongside the shipping PowerShell module (which
  stays authoritative until each piece is retired): trial/pure score per
  optimization target, the balanced score with its CPU/RAM/VRAM/variance/
  stability factor breakdown, cross-phase stability index, soak-collapse
  detection, TDS scorecard parsing with safety gates, uplift A/B statistics,
  and the memory-quality gate — every pinned literal carried over as a golden
  test (42 tests). Shared eval primitives come from the public `localx-llama`
  crate tier via a rev-pinned git dependency.

## v1.2.1 - 2026-07-01

Coordinated LocalX release.

## v1.2.0 - 2026-06-30

Coordinated LocalX release.

- **The claude-code arm's wall-clock budget defaults to the per-exercise backstop,
  and wall-clock timeouts are surfaced as a caveat.** A tighter 600s default on
  `Invoke-ClaudeCodeSolver` produced 14 `claude -p` timeouts in the v1.1.0 sweep
  that depressed the cc solve rate — an arm penalised for a short budget, not a
  capability gap. The default is now 900s (the same `-EvalTimeout` per-exercise
  backstop the other arms get; `--max-turns 40` stays the matched turn cap, which
  was not the binding limit). `status-v1.ps1` and the runner's harness-comparison
  standings now print a caveat counting the claude-code cells that hit the
  wall-clock budget, so the cc rate is read as a floor, not a pure capability
  number.
- **A compile-with-zero-tests is no longer scored solved.** The aider-arms grade
  keyed "solved" on the test command's exit code alone, so a solution that
  compiles but runs no tests (`test result: ok. 0 passed; 0 failed`) was counted
  a pass — found on `rust/xorcism`, where it inflated arms. `Invoke-Grade` now
  also parses how many tests actually ran (`Get-TestCount`, a table-driven
  per-language parse over the **full** captured output — rust sums every
  `test result:` line because the truncated display tail is often the doc-test
  `0 passed` even on a real pass) and scores **solved only when exit 0 AND
  tests_run > 0**. A zero-test or unparsed-output run fails closed (not solved)
  and logs loudly; the cell ledger now records `tests_run`. Pester-pinned against
  real grade tails per language. Note: a few cells in prior runs may have been
  inflated by this; a fresh sweep grades them honestly.
- **The aider-arms rust grade now builds offline.** Several exercises declare
  deps (`gigasecond` → time, `grep` → anyhow, `simple-cipher` → rand) and the
  agentic arms add more (rayon, thiserror, num-bigint, …); the offline grade
  (`--network=none` + `CARGO_NET_OFFLINE=true`) had no cache for them, so the
  rust cells false-failed for *every* arm and read as a capability gap. A new
  one-time warm step (`Initialize-CargoCache`) `cargo fetch`es the union of every
  exercise's declared deps (scanned from each `Cargo.toml` via the recorded,
  Pester-pinned `Get-RustCargoCacheDeps`) plus a curated common-model-dep list
  into a shared `aider-cargo-cache` registry volume, mounted into the rust grade —
  mirroring the warmed `aider-gradle-cache`. Network isolation is preserved: the
  warm is the only network step (`--network=bridge`, no model code) and the grade
  stays `--network=none`. The rust grade image moved to current stable
  (`rust:1.96-slim-bookworm`) because the corpus's current dep versions need a
  toolchain newer than the old `rust:1.82` image (e.g. `time-core 0.1.9` requires
  the edition2024 cargo feature). Verified live: 8/8 dep-using rust golds build
  offline with the warmed cache.
- **`examples/aider-arms/status-v1.ps1` now shows the corpus `total` column** (exercise
  count per language + grand total) so partial-sweep standings read against the full
  denominator at a glance. Reconciled the operator working copy into the versioned
  canonical script.
- **Cleared a pre-existing PSScriptAnalyzer warning** (`PSAvoidUsingEmptyCatchBlock`) on
  the eval-solver adapter's best-effort kill-on-timeout with a justified, narrowly scoped
  per-function `SuppressMessage` — the convention `PSScriptAnalyzerSettings.psd1` already
  documents — so the lint gate is clean.

## v1.1.0 - 2026-06-29

Coordinated LocalX release.

- **The `warm` arm now exercises embedding-backed semantic dedup, and the sweep
  pre-flights the embed endpoint.** `New-LocalMindWarmConfig` emits the CPU
  embedding endpoint (`embedding_base_url`/`embedding_model`, default `:8090`),
  `[inference.features] embeddings = true`, `[review] semantic_dedup = true`
  (so the warm store catches paraphrase duplicates), and a forward-compatible
  `[retrieval] rerank = true` (inert until the host retrieval path wires an
  embedder) — Pester-pinned in `tests/arm-configs.tests.ps1`.
  The solve phase gains an **embed-endpoint pre-flight** (`Test-EmbedEndpointHealthy`)
  that mirrors the Docker pre-flight: it proves the endpoint returns a vector
  before the first cell and aborts loud (ledger intact) otherwise. The gate runs
  **only when `warm` is in `-Arms`**, so the other arms are unaffected, and the
  embed server is **CPU-only (`-ngl 0`, no GPU VRAM)** so the chat model stays
  byte-identical across arms (the warm-vs-other deltas stay fair). New
  `-EmbedBaseUrl` param; start the server with LocalBox `llmembedserve`. See
  `examples/aider-arms/README.md`.

- **The arm sweep fails loud and early on a wedged Docker engine instead of
  burning hours.** A Docker-for-Windows/WSL2 engine can wedge so every `docker
  run` hangs and each grade times out — which, cell after cell, silently wastes a
  whole run as false failures. Two guards now catch it: a **pre-flight**
  (`Test-DockerHealthy`) that actually runs a throwaway container before the first
  cell and aborts the run if the engine will not (a `docker info` ping is not
  enough), and a **circuit breaker** that yields the run (ledger intact, supervisor
  exits so the watcher pings) after three consecutive grade timeouts — capping a
  mid-run wedge at minutes, not hours. `status-v1.ps1` surfaces `grader: docker
  OK/WEDGED` and the grade-timeout count at a glance.

- **The arm sweep's container grade is now bounded by a timeout.** `Invoke-Grade`
  ran `docker run` with no timeout, so a hung graded test (an infinite loop in the
  model's solution) or a wedged Docker-for-Windows engine (the `docker run` CLI
  left blocked with no container) froze the whole sweep indefinitely. Every grade
  `docker run` now goes through `Invoke-DockerBounded` (named container, async pipe
  reads, `WaitForExit(timeout)`; on expiry it `docker kill`s the container and
  reaps the CLI), mirroring the solver adapter's bounded-process pattern — a hung
  grade fails its cell and the loop continues instead of stalling. Added
  `_launch-warm-cc.ps1` (the detached supervisor launcher) and fixed `watch-v1.ps1`'s
  warm-store path.

- **Operator scripts for the Aider-polyglot arm sweep** (`examples/aider-arms/`).
  The recorded, resumable recipe that drives the model-pinned sweep:
  `aider-arms.ps1` (resumable JSONL-ledger runner), `status-v1.ps1` (standings +
  the key deltas + the warm store's accumulated-lesson count), `watch-v1.ps1`
  (supervisor watch), `reaper.ps1` (backstop that kills leaked/runaway grading
  processes), and `keepawake.ps1`. Paths are operator-specific; the README
  documents the arms (baseline / full / warm / claude-code), prerequisites, and
  the run command.

- **Harness-arm configs for the model-pinned convergence sweep.** Added
  `New-LocalPilotArmConfig` / `New-LocalMindWarmConfig` (`src/lib/81-arm-configs.ps1`)
  that emit the exact `.localpilot.toml` / `.localmind.toml` each arm runs with, so
  an arm is a *recorded config*, not a label: `full` (original), `fair` (rails
  matched to Claude Code's `--max-turns` cap + verify on — the fair LP-vs-CC arm),
  `verify` (the verify-gate ablation's +verify side), and `warm` (fair + a
  persistent machine-wide global LocalMind store shared across exercises).
  `Get-LocalBenchHarnessArmDefinition` records the matrix. The `localpilot eval`
  solver adapter gained a `-Verify` passthrough (`localpilot eval --verify`). All
  Pester-pinned (`tests/arm-configs.tests.ps1`). The live GPU run is opportunistic
  (D008).

- **`Invoke-LocalBenchTdsScorer` no longer silently reports offline FakeProvider
  numbers as a live result.** A live run now requires the `live tool-discipline
  scorecard:` line; if the LocalPilot live test skipped (e.g. it could not resolve
  a provider from the harness crate dir), the scorer fails with the skip reason
  instead of falling back to the bare-token offline scorecard. The parse-only
  (`-CapturedOutput`) path is unchanged; `-Live` exercises the strict selection on
  captured output. Caught when two different models reported bit-identical numbers.

## v1.0.0 - 2026-06-24

Coordinated LocalX 1.0 release. Adds the external-benchmark capability runner
(network-isolated container grading, acquire-don't-vendor corpora) proven
end-to-end on Docker.

- **Exported `Get-LlamaCppTopNCpuMoeFromCandidates` from the module.** The
  function was defined in `src/lib/45-trial-cache.ps1` but never added to
  `Export-ModuleMember`, so the LocalBox bridge — which probes for it with
  `Get-Command` after importing LocalBench — could never see it and silently
  fell back to an empty cpu-moe candidate list. It is now exported, so the
  bridge probe resolves.
- **Dead-code cleanup: removed 8 unused profile-store / tuner / path helpers.**
  No call sites remained for `Get-LocalBenchRunRoot`, `Get-LlamaCppTrialPrompt`,
  `Get-LlamaCppBestConfigCandidates`, `Remove-LlamaCppBestConfig`,
  `Test-LlamaCppBestConfigStale`, `Save-BestLlamaCppConfig`,
  `New-LlamaCppFineTuneMoeOverlays`, or `New-LlamaCppMtpComparisonOverrides`
  (none were exported). No behaviour change.
- Added the **arm-matrix runner** (`src/lib/79-arm-matrix.ps1`,
  `Invoke-LocalBenchArmMatrix`): drives the solver-under-test across the full
  matrix of harness arms × tasks and assembles the comparative
  `localbench-capability-v1` report, closing the gap where each (arm, task) cell
  had to be driven by hand. It reuses the single-cell grader
  (`Invoke-LocalBenchContainerGrade`) — so the arm-isolation guard runs per cell —
  and the comparative report builder; a failing or hung cell is isolated as an
  unsolved card so it can never abort the matrix. The runner<->solver contract
  stays caller-provided: the concrete LocalPilot adapter ships separately as
  `src/adapters/localpilot-eval-solver.ps1` (`Invoke-LocalPilotSolver` wraps
  `localpilot eval`, bounded by a connect/total timeout so a hung solver fails the
  cell), dot-sourced on demand so the module keeps no hard LocalPilot dependency.
  Mock-satisfiable end to end (no binary launched in tests). The runner<->solver
  contract now also passes the per-task id (`Task`), so the solver can label each
  scorecard per task (e.g. `localpilot eval --task`) instead of every cell sharing
  a default label.
- Wired the lesson-uplift eval to a **live model** via LocalPilot `print`:
  `New-LocalBenchUpliftPrintDriver` builds the per-task driver (run `localpilot
  print`, read the turn's memories-used from the session log), `Get-LocalBenchUpliftMemoriesUsed`
  parses a session event-log's last `memories_used` event, and `Invoke-LocalBenchUpliftRun`
  sequences both arms → aggregate → injection assertion → report. All
  mock-satisfiable (an injected print invoker + a fixture session log), so the
  live path's logic is tested without a model; the live N-trial run itself is
  opportunistic (validation-evidence policy).
- Added the **lesson-uplift eval** (`src/lib/76-uplift.ps1`, schema
  `localbench-uplift-v1`): a multi-trial lesson-on/off A/B over a **headroom**
  task set (project-convention tasks the base model fails unguided — real
  headroom, not near-ceiling harness-enforced discipline). `Invoke-LocalBenchUpliftArm`
  runs N trials per arm through a mock-satisfiable driver; `Get-LocalBenchUpliftAggregate`
  gives per-trial success rate as mean ± stddev; `Get-LocalBenchUpliftSignificance`
  reads the delta against the pooled-stddev band as **uplift / no-effect / regression**
  (never a bare number); `Assert-LocalBenchUpliftInjection` **voids** a result whose
  lesson arm did not inject the intended memories (or whose baseline injected any),
  via the memories-used audit; `Assert-LocalBenchUpliftGrader` is the deterministic
  grader's instrument self-test. The bundled headroom task set is
  `data/uplift/headroom-tasks-v1.json` (authored for LocalBench, never copied from
  an external benchmark). `localbench uplift --report <file>` renders a saved report.
  Offline and deterministic (D008); a live local-model A/B is driven through the
  module API + LocalPilot `print` and is opportunistic only.
- Removed the deprecated tuner alias `Invoke-LocalBenchLocalBoxLlamaCppTuner`.
  The one-window compatibility shim for the pre-rename name is gone; use the
  LocalBench-owned `Invoke-LocalBenchLlamaCppTuner`.
- Added the **self-improvement eval gate** (`src/lib/79-self-improvement-gate.ps1`):
  gates a proposed self-improvement patch (LocalPilot ADR-0034) on agent-quality
  evals before it reaches the human reviewer, reusing the capability scorecard.
  `Test-LocalBenchSelfImprovementGate` scores seven dimensions — edit correctness,
  test pass-rate, tool discipline, retrieval correctness, patch minimality, no
  hallucinated files, follows-repo-conventions — against tunable thresholds
  (`Get-LocalBenchSelfImprovementThresholds`); `Assert-LocalBenchSelfImprovementGate`
  blocks a sub-threshold patch from the human queue. Correctness is the backbone,
  so a lean-but-wrong patch is blocked (anti-gaming). The gate is **necessary, not
  sufficient** — a green gate never replaces human approval. Offline and
  deterministic (D008); a live local-model run is opportunistic only. See
  [`docs/external-runner.md`](docs/external-runner.md).
- Added the **external capability runner** (`src/lib/78-external-runner.ps1`): a
  named, mock-satisfiable **solver contract** for driving LocalPilot headless, a
  parser for LocalPilot's capability scorecard JSON, the **clean-room path
  boundary** that refuses to place a public benchmark corpus under any
  LocalPilot checkout, no-vendor dataset acquisition (SWE-bench Lite / Aider
  polyglot, into a user-local cache), a container grader (Docker via WSL2; the
  live run deferred behind a security sign-off), and a comparative capability
  report (`localbench-capability-v1` schema) with a contamination caveat. See
  [`docs/external-runner.md`](docs/external-runner.md).
- Added the **ablation headline** render helpers: `Get-LocalBenchArmDelta`
  (per-metric baseline-vs-arm delta) and `Show-LocalBenchAttribution` (per-feature
  attribution table flagging inert features), plus a sample headline report
  (`reports/capability-ablation-sample.md`).
- Added the **arm-isolation boundary** to the external runner: each arm's
  effective config is made explicit (`Get-LocalBenchArmEffectiveConfig`) and a
  **baseline** arm that inherits harness behaviour — a `LOCALPILOT_*` env var, an
  ambient config file, an enabled plugin/skill, a non-empty system prompt, or
  retrieval left on — is refused by name (`Assert-LocalBenchArmIsolation`, wired
  into `Invoke-LocalBenchContainerGrade` via `-ArmConfig`). The comparative report
  records per-arm isolation provenance and `Get-LocalBenchArmDelta` refuses a delta
  against a contaminated baseline, so a harness-vs-baseline number can be trusted.
- Added the **anti-gaming paired-metric invariant** to the comparative report:
  `Get-LocalBenchArmDeltaVerdicts` annotates each delta so a "win" on a gameable
  metric (smaller diff, fewer tool calls, less redundancy — all improvable by
  under-delivering) only counts when the paired correctness signal (solve rate) is
  present and did not regress (`not-a-win (correctness regressed)` /
  `unverified (no correctness signal)` otherwise). "You wrote less because you did
  less" is caught, not rewarded.
- Added an **instrument self-test gate** to the external runner: every
  deterministic scorer/guard is exercised on a known-good and known-bad reference
  (`Get-LocalBenchInstrumentChecks`), and `Assert-LocalBenchInstrumentsReady`
  refuses a model run (naming the broken instruments) when any fails — wired into
  `Invoke-LocalBenchContainerGrade` before the solver runs.
- Added **offline rescore**: `Save-LocalBenchCell` preserves each run cell (raw
  scorecard + arm metadata) and `Invoke-LocalBenchRescore` rebuilds the comparative
  report from kept cells with no solver run, so a metric/scorer change is re-applied
  without re-paying the API.
- Added a **cheap-prompt control arm** (`Get-LocalBenchControlArm`): a distinct,
  non-baseline arm carrying a one-line system-prompt nudge, so the harness must beat
  a cheap instruction rather than only an empty baseline.

## v0.3.0-beta.3 - 2026-06-18

Coordinated LocalX beta release.

- Added the Tool Discipline Score (TDS) comparative report surface, and
  refreshed the memory-quality sample report with the new extraction fixture and
  the model-extraction lift block.

### 2026-06-17 - Documentation restructure

- Split the long top-level `README.md` into a lean overview plus owned `docs/`
  pages (command surface, tuning guide, TUI, repository layout), indexed by
  `docs/README.md`.
- Documented the `memory-quality` command in the command surface (it shipped in
  beta.2 but was missing from the README).
- Added an in-repo wiki source (`docs/wiki/`) that is one-way CI-synced to the
  GitHub Wiki, plus an offline link check over the docs.

## v0.3.0-beta.2 - 2026-06-15

Coordinated LocalX beta release.

- Added a `memory-quality` report target: LocalBench now runs the LocalMind
  memory-quality evaluation (`localmind eval --json`) and renders a pass/fail
  report, making it the single evidence product for runtime **and** memory
  quality.
- Added consumer-side schema conformance: the tuner best-config store LocalBox
  consumes is validated against the same versioned schema LocalBench emits, so
  producer and consumer cannot drift unnoticed.
- Split the tuner reporting/display helpers out of `55-tuner.ps1` into
  `56-tuner-reporting.ps1` (behaviour unchanged).

## v0.3.0-beta.1 - 2026-06-12

Coordinated LocalX beta release (first tracked changelog entry).
