# Aider-polyglot harness-arm sweep (operator scripts)

Operator scripts for the **model-pinned** Aider-polyglot sweep: the same local
model is driven through several harness configurations ("arms") so the whole
score delta is attributable to harness quality, not the model.

These are **operator scripts**, not library code: they hard-code the operator's
working tree under `~/.localbench/runs/` and assume a local model server. Treat
them as a recorded, reproducible recipe — copy and adjust the paths for your
machine. They predate the native `localbench arms` command, which now owns the arm-config emitters, the bounded eval-solver spawn, and the container grade loop — prefer it for new sweeps.

## Arms

| Arm | Driver | What it measures |
| --- | --- | --- |
| `baseline` | raw model | single-shot, no harness/tools (the floor) |
| `full` | LocalPilot | agentic harness, learning off (harness vs raw) |
| `warm` | LocalPilot | harness + persistent machine-wide learning (`--learn`, auto-accept, global store) + embedding-backed semantic dedup of the store — "smarter as you use it" |
| `claude-code` | Claude Code | `claude -p` driving the same pinned model (`--max-turns 40`) |

The key deltas: `full − baseline` (harness vs raw), `warm − full` (learning
lift), `warm − claude-code` (LocalPilot-as-shipped vs Claude Code).

## Grade contract

An exercise cell is **solved** only when the in-container test command **exits 0
AND at least one test actually ran** — a solution that compiles but runs no tests
(`rust`: `test result: ok. 0 passed; 0 failed`) proves nothing and is **not** a
pass. `Invoke-Grade` parses the test count from the **full** captured output with
a per-language helper (`Get-TestCount`, recorded + Pester-pinned), summing across
every result line where the runner emits more than one (rust prints a
`test result:` line per binary, so the truncated display tail is often the
doc-test `0 passed` even on a real pass). A run that exits 0 but where no test
count can be parsed **fails closed** (not solved) and logs; the ledger records
`tests_run` alongside `solved`. Every grade runs in a throwaway
`--network=none` container, so model code never has network.

## Prerequisites

- A local model served OpenAI-compatible on `:8080` and via the Anthropic
  no-think proxy on `:11435` (the sweep starts `llmdefaultserve` and probes
  `/health`). The proxy folds in-array `system` messages into the top-level
  `system` field so `claude -p` works against the qwen3 chat template.
- The LocalPilot CLI (`localpilot`) and Claude Code (`claude`) on `PATH`.
- Docker, for grading (each exercise is graded on a throwaway copy in a
  container, so a runaway solution can't corrupt the workspace). Each graded
  container runs `--network=none` — model code never has network. The two
  network-using steps are isolated and run no model code: the optional
  per-language dependency prep (`--network=bridge`, trusted pinned test deps) and
  the one-time cargo-cache warm (below).
- For the Java exercises, a JDK on `PATH` (`JAVA_HOME` set). The shared
  `aider-gradle-cache` volume is warmed once (gradle dist + junit) so each fresh
  workspace compiles + runs fully offline.
- For the Rust exercises, the sweep warms a shared `aider-cargo-cache` cargo
  registry volume once before the rust grades (`Initialize-CargoCache`, only when
  `rust` is in `-Languages`), mirroring the gradle cache. It `cargo fetch`es the
  union of every exercise's declared deps (scanned from each `Cargo.toml`) and a
  curated list of crates an agentic arm commonly adds (rayon, itertools,
  thiserror, num-bigint, …) so the offline rust grade
  (`--network=none` + `CARGO_NET_OFFLINE=true`) builds without false offline-dep
  failures. The dep set is the recorded `Get-RustCargoCacheDeps` (Pester-pinned);
  a dep the cache misses still fails loud in the grade tail (a cargo error), never
  silently. The warm is the **only** network step for rust and is idempotent
  (rebuild the cache any time with `docker volume rm aider-cargo-cache`). The rust
  grade image is current stable (`rust:1.96-slim-bookworm`); the corpus's declared
  deps now resolve to versions whose toolchain floor rose past the old 1.82 image.
- **`warm` arm only — a CPU embedding server.** The warm arm's global store now
  uses embedding-backed semantic dedup (catches paraphrase duplicates), so it
  needs an OpenAI-compatible `/v1/embeddings` endpoint.
  Start it with LocalBox `llmembedserve` (defaults to `:8090`); it runs the embed
  model **on the CPU (`-ngl 0`)**, so it adds **zero GPU VRAM** and the chat model
  stays byte-identical across every arm (the warm-vs-other deltas stay fair). The
  sweep **pre-flights** this endpoint — like the Docker pre-flight — and aborts
  loud (ledger intact) if it can't return a vector; the gate is skipped entirely
  when `warm` is not in `-Arms`, so the other arms are unaffected. Override the
  endpoint with `-EmbedBaseUrl`.

## Scripts

- **`aider-arms.ps1`** — the sweep runner. Resumable via a JSONL ledger
  (`~/.localbench/runs/aider-arms<RunTag>/cells.jsonl`); re-running skips cells
  already recorded. Key params: `-Arms`, `-Languages`, `-RunTag`, `-EvalTimeout`,
  `-Model`, `-EmbedBaseUrl` (the CPU embed endpoint the `warm` arm pre-flights).
- **`status-v1.ps1`** — read-only standings table (solved/total per language per
  arm) + the key deltas + the warm store's accumulated-lesson count + a
  **claude-code wall-clock-timeout caveat** (a cc cell that hit its budget is
  counted unsolved but may be budget-bound, not capability-bound, so the cc rate
  is a floor). GPU-free.
- **`watch-v1.ps1`** — supervises a run: exits when the supervisor dies, all
  agentic cells land, or progress stalls.
- **`reaper.ps1`** — backstop that kills leaked/runaway solution processes. On
  these exercises any host `python`/`node`/`java` invocation (the model running
  its own tests or inline evaluation via `run_shell`) finishes in seconds; one
  alive after `-MaxAgeMin` is a hung/runaway solution (infinite loop / memory
  blowup — a `python -c` runaway can hold ~10 GB and, multiplied, OOM the model
  server). Matches by runtime + age and never touches the no-think proxy.
  LocalPilot's `run_shell` now reaps its own process tree on timeout, so this is
  belt-and-braces for whatever escapes (the grader path, an over-long tool
  timeout).
- **`keepawake.ps1`** — keeps the machine awake for the duration of a long sweep.

## Run

```powershell
# from ~/.localbench (adjust paths for your machine)
# warm arm only: start the CPU embedding server first (no GPU VRAM):
llmembedserve
pwsh ./aider-arms.ps1 -Phase solve -Arms full,warm,claude-code `
  -Languages python,go,rust,cpp,javascript,java -RunTag '-v1' -EvalTimeout 900
pwsh ./status-v1.ps1   # standings any time (read-only)
```
