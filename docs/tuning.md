# Tuning guide

Part of the [LocalBench documentation](README.md).

`localbench findbest` measures candidate llama.cpp server configurations for
one model on your machine and saves the best one as a launcher profile. Every
candidate is measured through `llama-server` — the same binary LocalBox will
actually launch — via the OpenAI-compatible `/v1/chat/completions` endpoint
with a deterministic user message and bounded generation controls. LocalBench
reads visible assistant content and llama.cpp's prompt/decode timing fields;
there is no raw-completion fallback and nothing is approximated with
`llama-bench`. Turboquant candidates run through the forked runtime so
`turbo3` / `turbo4` KV types are measured on the binary that supports them.

Candidate launches reuse LocalBox's settings overlay and single-session
defaults. A candidate's explicit values win, then configured LocalBox settings,
then `--parallel 1` and `--cache-reuse 256`. This keeps the measured KV-cache
shape aligned with the server LocalBox actually exposes.

```text
localbench findbest --model <key> [--context <k>]
                    [--mode native|turboquant|mtpturbo|prism] [--quant <q>]
                    [--profile pure|balanced|both] [--budget <n>]
                    [--beam-width <1..100>]
                    [--runs <n>] [--optimize gen|prompt|both|coding-agent]
                    [--no-save] [--no-cache]
```

Model-file quants and KV-cache encodings are separate axes. `--quant` selects
the GGUF file and is not changed during a single tuning run; `KvK` / `KvV`
tune only the runtime KV-cache representation.

Before any trial, the launcher is gated through the
[launcher contract](launcher-contract.md) (version floor + version triple +
declared target/runtime pairing). A GGUF that is not on disk yet is downloaded
first — through the launcher library's own resumable fetch, the same download a
LocalBox launch performs (`.partial` sidecar, same Hugging Face URL) — with
coarse progress on stderr, so a model straight from the catalog can be tuned
without launching it once. Only the GGUF is fetched; a configured vision
projector or draft model is never pulled by a tuning run.

LocalBench honors the catalog's required engine. For example, `tbonsai27b`
selects the PrismML build automatically; `--mode prism` is accepted explicitly,
while an incompatible override such as `--mode native` is rejected before any
trial starts. AutoBest stores the shared wire spelling `prismml`.

## What gets searched

Phases run in order, spend from one trial budget (`--budget`, default 30,
clamped to `[1, 100]`), and deduplicate by config signature — a candidate is
its serialized config, never a label. After each phase the top candidate
lineages are retained as a beam and every retained lineage is expanded by the
next phase. `--beam-width` controls that retention (default 3, range 1–100);
width 1 is the former single-best behaviour. The saved AutoBest profile records
`searchStrategy: "beam"` and the effective `beamWidth` as search provenance.

The budget is shared, not first-come: three trials are held back for the fresh
verification ladder, and every phase that has not run yet keeps a floor of two.
A wide beam therefore buys breadth *inside* a phase instead of consuming the
phases after it — without that floor the batching sweep alone takes the whole
default budget at width 3. A phase that reaches the reserve prints that it did,
so a phase measuring nothing is never silent.

1. **baseline** — the catalog defaults: the model's KV pair, and for MoE
   models its catalog `NCpuMoe` offload.
2. **vram-fit** — fit the model into VRAM on the lever its architecture
   actually has, read from the GGUF header (expert count and layer count).
   - *MoE models* sweep `--n-cpu-moe`: the baseline value, the smart-seed
     offload candidates (biased by your VRAM, RAM, and the GGUF size),
     stride-5 descent toward the estimated minimum, dense single steps below
     it when the baseline was healthy, and stride-5 ascent to the upper bound.
     A failed baseline derives a climb-higher recovery worklist with exponential
     jumps, bisection fill, and a dense tail; lower values are already known
     unviable. MTP runs apply their VRAM-headroom minimum before scheduling.
     The phase reports scheduled, planned, and budget-skipped coverage counts,
     so a truncated offload sweep cannot read as complete coverage.
   - *Dense models* have no expert lever, so this phase is a recovery ladder
     that runs **only when the baseline OOM'd**: it first shrinks the KV cache
     (turbo pairs, every layer still on the GPU — the fastest fit for a
     context-bound OOM), then lowers `--n-gpu-layers`, halved from the real
     layer count. A dense model whose baseline already starts spends no trials
     here (lowering `-ngl` from full offload only makes it slower). If the GGUF
     header cannot be read the model falls back to the catalog heuristic.
3. **batching** — joint sweep of `(--ubatch-size, --batch-size)` with
   `b >= ub`, pruned by OOM dominance: a pair equal-or-larger on both axes
   than an already-OOM'd pair is never measured.
4. **flash-attn** — flash-attention on vs off, overlaid on the current beam.
5. **memory-flags** — `--mlock`, `--no-mmap`, and both together.
6. **cache-flags** — default SWA/cache behaviour vs `--swa-full`,
   `--cache-prompt`, and both together with `CacheReuse=256`.
7. **threads** — CPU thread sweep, only when the current best actually keeps
   MoE experts on the CPU (`NCpuMoe > 0`); candidates come from the smart
   seeds (your logical-core count, minus headroom for the balanced profile).
8. **kv-types** — KV-cache encoding pairs. Native mode sweeps the model's
   baseline types; turbo-capable modes add `turbo3` / `turbo4` and their
   crosses.
9. **refine** — a symmetric ±1..±5 fine-tune grid around every retained
   `NCpuMoe`, plus unmeasured stride-1 probes down from the lowest stable
   offload edge. Both are clamped to the model's valid range.
10. **verify** — the winner is re-measured fresh. If the verification
    measurement fails, every trace of that config is purged from the history
    and the next-best candidate is verified instead (up to three attempts),
    so a config that cannot start twice is never saved.

Long-context soak validation and the MTP on-vs-off double search from the
previous generation are not part of `findbest` in this release; their
underlying rules (soak evaluation, stress-recovery ladder, MTP-regime merge)
ship as tested library behaviour only.

## Live output

Progress goes to stderr, one line per phase and per trial:

```text
phase: vram-fit
trial 4/30 [vram-fit] KvK=q8_0;KvV=q8_0;NCpuMoe=15 -> 559.4
trial 5/30 [batching] KvK=q8_0;KvV=q8_0;NCpuMoe=15;UbatchSize=1024;BatchSize=2048 -> readiness/readiness_exited_oom (evidence: ...)
winner: KvK=q8_0;KvV=q8_0;NCpuMoe=13 (pure = 581.2, verified: true)
```

Each trial line shows the config **signature** (the `Name=value` pairs that
define the candidate) and its score, or a stable `stage/reason` failure such as
`response/missing_timings` or `content/thinking_only`. When available, the line
links the candidate's unique server log; the final result and terminal error
name the run manifest. The signature keys:

- `NCpuMoe` — llama.cpp `--n-cpu-moe`, MoE expert layers kept on CPU. Lower
  moves more expert work to GPU: faster, more VRAM.
- `NGpuLayers` — llama.cpp `-ngl`, dense-model layer offload count.
- `UbatchSize` / `BatchSize` — llama.cpp `--ubatch-size` / `--batch-size`.
- `Threads` / `ThreadsBatch` — llama.cpp `--threads` / `--threads-batch`.
- `FlashAttn` — flash-attention on/off.
- `Mlock` / `NoMmap` — llama.cpp `--mlock` / `--no-mmap`.
- `KvK` / `KvV` — KV-cache types for keys/values, passed as `-ctk` / `-ctv`.
- `SwaFull` / `CachePrompt` / `CacheReuse` — SWA and prompt-cache flags.

KV-cache values are llama.cpp cache encodings such as `f16`, `q8_0`, or, in
turboquant builds, `turbo3` / `turbo4`. They are not GGUF model quants such
as `Q4_K_M` or APEX variants.

## Scoring

`--optimize` picks what the score follows: `gen` follows decode throughput,
`prompt` follows prefill, `both` balances them, and the default
`coding-agent` is effective end-to-end throughput for a local coding-agent
request — large prompt prefill plus a moderate generated reply. That default
prevents decode-only winners with poor prompt processing from being saved as
"best".

`--runs` (default 3) samples each trial several times; medians are used, and
the run-to-run decode variance feeds the stability scoring.

`pure` and `balanced` differ only in the score they rank by:

- **pure** — measured end-to-end throughput.
- **balanced** — the pure score times soft *headroom factors*: free-VRAM
  margin scored relative to its observed jitter (`z = free_min / sigma_eff`,
  full credit at a comfortable multi-sigma margin, discounted toward ×0.60 as
  the margin approaches the noise), plus comparable factors for free RAM,
  sustained CPU load, and within-run throughput variance. Every factor is
  driven by a measured signal, and **an absent signal never penalizes** — a
  candidate is only discounted for headroom pressure that was actually
  observed.

  > The free-VRAM / free-RAM / CPU-load headroom factors activate only when the
  > runner samples host telemetry during trials. The live server runner does not
  > yet collect it, so a **live** `balanced` run is discounted by within-run
  > throughput variance and by the **cross-phase stability factor** — the
  > stability index is rebuilt from the run's own trial history at the final
  > ranking, so a config that measured fast in one phase and slow in another
  > is penalized as documented. The richer headroom factors are exercised by
  > the scoring tests.

`--profile both` ranks the same measured candidates by the **better of** the
pure and balanced score and exports that single winning entry (it is a
best-of-either selector, not a "save one of each" — a `both` export lands in
one AutoBest slot tagged `pure`). Use `--profile balanced` explicitly to export
a balanced-tagged winner.

A measurement is rankable only when startup succeeded, chat transport and
schema succeeded, prompt/decode timings are finite and positive, and the
assistant returned visible non-degenerate text. A ready server with an HTTP,
schema, timing, empty, thinking-only, or degenerate-content failure remains a
startup success but is explicitly unmeasured. It cannot rank, enter the success
cache, verify, or be saved, and it does not start a VRAM recovery sweep. Only an
actual startup/fit failure or engine OOM evidence enters memory recovery.

A trial whose server **process exits** during startup is recognized the moment
it exits, not at the end of the startup budget: the readiness wait watches the
child alongside the `/health` poll, so an OOM that kills the server seven
seconds in is marked `OOM` in about a second rather than after the full 300 s.
The budget bounds only a server that is *still alive and still loading* — a
large GGUF off a cold disk genuinely needs it. `--startup-timeout <secs>`
(default 300) tunes that budget: lower it on a fast NVMe, raise it on a slow
disk. A trial that hits the budget while its server is still running is the one
worth noticing — it prints a warning suggesting a higher `--startup-timeout`.

## The trial cache

Decisive measurements — a healthy trial or a definite OOM — persist across
runs in `~/.local-llm/tuner/trial-cache-<key>[-<contextKey>].json`, so a
repeated or interrupted tune skips configs it already measured. The cache is
saved crash-safely after every stored measurement (temp file + atomic swap,
previous copy kept as `.bak`).

- A plain startup failure is transient and is always re-measured.
- The `verify` phase never trusts the cache — the winner is always
  re-measured fresh.
- The whole cache is invalidated when anything shaping a measurement changes
  — the model key, context, mode, quant, GGUF file identity (path, size,
  mtime), chat protocol, request/prompt/template shape, required response
  schema, effective session settings/defaults, optimize/profile, samples per
  trial, VRAM, or the tuner version. A mismatch names the differing fields on
  stderr and the run starts fresh. This deliberately invalidates raw-completion
  and auto-parallel measurements.
- `--no-cache` ignores the cache for this run: nothing is read or written.

## Trial diagnostics

Every live or cached attempt appends one JSON record to
`~/.local-llm/logs/tuner/run-<run-id>.jsonl`. It records the phase, ordinal,
candidate signature, cache source, requested overrides, typed outcome,
startup/OOM/usability facts, process status, timing summary, final argv,
authoritative runtime metadata available from `/props`, and the unique live log
path. Requested settings and observed metadata are separate. A known engine
adjustment found only in a warning is stored under `advisory_observations`, not
claimed as effective configuration; unknown values remain unknown.

Live logs use `trial-<run>-<ordinal>-<phase>-<signature-hash>-<port>.log`, so a
later phase or reused port cannot overwrite earlier evidence. Excerpts and
arguments are bounded and credential-like values are redacted; the generated
stress prompt and request body are never placed in the manifest. Each line is
flushed and synced independently, so a killed run retains its earlier records
and a truncated final line can be ignored safely. An `.active` marker protects
interrupted runs from cleanup. Completed runs are bounded to the newest 20;
this retention never touches saved profiles or the trial cache.

## The saved profile

Unless `--no-save` is passed, the winner is merged into
`~/.local-llm/tuner/best-<key>.json` (schema:
`schemas/tuner-best-config.schema.json`, currently `tuner_version` 5). The
store keeps one slot per quant + context + mode + VRAM + prompt-length +
profile + vision combination — vision and text entries never overwrite each
other — and every write re-validates against the schema; a corrupt existing
store is refused, never clobbered. LocalBox's guided launcher replays the
matching entry as the auto-tuned profile.

Version 4 and older entries remain on disk but are ineligible in current
LocalBox. Run `localbench findbest --model <key> --context <k>` to create a
serving-faithful version-5 measurement; a matching current entry can coexist
with or replace only its own store slot.

LocalBench treats context as a benchmark dimension: saved entries record both
the context key and the resolved token count, and replay requires the
matching context key.
