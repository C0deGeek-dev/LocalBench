# Repository layout

Part of the [LocalBench documentation](README.md).

LocalBench is a Rust workspace. The three library crates split the concerns
that once lived in a single ~6k-line engine; the app crate owns the CLI and
everything that touches a live process.

```text
crates/
  localbench-scoring/               pure scoring math
    stats.rs                        median/mean/stddev/rounding
    score.rs                        trial score, balanced score + headroom factors,
                                    VRAM headroom (z-margin), stability index
    tds.rs                          Tool Discipline Score: parse, safety gates, deltas
    uplift.rs                       lesson-uplift A/B statistics + injection contract
    memory_quality.rs               memory-quality report gate
  localbench-search/                search strategy (no I/O)
    candidate.rs                    candidate construction, signatures, beam selection
    space.rs                        KV/MoE/batching search space, budgets, recovery
    seeds.rs                        smart seeds from host facts
    probe.rs                        long-context probe policy
    regime.rs                       MTP-regime merge rules
    overrides.rs                    override maps + canonical config signature
  localbench-measure/               measurement support (mockable seams)
    classify.rs                     OOM-signature classification, output quality
    soak.rs                         soak evaluation + stress-recovery ladder
    prompt.rs                       deterministic stress prompt
    cache.rs                        fingerprinted trial cache (crash-safe)
    grade.rs                        per-language test counting, offline grade cache
    container.rs                    named network-isolated plans, exact cleanup commands,
                                    Docker-wedge breaker
    arms.rs                         harness-arm configs + arm-isolation contract
    runner.rs                       scorecard parse, capability report, instruments,
                                    cell persistence + offline rescore
  localbench/                       the binary
    main.rs                         CLI dispatch
    consumer.rs                     launcher-contract consumer gate
    trial.rs                        live trial runner over the launcher + cached runner
    tuner.rs                        findbest phase orchestration
    diagnostics.rs                  append-only per-attempt run evidence + retention
    solver.rs                       bounded process-tree runner + localpilot eval seam
    matrix.rs                       live matrix, container executor + exact compensation
    upliftrun.rs                    task sets, live uplift A/B, session-log audit
    export.rs                       best-config store export + path conventions
    output.rs                       machine-output discipline (stdout/stderr, JSONL)
schemas/                            versioned wire-format schemas (binding)
data/                               benchmark fixtures (long-context prompt,
                                    headroom task set)
examples/                           operator example scripts and sample artifacts
reports/                            sample rendered reports
docs/                               this documentation
```

The default live-response quality boundary is owned by
`localbench-measure::classify`: `QUALITY_MIN_CHARS` and `QUALITY_MIN_WORDS`
define the shared 80-character and 20-word minimums. The binary's live trial
gate consumes those constants directly so measurement helpers and shipped
classification cannot drift to different defaults.

The shared model/launcher tier (`localx-llama-core`, `localx-llama-runtime`,
`localx-eval-core`) comes from the public
[`localx-llama`](https://github.com/C0deGeek-dev/localx-llama) repository as a
rev-pinned git dependency; LocalBox's `localbox-launcher` crate supplies the
real launcher implementation behind the trait. Both git-dependency revisions
are tracked in `Cargo.toml`/`Cargo.lock` and must move in lockstep — two revs
of one shared crate would split the `Launcher` trait into incompatible types.
