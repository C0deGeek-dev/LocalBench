# Command surface

Part of the [LocalBench documentation](README.md).

LocalBench is a single native binary. Structured results print to **stdout
clean** (JSON by default when stdout is piped; `--format text|json`
overrides), human logs go to **stderr**, and a long `arms` run streams JSONL
lifecycle events — a `started` event (with the total cell count), one
`result` event per persisted cell (carrying that cell), and a terminal
`completed` event (the full outcome) or `error` event (the abort reason) —
so `localbench ... | jq` never fights log noise and a supervising harness
never reads a `started` with no terminal event.

```text
localbench version
    Print the version envelope (JSON): product version, api_version,
    launcher_export_version, supported targets/runtimes.

localbench instruments
    Run the deterministic instrument self-tests (scorecard parser,
    clean-room boundary, arm isolation, paired-metric verdict). Exit 0
    only when every instrument passes its known-good/known-bad pair.

localbench findbest --model <key> [--context <k>]
                    [--mode native|turboquant|prism] [--quant <q>]
                    [--profile pure|balanced|both] [--budget <n>]
                    [--runs <n>] [--optimize gen|prompt|both|coding-agent]
                    [--no-save] [--no-cache] [--no-oracle] [--no-screen]
                    [--startup-timeout <secs>]
    Tune the model live against the launcher (LocalBox) and save the
    winner to ~/.local-llm/tuner/best-<key>.json. Candidates are measured as
    templated chat under LocalBox's single-session defaults. Decisive
    measurements persist in the trial cache and every live/cached attempt is
    recorded under ~/.local-llm/logs/tuner/run-<run-id>.jsonl; the JSON result
    includes that diagnostics path. llama.cpp's memory fitter
    (llama-fit-params, shipped with each engine) places the baseline and
    bounds the VRAM search; --no-oracle searches by trial alone. llama-bench
    screens the batching, flash-attention, thread and KV-type candidates so
    only the best reach a server trial; --no-screen measures every one. The
    JSON result's search_aids names which helpers ran. See the tuning guide.
    Models with a catalog-required engine select it automatically; an
    incompatible explicit --mode is rejected before any trial starts.
    --startup-timeout sets the per-trial startup budget in seconds
    (default 300). A trial whose server process exits is classified the
    moment it exits regardless of this budget; the budget bounds only a
    server that is still alive and still loading, so raise it on a slow
    disk and lower it on a fast NVMe.

localbench arms --spec <run-spec.json> [--cells-dir <dir>]
                [--grade docker|none] [--localpilot <bin>]
                [--solver-timeout <s>] [--grade-timeout <s>]
    Drive the solver-under-test (localpilot eval) across every harness
    arm x task cell, grade each cell in a network-isolated container,
    and keep every cell for offline rescore. Solver and grader subprocesses
    are process-tree bounded; Docker timeout compensation removes only the
    invocation's exact unique container name. Cleanup failures are secondary
    caveats and never overwrite the primary grade outcome. See the external
    runner doc.

localbench uplift --report <file>
    Render a saved lesson-uplift report as Markdown.

localbench uplift --task-set <file> --workspace <dir> --model <key>
                  [--trials <n>] [--localpilot <bin>] [--timeout <s>]
                  [--intended <id,id,...>] [--emit-seed-pack] [--answer-only]
    Run the live lesson-on/off A/B via localpilot print (or print the
    seed pack the lesson arm needs). Seed the lesson arm first; the
    injection contract voids a result when an arm did not inject as
    configured. --timeout bounds each solver turn (default 600 s);
    --timeout 0 sets no bound, for a slow local model. --answer-only supplies
    project context beside the question without tools and binds that solver
    mode into arm files and receipts. Use the same mode on both arms.

localbench uplift --emit-arm-config baseline|lessons
    Print the .localmind.toml an arm's workspace must be staged with:
    learning off for the baseline; learning on, project scope only, for
    the lesson arm.

localbench uplift --task-set <file> --arm baseline|lessons
                  --workspace <dir> --model <key> --binding <id>
                  --out <arm-file> [--trials <n>] [--localpilot <bin>]
                  [--timeout <s>] [--intended <id,id,...>] [--answer-only]
    Run ONE arm in a workspace staged for it and write its arm file. A
    workspace whose .localmind.toml is not the arm's configuration is
    refused before any turn.

localbench uplift --combine <arm-file> --with <arm-file> [--out <file>]
    Join one baseline and one lesson arm file of the same request into
    the identity-bound receipt (localbench-uplift-v2). Exit 3 with a VOID
    receipt when an arm did not inject as configured.

localbench rescore --dir <cells-dir> [--corpus first-party|external]
    Recompute the comparative capability report from kept cells — no
    solver run, no model, deterministic.
```
