# How-To guides

Task-oriented recipes — each answers a single "how do I…?" against shipped
behaviour at the current `VERSION`. See **[[Getting-Started]]** first.

> **Do not edit on github.com.** This wiki is generated from in-repo Markdown
> under `docs/wiki/` and synced one-way on every push to `main`. Edit the source
> in `docs/wiki/`; web edits are overwritten on the next sync.

## Run a benchmark / tune a model

```text
localbench findbest --model q36plus --context 64k     # default goal: coding-agent latency
localbench findbest --model q36plus --context 64k --budget 20   # bound the search
localbench findbest --model q36plus --context 64k --no-save     # measure only
```

llama.cpp's own memory fitter (`llama-fit-params`, shipped with each engine)
places the starting point, so the tuner no longer finds the VRAM edge by
running out of memory (`--no-oracle` turns that off). It probes past that edge
one step at a time until a step fails, and every later candidate runs at the
edge its own memory shape allows — a lighter KV cache moves more of the model
onto the GPU. The tuner then sweeps batching, flash-attention, memory flags,
SWA, CPU threads, KV-cache types, and n-gram speculative decoding, refines the
edge, re-measures the winner fresh before trusting it, and saves the result.
Batching (and KV types on turbo builds) are first screened in one `llama-bench`
process, so only the best candidates cost a server start (`--no-screen` turns
that off). It measures templated chat with LocalBox's
single-session defaults. Decisive measurements persist in the trial cache, so a
repeated or interrupted tune skips configs it already measured; every attempted
or cached candidate is also recorded in a run manifest under
`~/.local-llm/logs/tuner/`. Full search detail:
[tuning.md](https://github.com/C0deGeek-dev/LocalBench/blob/main/docs/tuning.md).

## Define / choose a profile

`pure` ranks measured end-to-end throughput; `balanced` multiplies that by soft
headroom factors so a config that barely fits loses to a slightly slower one with
breathing room. `--profile both` picks the **better of** the two scores and saves
that one winner (a `both` export lands in a `pure`-tagged slot); run
`--profile balanced` explicitly to save a balanced-tagged winner:

```text
localbench findbest --model q36plus --context 64k --profile balanced
```

LocalBox's guided launcher replays the saved profile (balanced preferred, then
pure). The scoring math and the full flag list are in
[tuning.md](https://github.com/C0deGeek-dev/LocalBench/blob/main/docs/tuning.md);
the CLI in
[command-surface.md](https://github.com/C0deGeek-dev/LocalBench/blob/main/docs/command-surface.md).

## Run the harness arm matrix / rescore offline

```text
localbench arms --spec run-spec.json         # drive every arm x task cell, grade, keep cells
localbench rescore --dir <cells-dir>         # recompute the report from kept cells, no model
localbench uplift --report <file>            # render a saved lesson-uplift report
```

The run-spec and task-set formats, the isolation and clean-room boundaries, and
the grading rules are in
[external-runner.md](https://github.com/C0deGeek-dev/LocalBench/blob/main/docs/external-runner.md).
