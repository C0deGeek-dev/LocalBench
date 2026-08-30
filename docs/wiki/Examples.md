# Examples

Copy-pasteable samples that match shipped behaviour at the current `VERSION`.

> **Do not edit on github.com.** This wiki is generated from in-repo Markdown
> under `docs/wiki/` and synced one-way on every push to `main`. Edit the source
> in `docs/wiki/`; web edits are overwritten on the next sync.

## Tune a model, then replay it in LocalBox

```text
# tune at 64k, save both pure and balanced profiles
localbench findbest --model q36plus --context 64k --profile both

# replay: LocalBox's guided launcher picks the saved profile up automatically
localbox
```

Expect a saved `best-q36plus.json` under `~/.local-llm/tuner/`, recording the
context key and resolved token count plus the export provenance.

## Read the live trial output

During a run LocalBench logs one line per candidate to stderr (the JSON result
stays clean on stdout):

```text
trial 4/30 [vram-fit] KvK=q8_0;KvV=q8_0;NCpuMoe=20 -> 559.4 (evidence: ...trial-...log)
trial 5/30 [batching] KvK=q8_0;KvV=q8_0;BatchSize=2048 -> response/missing_timings (evidence: ...trial-...log)
```

The number is the optimized score (end-to-end for the default coding-agent
goal); a `stage/reason` value is an unrankable typed failure. The bracketed name
is the search phase, semicolon-separated `Name=value` pairs are the candidate
settings, and `evidence` points to its unique log. The final JSON result includes
the run manifest path. Full legend:
[tuning.md](https://github.com/C0deGeek-dev/LocalBench/blob/main/docs/tuning.md).

## Reuse measurements across runs

```text
localbench findbest --model q36plus --context 64k             # first run measures
localbench findbest --model q36plus --context 64k             # re-run reuses decisive trials
localbench findbest --model q36plus --context 64k --no-cache  # force fresh this run
```

The cache invalidates itself — naming the differing fields — when anything
shaping a measurement changes (model file, chat protocol/template, session
defaults, runs, optimize goal, tuner version, …).
