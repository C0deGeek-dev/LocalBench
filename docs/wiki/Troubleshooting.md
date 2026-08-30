# Troubleshooting & FAQ

Common problems and fixes. Entries match shipped behaviour at the current
`VERSION`.

> **Do not edit on github.com.** This wiki is generated from in-repo Markdown
> under `docs/wiki/` and synced one-way on every push to `main`. Edit the source
> in `docs/wiki/`; web edits are overwritten on the next sync.

## `findbest` can't find models

The tuner reads model definitions from a LocalBox install at run time
(`~/.local-llm`). A working LocalBox setup is a prerequisite; download the
model there first (`localbox` or `localbox launch <model> --dry-run` shows
where the GGUF resolves).

## The contract gate refuses the launcher

`findbest` gates the launcher's version envelope before any trial (version
floor, `api_version`/`launcher_export_version`, declared target/runtime).
`localbox version` prints the envelope; an old LocalBox needs updating before
tuning.

## Every candidate reports a failure

Trial lines use stable `stage/reason` labels. `readiness/readiness_exited_oom`
means the engine supplied real memory-fit evidence, so lowering context, using a
smaller quant, or increasing CPU offload can help. `response/http_status`,
`response/missing_timings`, `content/empty_content`, and similar labels mean the
server started but did not produce a usable chat measurement; those failures do
not enter the VRAM-fit ladder and are never scored or cached as successes.

The terminal message names a JSONL manifest under
`~/.local-llm/logs/tuner/`. Each record points to that attempt's unique server
log when one was launched. Inspect the reported stage/reason and log rather than
deleting the saved profile blindly. The newest 20 completed diagnostic runs are
retained; interrupted runs keep an active marker and are not pruned.

## The saved config is slower than a trial I saw

By design. The winner is re-measured fresh before it is trusted, and a config
whose batching OOM'd is purged with the next-best retried. A
marginally-fitting fast config loses to a roomier, reliable one.

## Re-tuning after a hardware or llama.cpp change

The trial cache fingerprints the model file, tuner version, chat protocol,
request/prompt/template shape, response schema, session defaults/settings, and
measurement knobs; any change invalidates it whole (the mismatch names the
differing fields) so stale numbers are never reused. Current LocalBox accepts
only tuner version 5. Older entries remain on disk but require a re-tune before
AutoBest can replay them.

## Force fresh measurements

```text
localbench findbest --model q36plus --context 64k --no-cache
```

Or delete `~/.local-llm/tuner/trial-cache-<key>[-<context>].json` to clear the
cache entirely. Full flag reference:
[tuning.md](https://github.com/C0deGeek-dev/LocalBench/blob/main/docs/tuning.md).
