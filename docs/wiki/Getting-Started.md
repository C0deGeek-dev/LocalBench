# Getting started

LocalBench benchmarks local llama.cpp models on your machine and exports the
fastest safe launcher profile back to LocalBox. Given a machine, model, runtime,
context target, and quality policy, it answers: can this run comfortably, which
settings are fastest within the quality boundary, and why this profile over the
fastest raw trial.

> **Do not edit on github.com.** This wiki is generated from in-repo Markdown
> under `docs/wiki/` and synced one-way on every push to `main`. Edit the source
> in `docs/wiki/`; web edits are overwritten on the next sync.

## Setup

LocalBench is a single native binary — no PowerShell, .NET, or Python at
runtime. From a checkout (or use a release binary):

```text
cargo install --path crates/localbench --locked
```

> The tuner reads model definitions from a LocalBox install at run time
> (`~/.local-llm`), so a working LocalBox setup is a prerequisite for tuning.

## First commands

```text
localbench version         # the launcher-contract version envelope (JSON)
localbench instruments     # deterministic instrument self-tests
```

Then tune one of your installed model keys:

```text
localbench findbest --model <model-key> --context 64k
```

## Next steps

- [[How-To]] — run a tune, rescore kept cells, read a report.
- [[Examples]] — a full tune → replay walkthrough.
- [[Reference]] — the tuning guide, command surface, and launcher contract.
- [[Troubleshooting]] — common problems and fixes.
