```text
                   ▄▄██║
             ▄▄██║ ████║  ██╗      ██████╗  ██████╗ █████╗ ██╗     ██████╗ ███████╗███╗   ██╗ ██████╗██╗  ██╗
       ▄▄██║ ████║ ████║  ██║     ██╔═══██╗██╔════╝██╔══██╗██║     ██╔══██╗██╔════╝████╗  ██║██╔════╝██║  ██║
 ▄▄██║ ████║ ████║ ████║  ██║     ██║   ██║██║     ███████║██║     ██████╔╝█████╗  ██╔██╗ ██║██║     ███████║
 ████║ ████║ ████║ ████║  ██║     ██║   ██║██║     ██╔══██║██║     ██╔══██╗██╔══╝  ██║╚██╗██║██║     ██╔══██║
 ████║ ████║ ████║ ████║  ███████╗╚██████╔╝╚██████╗██║  ██║███████╗██████╔╝███████╗██║ ╚████║╚██████╗██║  ██║
 ████║ ████║ ████║ ████║  ╚══════╝ ╚═════╝  ╚═════╝╚═╝  ╚═╝╚══════╝╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝
╔══════════════════════╗
╚══════════════════════╝
                                        >  Automated LLM Benchmarking  <
```

<div align="center">
  <h1>LocalBench</h1>
  <p><strong>Find the fastest stable local-model settings for your machine.</strong></p>
  <p>
    <a href="docs/command-surface.md">Command reference</a> ·
    <a href="docs/tuning.md">Tuning guide</a> ·
    <a href="https://c0degeek-dev.github.io/LocalStack/">LocalX</a>
  </p>
  <p>
    <img alt="LocalX release train 5.0.0" src="https://img.shields.io/badge/release%20train-v5.0.0-f0b75a?style=flat-square">
    <img alt="Rust" src="https://img.shields.io/badge/platform-Rust-4d8df7?style=flat-square">
    <img alt="llama.cpp runtime" src="https://img.shields.io/badge/runtime-llama.cpp-59636e?style=flat-square">
    <img alt="GitHub stars" src="https://img.shields.io/github/stars/C0deGeek-dev/LocalBench?style=flat-square&amp;label=stars">
  </p>
</div>

LocalBench benchmarks the models and runtimes you already use through
[LocalBox](https://github.com/C0deGeek-dev/LocalBox). It measures real workloads,
checks stability, explains the trade-offs, and exports a profile you can use
again instead of tuning by feel.

| At a glance | |
|---|---|
| **Use it when** | A model runs, but you do not know which settings are actually best |
| **It measures** | Hardware fit, prompt processing, generation, memory pressure, and stability |
| **It produces** | Recommendations, Markdown reports, and LocalBox-compatible AutoBest profiles |
| **Default goal** | Coding-agent latency, where prompt prefill matters as much as generation speed |

## Privacy by design

LocalBench measures your machine locally and writes results for you, not for us.

- **No usage telemetry is sent.** Hardware measurements, prompts, scores, and
  benchmark results are not reported to LocalX or an analytics service.
- **Local endpoints only.** The default tuning and evaluation path runs against
  models on your own machine.
- **You own every result.** Reports, caches, and exported profiles are ordinary
  local files you can inspect, move, keep, or delete.
- **No account required.** Benchmarking does not require a LocalX account or a
  hosted API key.

## Quick start

The quickest install is the LocalX one-liner, which installs `localbench`
alongside the rest of the stack at one version — no Rust toolchain needed:

```sh
# Linux / macOS
curl -fsSL https://raw.githubusercontent.com/C0deGeek-dev/LocalPilot/main/install/install.sh | sh
```

```powershell
# Windows
irm https://raw.githubusercontent.com/C0deGeek-dev/LocalPilot/main/install/install.ps1 | iex
```

The tools are cut as a set and only tested together, so they are installed as a
set; `localpilot update --all` re-runs it. Each release also publishes verified
per-platform archives if you would rather install `localbench` on its own.

LocalBench is a single native binary — no PowerShell, .NET, or Python needed.
Build it from this repository (or use a release binary):

```text
cargo install --path crates/localbench --locked
```

Then tune one of your installed model keys:

```text
localbench findbest --model <model-key> --context <context>
```

The winner is saved to `~/.local-llm/tuner/best-<key>.json`, where LocalBox's
guided launcher replays it as the auto-tuned profile.

> [!NOTE]
> Real tuning still needs a launcher that satisfies LocalBench's
> [launcher contract](docs/launcher-contract.md). LocalBox is the default
> adapter and supplies the model catalog, llama-server arguments, VRAM logic,
> and server lifecycle.

## What question does it answer?

Given a machine, model, runtime, context target, and quality policy, LocalBench
answers:

- Will this combination fit comfortably?
- Which settings are fastest without crossing the quality boundary?
- What should I use for coding, chat, or long-context work?
- Why was this profile chosen over the fastest isolated trial?

```text
machine + model + workload
            │
            ▼
      bounded search ──> stability checks ──> scored recommendation
                                                    │
                                                    └── report + AutoBest profile
```

## Sensible defaults

`localbench findbest` optimizes for `coding-agent` by default. That score
models the end-to-end feel of Claude Code or LocalPilot work, where a large
prompt often dominates latency. Use `--optimize gen` only when decode
throughput is the thing you explicitly care about.

Every candidate is measured through `llama-server` — the same binary LocalBox
will actually launch — as templated `/v1/chat/completions` traffic under the
same single-session launcher defaults. It is never approximated with
`llama-bench` numbers. Typed failures and a per-run manifest keep an unusable
HTTP/schema/content response out of ranking while preserving its evidence.

> [!NOTE]
> `--profile balanced` discounts a winner by its measured risk signals. On a
> live run those are within-run throughput variance and cross-phase stability;
> the richer free-VRAM/RAM/CPU headroom factors need host telemetry the live
> runner does not yet collect, so a live balanced result can track `pure`
> closely (see [Tuning](docs/tuning.md)). The findbest output's `confidence`
> field says which case you got: `full` when every factor saw its input,
> `partial` otherwise.

## Project status

The tuning engine and its search, scoring, and export pipeline are mature and
golden-tested against the pinned scoring behaviour. The binary also carries
the harness-capability benchmark (`arms`, `rescore`) and the lesson-uplift
A/B (`uplift`); live full-matrix runs against a local model are opportunistic.
An arm spec can also name a scripted coach that drives the solver over its MCP
surface, adding an interventions count to the scorecard and report — see
[the coached arm](docs/external-runner.md#the-coached-arm-scripted-coach-over-mcp).

## Documentation

| Topic | Guide |
|---|---|
| CLI commands | [Command surface](docs/command-surface.md) |
| Search phases, scoring, and flags | [Tuning](docs/tuning.md) |
| Capability benchmark and uplift A/B | [External runner](docs/external-runner.md) |
| Launcher boundary | [Launcher contract](docs/launcher-contract.md) |
| Repository structure | [Architecture](docs/architecture.md) |
| Full documentation map | [Docs index](docs/README.md) |

## LocalX

LocalBench is the measurement layer in the
[LocalX toolchain](https://c0degeek-dev.github.io/LocalStack/):

| Project | Role |
|---|---|
| [LocalBox](https://github.com/C0deGeek-dev/LocalBox) | Run local models |
| **LocalBench** | Find fast, stable settings |
| [LocalPilot](https://github.com/C0deGeek-dev/LocalPilot) | Code through the agent harness |
| [LocalMind](https://github.com/C0deGeek-dev/LocalMind) | Turn reviewed sessions into reusable project memory |

Release history lives in [CHANGELOG.md](CHANGELOG.md).

## License

![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/license-PolyForm_Noncommercial_1.0.0-blue.svg)

LocalX-owned source is available under the
[PolyForm Noncommercial License 1.0.0](LICENSE). Commercial use requires a
separate license. See [LICENSING.md](LICENSING.md) for the commercial contact,
the 30 August 2026 licensing boundary, and third-party terms.
