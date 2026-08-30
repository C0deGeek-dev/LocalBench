# Aider-polyglot 4-arm sweep — measured result (2026-06-30)

**This is a measured run, not an illustrative sample.** One local model, pinned
and identical across every arm; the number that matters is the *delta between
arms*, because every arm sees the same tasks.

- **Model:** a single pinned local model (Qwen-3.6 35B-A3B class, APEX
  `apex-i-quality` quant), served locally through LocalBox's no-think proxy —
  identical across all four arms. A CPU embedding server backs the `warm` arm's
  semantic dedup.
- **Corpus:** Aider-polyglot, 225 exercises per arm × 4 arms = **900 cells**, six
  languages (python, go, rust, cpp, javascript, java).
- **Grading:** each solution graded in a **network-isolated container**
  (`docker run --network=none`) with that language's real test runner; 600 s per
  exercise (a timeout counts as unsolved); identical cap for every language.
- **Harness:** LocalPilot 1.1.0.

## Arms

| Arm | What it is |
|---|---|
| `baseline` | the raw local model, single-shot, no tools or loop |
| `full` | the LocalPilot agentic harness (tools, in-workspace build/test iteration, verify-before-done) driving that model |
| `claude-code` | **Claude Code (headless) driving the _same local model_** via the Anthropic-style proxy — a harness-vs-harness comparison on one model, **not** frontier Claude |
| `warm` | the `full` harness plus reviewed local learning (memory) |

## Result — solved / total

| lang | total | baseline | full | claude-code | warm |
|---|---|---|---|---|---|
| python | 34 | 13 | 33 | 29 | 34 |
| go | 39 | 11 | 38 | 35 | 38 |
| rust | 30 | 10 | 20 | 20 | 24 |
| cpp | 26 | 2 | 23 | 22 | 24 |
| javascript | 49 | 14 | 49 | 48 | 49 |
| java | 47 | 7 | 44 | 44 | 44 |
| **TOTAL** | **225** | **57 (25%)** | **207 (92%)** | **198 (88%)** | **213 (95%)** |

## Deltas (same pinned model throughout)

- **Harness vs raw** (`full` − `baseline`): **92% − 25% = +67 points**, ≈ **3.7×**
  the bare model's solve rate. The tools + agentic loop + iterate-against-tests
  are what move the number, with the model held constant.
- **LocalPilot vs Claude Code** (`warm` − `claude-code`): **95% − 88% = +7
  points**. Read this precisely: it says LocalPilot's harness + learning beats
  **Claude Code's harness on an identical local model**. It does **not** say
  anything about frontier Claude — both arms drove the same local model.
- **Learning lift** (`warm` − `full`): **95% − 92% = +3 points**.

## Read these honestly

- **The "vs Claude Code" number is a harness comparison on one shared local
  model**, not a comparison against a hosted frontier model. Do not read it as
  "beats Claude."
- **Public-corpus absolutes are contamination-suspect.** Trust the *arm delta*
  (both arms see the same possibly-trained-on tasks), not the absolute solve rate.
- **The solver self-tests on the host.** An early java run scored artificially low
  (9%) because the host lacked a JDK, so the solver was iterating blind; with the
  host toolchain installed it reached 68→94% in line with the other languages.
  When the solver isn't isolated it needs every target language's toolchain
  present, or that language's harnessed score is a host artifact.
- **One model, one quant.** This grades the *harness* on this local model; a
  stronger model would lift every arm.

## Provenance & reproduction

The runner, supervisor, per-language container grading scripts, and the full
900-cell ledger are retained with the maintainers. Reproduce the summary table
from the ledger with the sweep runner's `-Phase summary`. This report is the
public, sanitized summary of that run; the raw per-cell ledger is not published.
