# Harness capability — comparative (2 arm(s), corpus: external)

> Contamination caveat: this is a public benchmark; treat absolute
> numbers as suspect and read them as deltas between harness arms.

| arm | model | solved | solve rate | avg tool calls | avg redundant | avg interventions | judge | isolation |
|---|---|---|---|---|---|---|---|---|
| baseline | example-local-model | 14/25 | 56% | 18.20 | 1.40 | 0.00 | 0.61 | clean |
| coached | example-local-model | 17/25 | 68% | 16.90 | 1.10 | 1.80 | 0.66 | clean |

> Sample report — illustrative numbers, not a measured run. This is the text
> rendering of `localbench arms --format text` (`render_capability_report`); the
> machine-readable form is
> `examples/localbench/capability/sample-capability-report.json` (schema
> `localbench-capability-v1`). The headline is the **delta** between the
> `coached` and `baseline` arms with the model held fixed: the coached arm — a
> scripted coach steering the solver over its MCP surface — solves more tasks
> with fewer, less-redundant tool calls and a higher judge score. The
> `avg interventions` column counts coach steers per task (an uncoached arm
> reports 0.00), and the `isolation` column records each arm's provenance —
> both tracked arms here ran `clean`.
