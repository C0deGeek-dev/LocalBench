# Harness capability — ablation headline (first-party corpus)

> Sample report — illustrative numbers, not a measured run. The model is held
> fixed across every arm (temp 0, fixed seed, N-seed repeats), so each delta
> grades the **harness**, not the model. Rendered by the capability-report and
> attribution helpers in `localbench-measure`.

## Per-arm comparison

| arm | model | solved | solve rate | avg tool calls | avg redundant | judge |
|---|---|---|---|---|---|---|
| full | local-coder | 8/10 | 80% | 5.9 | 0.20 | 4.20 |
| baseline | local-coder | 4/10 | 40% | 9.8 | 1.60 | 3.40 |
| no-retrieval | local-coder | 6/10 | 60% | 7.4 | 0.50 | 3.90 |
| no-code_intelligence | local-coder | 6/10 | 60% | 6.8 | 0.40 | 4.00 |
| no-tool_budget | local-coder | 8/10 | 80% | 8.1 | 0.30 | 4.20 |
| no-check_before_launch | local-coder | 8/10 | 80% | 6.0 | 0.20 | 4.15 |
| no-tool_pull_discovery | local-coder | 7/10 | 70% | 6.2 | 0.90 | 4.10 |

## full vs baseline (delta, model pinned)

| metric | baseline | full | delta |
|---|---|---|---|
| solveRate | 0.40 | 0.80 | +0.40 |
| avgToolCalls | 9.80 | 5.90 | −3.90 |
| avgRedundant | 1.60 | 0.20 | −1.40 |
| judgeOverall | 3.40 | 4.20 | +0.80 |

## Per-feature attribution

Each feature is compared against the `no-<feature>` arm: did removing it move the
process signal it is supposed to move?

| feature | signal | full | ablated | verdict |
|---|---|---|---|---|
| retrieval | retrieval_used | 1.00 | 0.40 | moves signal |
| code_intelligence | reproduce_before_fix | 0.90 | 0.55 | moves signal |
| tool_budget | tool_calls | 5.90 | 8.10 | moves signal |
| check_before_launch | reproduce_before_fix | 0.90 | 0.90 | INERT |
| tool_pull_discovery | redundant_calls | 0.20 | 0.90 | moves signal |

> Reading: `retrieval` and `tool_pull_discovery` clearly earn their place
> (removing them moves their signal and drops the solve rate). `tool_budget`
> moves tool economy without changing the solve rate — it controls cost, not
> correctness. `check_before_launch` is **INERT** on this corpus (no task names a
> local serve target), an expected null — the attribution surfaces it rather than
> hiding it. Variance bars (N-seed stddev) accompany each number in a live run.
