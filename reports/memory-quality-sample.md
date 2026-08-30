# LocalBench Memory-Quality Report

> Sample report — illustrative numbers, not a measured run. Memory-quality is
> **library-only / unwired**: no `localbench` command emits this report and it
> has no schema. The scorer and fixtures live in LocalMind (`localmind eval
> --json`); `localbench-scoring::memory_quality` only parses and gates that
> payload.

- Retrieval cutoff: recall@5
- Threshold: 0.9
- Result: PASS

| Fixture | Candidates | Precision | Recall | Recall@5 |
|---|---|---|---|---|
| exporter-bugfix | 2 | 1.000 | 1.000 | 1.000 |
| dumped-file-content | 0 | 1.000 | 1.000 | 1.000 |
| lock-order-deadlock | 2 | 1.000 | 1.000 | 1.000 |
| **mean** |  | 1.000 | 1.000 | 1.000 |

## Model-extraction lift (vs deterministic baseline)

| Precision delta | Recall delta | Recall@5 delta |
|---|---|---|
| 0.000 | 0.000 | 0.000 |