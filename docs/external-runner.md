# Capability benchmark and uplift A/B

Part of the [LocalBench documentation](README.md).

LocalBench can drive **LocalPilot** (the *solver-under-test*) headless against
benchmark workspaces, grade each task in an isolated container, and render a
comparative capability report. The harness and the per-task scorecard belong
to LocalPilot; LocalBench owns running the solver per task, the clean-room
path boundary, the container grader, and the report. Surface:
`localbench arms` / `localbench rescore` / `localbench uplift`
(implementation: `crates/localbench/src/{solver,matrix,upliftrun}.rs` over
`crates/localbench-measure`).

## The run spec (`arms --spec`)

One JSON file describes the whole matrix:

```json
{
  "schema": 1,
  "model": "<model key>",
  "corpus": "external",
  "arms": [
    { "arm": "baseline", "config": { "is_baseline": true } },
    { "arm": "full" },
    { "arm": "verify", "verify": true },
    { "arm": "warm", "learn": true }
  ],
  "tasks": [
    {
      "id": "astropy__astropy-12907",
      "workspace": "C:/bench/work/astropy-12907",
      "problem": "<the task statement handed to the solver>",
      "test_command": "python -m pytest -q",
      "image": "swebench/task:astropy__astropy-12907",
      "grade_label": "python"
    }
  ]
}
```

- `corpus` is `external` (default) or `first-party`.
- An arm's optional `config` records its effective channels
  (`is_baseline`, `env`, `config_file`, `plugins`, `system_prompt`,
  `retrieval`) so its isolation can be checked; `verify` enables the
  verify-before-done gate for that arm's runs and `learn` closes each run
  out into review-gated memory (the warm/teaching arm).
- A task's `image` overrides the per-task SWE-bench image convention;
  `grade_label` picks the test-counting language (`rust`, `python`, `cpp`,
  `javascript`, `go`, `java`; anything else uses the generic fail-closed
  counter). Rust tasks must set `cargo_cache` to a shared Cargo registry path;
  the run spec fails before spending when it is absent or empty.
  Before the first graded cell using a cache, LocalBench scans the surrounding
  `rust/exercises/practice` corpus, writes a deterministic warm manifest, and
  runs `cargo fetch` outside the graded window. Later cells using the same
  cache and manifest skip that fetch; the grade itself mounts the cache
  read-only with networking disabled.

Every (arm × task) cell drives `localpilot eval` headless in the task
workspace under a wall-clock bound (`--solver-timeout`, default 600 s); a
hung or failing cell is killed and recorded as an unsolved card, never
aborting the matrix.

## Clean-room boundary

> **The copied public corpus is never written under a LocalPilot checkout.**
> A LocalPilot tree is governed by a clean-room provenance policy: no
> benchmark corpus, fixture, or task instance copied from an external source
> may enter it. External corpora live in a user-local, git-ignored cache
> outside any LocalPilot tree and are never committed.

The boundary is enforced in code, not by convention: any task workspace whose
path contains a `LocalPilot` segment is refused before a single solver run.
It is the durable counterpart of LocalPilot's clean-room provenance policy
(`docs/00-clean-room.md`, ADR-0033 in that repo).

## Arm-isolation boundary

> **A capability delta only grades the harness if the baseline arm ran with
> the harness OFF.** The hazard is a baseline arm silently inheriting harness
> behaviour through an ambient channel, so the baseline secretly runs the
> solver-under-test and the harness-vs-baseline delta collapses toward zero.

The isolation contract is **checked against each arm's declared config**, not
against a toml LocalBench itself writes: an arm's effective harness config is
applied by `localpilot eval --arm` (plus the `--verify`/`--learn` flags), so the
canonical per-arm `.localpilot.toml` / `.localmind.toml` the emitters produce is
what an operator stages, not something the live driver injects. Against that
declared config the runner enforces three things at setup, before any solver
run:

- A **baseline** arm must carry a `config` block — otherwise its isolation
  cannot be checked, so it is refused rather than allowed to anchor a delta
  unchecked.
- A baseline arm that declares any harness behaviour — a `LOCALPILOT_*` env
  var, an ambient config file, an enabled plugin, a non-empty system prompt, or
  retrieval left on — is refused, naming the exact leaked channel.
- The declared clean config is cross-checked against the arm's **actual solver
  invocation**: a baseline that also sets `verify`/`learn` is refused, because
  those flags reach `localpilot eval` as `--verify`/`--learn` (harness
  behaviour), contradicting the "harness off" the config asserts.

A harness arm is always allowed; it is *meant* to carry harness behaviour. The
report records each arm's isolation as provenance (`clean`,
`CONTAMINATED: <channels>`, or `n/a (harness arm)`), and a delta against a
contaminated baseline is refused rather than reported as a quietly-invalid
number.

## Instrument self-test gate

Before any model run, the deterministic instruments are proved: each
scorer/guard is exercised on a **known-good** and a **known-bad** reference
and is `ok` only when the good passes and the bad is caught. The matrix
refuses to start, naming the broken instruments, when any fails —
"instruments broken, refuse to spend." `localbench instruments` runs the same
checks standalone.

## Container grader

Each cell's grade runs in an isolated container: `docker run --rm
--network=none` with the task workspace mounted **read-only**, so a solve can
neither fetch its way to green nor mutate the graded tree. Before the first
grade, a pre-flight health check proves the engine actually *runs* a
container to completion (`docker info` can answer while `docker run` hangs).

- **Grade fidelity:** a cell passes only on container exit 0 **and** tests
  that actually ran (counted per `grade_label`); exit 0 with zero tests is
  not a pass.
- **Timeouts are caveats, not results:** a grade that hits its wall-clock
  bound (`--grade-timeout`, default 900 s) counts unsolved and the report
  carries a fairness caveat — the arm's rate reads as a floor.
- **Timeout cleanup is exact and bounded:** solver and Docker CLI commands own
  a process group/tree, so descendants that inherit output handles cannot keep
  the matrix blocked. Health, Cargo-warm, and grade containers have unique
  names; after a timeout or ambiguous CLI failure LocalBench runs
  `docker rm -f <exact-name>`. It never lists or kills containers by a shared
  prefix. A cleanup failure is logged as a secondary infrastructure caveat and
  does not replace the primary timeout or grade verdict. When process-tree
  cleanup fails, captured timeout output is explicitly identified as a bounded
  snapshot that may be truncated.
- **The Docker-wedge breaker:** three consecutive grade timeouts trip a
  circuit breaker and the run yields with its cell ledger intact instead of
  silently burning hours against a wedged engine.
- `--grade none` skips container grading; the solver's self-reported result
  stands.

If Docker is unhealthy, the pre-flight error names the failed health run and
its exact cleanup result. Repair or restart the engine before retrying. When
investigating a cleanup diagnostic, inspect only the exact container name it
reports; do not enumerate or delete containers by the shared LocalBench prefix.

## Offline rescore (never pay twice for a measurement)

Every cell — the raw JSON (graded verdict included) plus arm metadata, and a
synthetic unsolved cell for any failed or hung solve — is persisted under
`--cells-dir`. With no `--cells-dir`, each run writes to its own subfolder
(`~/.localbench/runs/arms-<model>-<timestamp>/`) so one run's cells never mix
with another's; the run prints the resolved directory. `localbench rescore --dir
<cells-dir>` re-parses the kept cells, regroups by arm, and rebuilds the report
deterministically — no model call, no spend. The persisted verdicts match
the live run, so a rescore of a completed matrix reproduces its report.

## Comparative report

The report (schema `localbench-capability-v1`,
`schemas/localbench-capability-v1.schema.json`) compares harness arms with
the model held fixed: tasks, solved, solve rate, average tool calls,
redundancy, diff size, the optional LLM-judge overall, and each arm's
isolation provenance. Public-benchmark numbers are **contamination-suspect**:
the report flags the caveat and is meant to be read as **deltas between
arms**, never as trusted absolutes. A sample rendering is in
[`reports/capability-sample.md`](../reports/capability-sample.md).

### Anti-gaming paired-metric invariant

Fewer tool calls, a smaller diff, less redundancy: each "improves" by doing
**less**, so a lower number can mean the arm under-delivered. The delta
verdicts enforce that **a win on a gameable metric only counts when the
paired correctness signal (solve rate) is present and did not regress**:
`improvement` only when solve rate held, `not-a-win (correctness regressed)`
when it dropped, `unverified (no correctness signal)` when there is nothing
to pair against.

### The cheap-prompt control arm

Besides `baseline` (harness off) and the harness arms, a **control-prompt**
arm carries only a one-line system-prompt nudge. It exists so the harness
must beat a cheap instruction, not only an empty baseline — if a one-liner
closes the gap, the report shows it. As a non-baseline arm it passes the
isolation contract.

## The coached arm (scripted coach over MCP)

An arm with a `"coach"` field names a **coach script**, and its cells are
driven through the solver's own MCP serve surface (`localpilot mcp serve`)
instead of a headless eval: LocalBench acts as the MCP client, submits the
problem, polls the event feed, and applies the script's rules — steer on a
matching event, answer a permission ask, or cancel. Closing the drive runs
the solver's normal closeout, so the coach's corrections land as
review-gated lesson candidates in the task workspace, exactly as with a
human-driven session.

```json
{ "arm": "coached", "coach": "coach-scripts/stuck-nudge.json" }
```

A coach script is versioned JSON — rules of `on` (event type, optional
`detail_contains` substring) and exactly one action (`steer` text, `reply`
allow/deny for asks, or `cancel`), each bounded by `max_fires`. The engine is
deterministic: the same event stream always produces the same interventions,
which is what makes a coached cell replayable offline. A frontier-model
coach is a live, opportunistic leg — never the accepted bar.

The cell's scorecard records what the drive observed (tool calls, exit
reason, and the new `interventions` count); pass/fail still comes from the
container grader. The report gains an **avg interventions** column, and its
delta is **gameable**: fewer interventions counts as a win only when solve
rate held — a coach that gives up on a failing run also intervenes less.

### The three-arm learning comparison

"Did coaching transfer into memory" is a three-arm question: `uncoached` vs
`coached` vs `post-lesson-uncoached`. The runner deliberately does **not**
automate the middle step: promotion of the coached arm's lesson candidates
is review-gated by design, and LocalBench never bypasses a memory gate. The
sequence is two invocations around one explicit review:

1. Run `arms` with the `uncoached` and `coached` arms.
2. Review and promote the coached arm's candidates in each task workspace
   (they are labelled `driver-intervention`, and their evidence names the
   coach script's client).
3. Re-run `arms` with a `post-lesson` arm (uncoached, same tasks, same
   workspaces): the solver now has the promoted lessons; its
   interventions-free solve rate against the coached arm's is the transfer
   signal.

## Lesson-uplift A/B (`uplift`)

The uplift eval answers one question with statistics rather than a bare
delta: **does injecting a lesson actually lift the model on tasks it fails
unguided?**

A *headroom task set* (shipping example:
[`data/uplift/headroom-tasks-v1.json`](../data/uplift/headroom-tasks-v1.json))
defines the tasks and the seed lessons:

```json
{
  "schema": 1,
  "name": "headroom-project-conventions-v1",
  "tasks": [
    {
      "id": "migrations",
      "prompt": "In the Larkspur service, what command applies pending database migrations?",
      "expect": { "mode": "substring", "value": "lark db sync" },
      "lesson_ids": ["lesson-migrations"]
    }
  ],
  "lessons": [
    { "id": "lesson-migrations", "body": "...", "category": "ProjectConvention",
      "confidence": 0.9, "tags": ["conventions"] }
  ]
}
```

- `expect` grades deterministically — `substring`, `all-substrings`, or
  `regex`; no LLM judge. The grader proves itself on a known pass/fail pair
  before any spend.
- `lesson_ids` ties each task to the lesson(s) that supply its answer, so
  the injection assertion knows what the arm was supposed to inject.
- `localbench uplift --task-set <file> --emit-seed-pack` prints the pack the
  lesson arm seeds (via `localpilot learning seed`); seeding is the caller's
  job before the run.

`localbench uplift --task-set <file> --workspace <dir> --model <key>` then
runs both arms, `--trials` times each (default 5), driving one
`localpilot print` per task and reading the turn's **memories-used audit**
from the workspace's newest session event log. Two contracts make the result
trustworthy:

- **Injection-void:** the lesson arm must record at least one intended
  lesson id in *every* trial, and the baseline arm must record *no* memories
  in any trial — otherwise the result is VOID, naming the arm and trial.
- **Significance, never a bare delta:** per-trial success rates aggregate to
  mean ± stddev per arm, and the delta must clear the pooled-stddev noise
  band (floored at 0.05) to count as `uplift` or `regression`; otherwise it
  is honestly `no-effect (within noise)`.

> **Per-arm memory staging is the operator's job — the CLI does not do it.**
> The baseline and lesson arms drive `localpilot print` in the **same
> `--workspace`**, and the CLI writes no per-arm `.localmind.toml` and runs no
> seeding. For a valid A/B each arm needs its own memory state: the **baseline**
> arm must run against a clean store with learning off (the
> `localmind_measurement_config` shape — no seeded memory), and the **lesson**
> arm against the seed pack staged with learning on (`localbench uplift
> --emit-seed-pack` → `localpilot learning seed`, the `localmind_warm_config`
> shape). Because both arms share one workspace, staging the seed *between*
> them is a manual step the CLI cannot perform — use the operator example
> scripts to stage each arm's memory. The injection-void contract is the
> backstop: a mis-staged run (baseline retrieved memory, or the lesson arm
> never injected) VOIDs loudly rather than reporting a bogus number. Automating
> the per-arm staging in the CLI is deferred: it needs a live LocalMind/
> LocalPilot store, and a live local-model A/B is opportunistic while the
> offline, deterministic path is the accepted evidence bar.

The report (schema `localbench-uplift-v1`,
`schemas/localbench-uplift-v1.schema.json`) records both arms, the injection
audit, and the significance verdict; `localbench uplift --report <file>`
renders a saved report as Markdown. A sample is in
[`examples/localbench/uplift/sample-uplift.json`](../examples/localbench/uplift/sample-uplift.json).
