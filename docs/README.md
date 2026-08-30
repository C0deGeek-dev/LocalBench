# LocalBench docs

Documentation index and doc-ownership map. Match a change to its owning doc
before editing; don't restate the same area in two places. The top-level
`README.md` is a lean overview — deep content lives here in `docs/`.

| Area | Owning doc |
|---|---|
| Project overview, status, setup, first commands | top-level `README.md` |
| Command surface — CLI verbs | [`command-surface.md`](command-surface.md) |
| Tuning guide — search phases, scoring, flags | [`tuning.md`](tuning.md) |
| Repository layout | [`architecture.md`](architecture.md) |
| LocalBox launcher contract (canonical boundary) | [`launcher-contract.md`](launcher-contract.md) |
| Capability benchmark + uplift A/B — solver seam, clean-room boundary, container grader | [`external-runner.md`](external-runner.md) |
| Version strings — API floor vs release train | [`versioning.md`](versioning.md) |

## Launcher contract

`launcher-contract.md` defines the launcher surface this tuner depends on: the
`Launcher` trait and its versioned envelope, shared with LocalBox through the
`localx-llama` crates. Cross-repo CI asserts conformance in both directions,
so any doc change describing that surface must match the conformance tests.

## Wiki

User-facing guides (Getting Started, How-Tos, Examples, Troubleshooting) are
authored as in-repo Markdown under `docs/wiki/` and one-way CI-synced to the
GitHub Wiki. The in-repo source is authoritative — never edit pages on
github.com. Wiki Reference pages link these `docs/` pages rather than
duplicating them.

## Changelog & version

Every user-facing change updates the top-level `CHANGELOG.md` in the same
checkpoint. No doc, README, or wiki page may claim behaviour beyond the current
`VERSION`.
