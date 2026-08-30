# Contributing

Thanks for your interest in LocalBench, part of the [LocalX](https://c0degeek-dev.github.io/LocalStack/) stack.

## Ground rules

- Keep changes scoped and focused; one concern per pull request.
- Match the surrounding code style and naming.
- Update the relevant docs when you change behavior or configuration.

## Building and testing

Rust workspace. Before opening a pull request:

```text
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

The bounded-runner regression is deliberately cross-platform: it creates a
grandchild that inherits stdout/stderr, forces a timeout, verifies the complete
owned tree exits, and proves an unrelated sibling survives. Keep it enabled on
Windows, Linux, and macOS CI when changing process spawning or cleanup.

## Pull requests

- Describe what changed and why, and link any related issue.
- Note how you tested the change.
- Expect review before merge.

## Reporting security issues

Do not open a public issue for a vulnerability. See [SECURITY.md](SECURITY.md).
