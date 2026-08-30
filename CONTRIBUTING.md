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

## Contributor Certificate of Origin

By submitting a contribution (for example, a pull request) to this
repository, you certify that:

1. the contribution was created in whole or in part by you and you have the
   right to submit it under the terms below; or
2. the contribution is based on prior work that, to your knowledge, is
   covered under an appropriate license, and you have the right to submit
   that work under this license; or
3. the contribution was provided to you by someone who certified (1) or (2)
   and you have not modified it.

You keep copyright in your contribution. You grant David Ben-Yishai and Bram
Hammer a perpetual, worldwide, non-exclusive, royalty-free, irrevocable
license to use, reproduce, modify, prepare derivative works of, publicly
display, publicly perform, sublicense, and distribute your contribution as
part of this project, including under license terms other than this
repository's PolyForm Noncommercial License — for example, a commercial
license. Sign off every commit with `git commit -s`. The resulting
`Signed-off-by` trailer certifies that you agree to this Contributor
Certificate of Origin and the license grant above; this project does not use
a separate signed CLA process beyond that trailer.
