# LocalBench versioning

LocalBench carries two distinct kinds of version string. Keeping them separate is
deliberate — conflating them breaks the host integration gate.

## API / compatibility floor (machine-compared)

- The product version in the launcher-contract envelope (`localbench version`
  reports it; the consumer gate compares it).

Floors are compared **numeric dotted, suffix-free**: the consumer gate strips
any prerelease suffix before comparing, so `1.2.1-beta.3` satisfies a `1.2.1`
floor and a train suffix can never silently disable an integration. Bump a
floor only when the consumed surface (the launcher trait, the export schema)
actually changes — independent of the release train.

`api_version` (integer, currently `1`) and `launcher_export_version` (integer,
currently `3`) are separate, coarser compatibility integers and follow the same
machine-compared rule.

## Release train (human-facing)

- `README.md` — the release-train badge.
- Every crate `version` in the workspace + `Cargo.lock`, stamped at each
  coordinated cut.
- `LocalStack/index.html` footer.

These carry the full coordinated train string, including any prerelease
suffix, and track the root `VERSION` file / the ecosystem coordinated release.
They are display/identification strings only.

`check-hub.ps1` guards every release-train source against the canonical `VERSION`
so these cannot drift again; it deliberately does **not** rewrite the API/floor
fields above.
