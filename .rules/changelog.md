# Changelog Standards

## One changelog
- **`docs/changelog.md` is the changelog.** It is published at
  <https://eeglab.org/pAMICA/changelog/>, linked from PyPI (`[project.urls]`),
  and its release sections are the GitHub release notes.
- **The root `CHANGELOG.md` only points to it.** Never add entries there.

## Every user-visible change gets an entry, in the same pull request
- A pull request into `dev` that changes user-visible behavior adds its entry under `## Unreleased`.
  User-visible behavior covers results, defaults, parameters, errors, saved formats, supported data, dependencies and installation.
- **`.github/workflows/changelog.yml` enforces it for package code:**
  `pamica/` outside its tests, `validate_implementations.py` and `pyproject.toml`.
- **The `skip-changelog` label is for changes with no user-visible effect:**
  a refactor with byte-identical behavior, or a test-only or tooling-only change.
  When in doubt, write the entry.
- CI, documentation and paper changes may add entries (under `### Continuous integration` or `### Documentation`); the check does not require them.

## What an entry says
- **Lead with the effect on the user,** as a bold one-line summary.
  Then say what changed and why, with the issue or pull-request number.
  Then say what a user has to do, for example refit, pass a setting explicitly, or re-export.
- **Changes to default results go into the release's opening warning** as well
  (the `!!! warning` block at the top of the section), so a user comparing against an earlier version sees them first.
- **Deliberate divergences from the Fortran reference** also get a row in `docs/guides/amica-differences.md`.
- **Keep one coherent account.**
  - File each entry under the existing topic subheadings of `## Unreleased` (`### Fitting follows the reference`, `### Persistence and exports`, ...) rather than one section per pull request.
  - When a later change supersedes an earlier entry in the same release, edit that entry rather than appending a contradiction.
- **Quote measured figures with their conditions:** data, iterations, backend and the reference build.
- **Style:**
  - American English. No em-dashes. Semantic line breaks. Define an abbreviation on first use.
  - Honest, nuanced wording: state what was measured and what it implies, without negation-contrast pairs ("X, not Y") or candor announcements ("plainly").
- **Links:** relative links to docs pages are fine. `scripts/changelog_section.py` rewrites them to site URLs for the release notes.

## Format
- Release headings are `## X.Y.Z - YYYY-MM-DD`, newest first, with at most one `## Unreleased` section above them.
- `pamica/tests/test_changelog.py` enforces the format and runs the release-notes extractor on the real file.

## At release time
1. **Release-prep pull request into `dev`:**
   - Rename `## Unreleased` to `## X.Y.Z - YYYY-MM-DD` (the planned release date).
   - Re-read the section as one release note.
   - For a minor or major release, set the version with `uv run python scripts/sync_version.py sync X.Y.0.dev0` and `uv lock`. Patch releases need no version change; the chain strips `.devN`.
     Avoid the commit subject prefix "Bump version to", which the CI guards reserve for the bot.
2. **Merge `dev` into `main`** with a regular merge commit (`.rules/git.md`).
   The changelog check on that pull request requires the release's section and no leftover `## Unreleased`.
3. **`auto-tag.yml` publishes the section as the GitHub release notes,** with GitHub's generated pull-request list appended.
4. **After the release,** the next change into `dev` starts a fresh `## Unreleased` section above the release.

## Refreshing an existing release's notes
`gh release edit vX.Y.Z --notes-file <(uv run python scripts/changelog_section.py X.Y.Z)`.
Editing a release fires only the `edited` event, which neither `publish.yml` nor `release-binaries.yml` listens to.
