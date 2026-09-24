"""Print one release's section of the changelog, ready to use as release notes.

``docs/changelog.md`` is the single changelog. Each release is a level-2 heading
``## X.Y.Z - YYYY-MM-DD``; changes merged since the last release collect under
``## Unreleased`` until release prep renames that heading. This script feeds the
release chain and its gate:

    python scripts/changelog_section.py 0.4.0          # print the section body
    python scripts/changelog_section.py 0.4.0 --check  # exit 1 if it is missing

``auto-tag.yml`` uses the printed section as the GitHub release notes, and the
changelog workflow runs ``--check`` on pull requests into ``main`` so a release
cannot merge without its section. The changelog's relative links resolve on the
documentation site but not on GitHub, so the printed section rewrites them to
absolute URLs on the published site.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHANGELOG = ROOT / "docs" / "changelog.md"
SITE_URL = "https://eeglab.org/pAMICA/"

# "## 0.3.3 - 2026-09-01" (release) or "## Unreleased".
_HEADING = re.compile(r"^## (?P<title>.+?)\s*$")
_RELEASE_TITLE = re.compile(
    r"^(?P<version>\d+\.\d+\.\d+\S*) - (?P<date>\d{4}-\d{2}-\d{2})$"
)
# A Markdown link target that is neither absolute nor an e-mail address.
_RELATIVE_LINK = re.compile(r"\]\((?!https?://|mailto:)(?P<target>[^)\s]+)\)")


def _site_url(target: str) -> str:
    """Absolute site URL for a link target written relative to ``docs/changelog.md``."""
    path, _, anchor = target.partition("#")
    if not path:  # an in-page anchor of the changelog itself
        page = "changelog/"
    elif path.endswith(".md"):
        page = path[: -len(".md")]
        page = "" if page == "index" else page.removesuffix("/index") + "/"
    else:  # an asset, such as a figure
        page = path
    return SITE_URL + page + (f"#{anchor}" if anchor else "")


def release_titles(text: str) -> list[str]:
    """Every level-2 heading title, in file order."""
    return [m["title"] for line in text.splitlines() if (m := _HEADING.match(line))]


def section(text: str, version: str) -> str | None:
    """The body of ``version``'s section, links made absolute, or ``None``."""
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        m = _HEADING.match(line)
        if m is None:
            continue
        if start is not None:
            end = i
            break
        title = _RELEASE_TITLE.match(m["title"])
        if title is not None and title["version"] == version:
            start = i + 1
    else:
        end = len(lines)
    if start is None:
        return None
    body = "\n".join(lines[start:end]).strip()
    if not body:
        return None
    return _RELATIVE_LINK.sub(lambda m: f"]({_site_url(m['target'])})", body) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print one release's changelog section as release notes."
    )
    parser.add_argument("version", help="release version, for example 0.4.0")
    parser.add_argument("--changelog", type=Path, default=CHANGELOG)
    parser.add_argument(
        "--check", action="store_true", help="only verify that the section exists"
    )
    args = parser.parse_args(argv)
    body = section(args.changelog.read_text(encoding="utf-8"), args.version)
    if body is None:
        print(
            f"{args.changelog}: no non-empty '## {args.version} - YYYY-MM-DD' section. "
            "Release prep renames '## Unreleased' to the version and its date.",
            file=sys.stderr,
        )
        return 1
    if not args.check:
        sys.stdout.write(body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
