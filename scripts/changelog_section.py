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
_FENCE = re.compile(r"^\s*(```|~~~)")
_ABSOLUTE = r"(?!https?://|mailto:)"
# An inline code span (left as written) or an inline link with a relative target.
_INLINE = re.compile(
    r"(?P<code>`+[^`]*?`+)|\]\(" + _ABSOLUTE + r"(?P<target>[^)\s]+)\)"
)
# A reference-style link definition, "[label]: target", with a relative target.
_REFERENCE = re.compile(
    r"^(?P<lead>\s{0,3}\[[^\]]+\]:\s*)" + _ABSOLUTE + r"(?P<target>\S+)(?P<rest>.*)$"
)


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


def _headings(lines: list[str]):
    """``(index, title)`` of each level-2 heading outside fenced code."""
    fenced = False
    for i, line in enumerate(lines):
        if _FENCE.match(line):
            fenced = not fenced
        elif not fenced and (m := _HEADING.match(line)):
            yield i, m["title"]


def release_titles(text: str) -> list[str]:
    """Every level-2 heading title, in file order."""
    return [title for _, title in _headings(text.splitlines())]


def _absolute_links(body: str) -> str:
    """Rewrite relative link targets to site URLs, outside code spans and fences."""
    out = []
    fenced = False
    for line in body.split("\n"):
        if _FENCE.match(line):
            fenced = not fenced
        elif not fenced:
            if m := _REFERENCE.match(line):
                line = m["lead"] + _site_url(m["target"]) + m["rest"]
            else:
                line = _INLINE.sub(
                    lambda m: m[0] if m["code"] else f"]({_site_url(m['target'])})",
                    line,
                )
        out.append(line)
    return "\n".join(out)


def section(text: str, version: str) -> str | None:
    """The body of ``version``'s section, links made absolute, or ``None``."""
    lines = text.splitlines()
    start = None
    end = len(lines)
    for i, heading in _headings(lines):
        if start is not None:
            end = i
            break
        title = _RELEASE_TITLE.match(heading)
        if title is not None and title["version"] == version:
            start = i + 1
    if start is None:
        return None
    body = "\n".join(lines[start:end]).strip()
    if not body:
        return None
    return _absolute_links(body) + "\n"


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
