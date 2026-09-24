"""The changelog's release format and the release-notes extractor.

``docs/changelog.md`` is the project's only changelog, and the release chain
reads it: ``scripts/changelog_section.py`` turns one release's section into the
GitHub release notes, and the changelog workflow refuses a release into ``main``
without that section (``.rules/changelog.md``). These tests hold the file to the
format the extractor expects and run the extractor on the real file.
"""

import importlib.util
import re
from datetime import date
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_CHANGELOG = (_REPO / "docs" / "changelog.md").read_text(encoding="utf-8")


def _load():
    path = _REPO / "scripts" / "changelog_section.py"
    spec = importlib.util.spec_from_file_location("changelog_section_script", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


script = _load()
_RELEASE = re.compile(r"^(\d+)\.(\d+)\.(\d+) - (\d{4}-\d{2}-\d{2})$")


def test_every_release_heading_is_versioned_and_dated_newest_first():
    titles = script.release_titles(_CHANGELOG)
    releases = [t for t in titles if t != "Unreleased"]
    assert titles.count("Unreleased") <= 1
    if "Unreleased" in titles:
        assert titles[0] == "Unreleased", "Unreleased belongs above every release"
    parsed = []
    for title in releases:
        m = _RELEASE.match(title)
        assert m, f"release heading {title!r} is not 'X.Y.Z - YYYY-MM-DD'"
        parsed.append((tuple(int(x) for x in m.groups()[:3]), date.fromisoformat(m[4])))
    versions = [v for v, _ in parsed]
    assert versions == sorted(versions, reverse=True), "releases are newest first"
    assert len(set(versions)) == len(versions)
    dates = [d for _, d in parsed]
    assert dates == sorted(dates, reverse=True)


def test_section_returns_one_release_and_stops_at_the_next():
    body = script.section(_CHANGELOG, "0.3.2")
    assert body is not None and body.strip()
    assert "## " not in body.splitlines()[0]
    older = script.section(_CHANGELOG, "0.3.1")
    assert older is not None
    assert older.strip() not in body


@pytest.mark.parametrize("version", ["9.9.9", "0.3", "Unreleased"])
def test_missing_release_has_no_section(version):
    assert script.section(_CHANGELOG, version) is None
    assert script.main([version, "--check"]) == 1


def test_newest_section_gets_absolute_links():
    # The newest section (Unreleased, or the release it became) carries relative
    # links into the docs, which must resolve on the GitHub release page. An
    # Unreleased heading is renamed the way release prep renames it.
    newest = script.release_titles(_CHANGELOG)[0]
    if newest == "Unreleased":
        text = _CHANGELOG.replace("## Unreleased", "## 9.9.9 - 2099-01-01", 1)
        version = "9.9.9"
    else:
        text, version = _CHANGELOG, newest.split(" - ")[0]
    body = script.section(text, version)
    assert body is not None
    assert "https://eeglab.org/pAMICA/" in body, "a relative docs link was rewritten"
    for target in re.findall(r"\]\(([^)\s]+)\)", body):
        assert target.startswith(("https://", "http://", "mailto:")), target


@pytest.mark.parametrize(
    ("target", "url"),
    [
        (
            "guides/amica-differences.md#unmapped-fortran-keywords",
            "https://eeglab.org/pAMICA/guides/amica-differences/#unmapped-fortran-keywords",
        ),
        (
            "#persistence-and-exports",
            "https://eeglab.org/pAMICA/changelog/#persistence-and-exports",
        ),
        ("index.md", "https://eeglab.org/pAMICA/"),
        ("guides/index.md", "https://eeglab.org/pAMICA/guides/"),
    ],
)
def test_site_url(target, url):
    assert script._site_url(target) == url


def test_check_passes_for_a_released_version(capsys):
    assert script.main(["0.3.3", "--check"]) == 0
    assert capsys.readouterr().out == ""


# Edge cases of the extractor, on small Markdown snippets written in the
# changelog's own format.
_SNIPPET = """# Changelog

## Unreleased

- Next change.

## 0.3.10 - 2026-09-10

- **A tenth patch.** See [the guide](guides/backends.md#devices).

## 0.3.1 - 2026-07-19   \n
- Inline code keeps `[text](guides/backends.md)` as written.
- Reference link to [the guide][guide].

[guide]: guides/amica-differences.md#default-settings

```markdown
## 9.9.9 - 2099-01-01
[fenced](guides/backends.md)
```

## 0.3.0 - 2026-07-18

- The last section.
"""


def test_version_prefix_does_not_match_a_longer_version():
    tenth = script.section(_SNIPPET, "0.3.10")
    first = script.section(_SNIPPET, "0.3.1")
    assert tenth is not None and "A tenth patch" in tenth
    assert first is not None and "A tenth patch" not in first
    assert "Inline code" in first and "Inline code" not in tenth


def test_heading_with_trailing_whitespace_is_found():
    assert "0.3.1 - 2026-07-19" in script.release_titles(_SNIPPET)


def test_code_spans_and_fences_are_left_as_written():
    body = script.section(_SNIPPET, "0.3.1")
    assert body is not None
    assert "`[text](guides/backends.md)`" in body
    assert "[fenced](guides/backends.md)" in body


def test_heading_inside_a_fence_is_not_a_release():
    assert "9.9.9 - 2099-01-01" not in script.release_titles(_SNIPPET)
    assert script.section(_SNIPPET, "9.9.9") is None
    body = script.section(_SNIPPET, "0.3.1")
    assert body is not None and body.rstrip().endswith("```")


def test_reference_definitions_are_rewritten():
    body = script.section(_SNIPPET, "0.3.1")
    assert body is not None
    assert (
        "[guide]: https://eeglab.org/pAMICA/guides/amica-differences/#default-settings"
        in body
    )


def test_inline_links_are_rewritten_and_the_last_section_is_found():
    tenth = script.section(_SNIPPET, "0.3.10")
    assert tenth is not None
    assert "](https://eeglab.org/pAMICA/guides/backends/#devices)" in tenth
    last = script.section(_SNIPPET, "0.3.0")
    assert last == "- The last section.\n"


def test_empty_section_is_missing():
    empty = "## 0.2.0 - 2026-07-18\n\n## 0.1.0 - 2026-07-11\n\n- First.\n"
    assert script.section(empty, "0.2.0") is None
    assert script.section(empty, "0.1.0") == "- First.\n"
