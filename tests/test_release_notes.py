"""Tests for the release-notes generator's parsing.

The generator runs unattended in CI, so the two functions that read other
people's text are pinned here: a changelog heading that stops matching would
otherwise silently drop upstream's summary from every release.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from make_release_notes import changelog_summary, split_tag  # noqa: E402

CHANGELOG = """# Changelog

## 0.20.6

Faster generation from a new draft head.
Checkpoints are 0.20.5's.

### Changed

- something long

## 0.20.5

Previous entry.
"""


def test_split_tag():
    assert split_tag("0.20.6-sidecar2") == ("0.20.6", "2")


def test_split_tag_rejects_junk():
    # gh api leaks its error body on stdout when no release exists yet.
    for bad in ["", "latest", "Not Found", "0.20.6", "v0.20.6-sidecar1"]:
        assert split_tag(bad) is None, bad


def test_changelog_summary_is_the_leading_paragraph():
    assert changelog_summary(CHANGELOG, "0.20.6") == (
        "> Faster generation from a new draft head.\n> Checkpoints are 0.20.5's."
    )


def test_changelog_summary_missing_version():
    assert changelog_summary(CHANGELOG, "0.99.0") is None


def test_changelog_summary_truncates():
    long = "## 1.0.0\n\n" + "\n".join(f"line {i}" for i in range(40)) + "\n\n## 0.9.0\n"
    out = changelog_summary(long, "1.0.0")
    assert out is not None
    lines = out.splitlines()
    assert lines[0] == "> line 0"
    assert lines[-1].startswith("... truncated")
    assert "line 39" not in out
