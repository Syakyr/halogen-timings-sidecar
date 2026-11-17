# /// script
# requires-python = ">=3.12"
# ///
"""Generate GitHub release notes for a <halogen>-sidecar<N> tag.

Every release of this repo is one upstream bump plus whatever the proxy
changed, so the notes are a pointer plus our own commit list: the base
version, upstream's own words for that version, the published digest, and
the sidecar commits since the previous release. Upstream text is quoted,
never paraphrased, and every lookup degrades to a bare link rather than
failing the release.

Usage:
    uv run scripts/make_release_notes.py --tag 0.20.6-sidecar1 \
        --halogen 0.20.6 --sidecar 1 \
        --image ghcr.io/owner/repo:0.20.6-sidecar1 \
        [--repo owner/repo] [--prev-tag 0.20.5-sidecar1] \
        [--tests "55 passed, coverage 78%"] [--run-url https://...]

Reads GH_TOKEN/GITHUB_TOKEN if present; nothing here needs one beyond the
API quota, since the base repo is public.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request

BASE_REPO = "peonist-ai/halogen-flash-server"
BASE_IMAGE = f"ghcr.io/{BASE_REPO}"
PIN_CHORE = re.compile(r"^chore: pin default base to \d+\.\d+\.\d+$")
MAX_SUMMARY_LINES = 10


def get(
    url: str, token: str | None = None, extra: dict[str, str] | None = None
) -> tuple[bytes, dict[str, str]]:
    headers = {"User-Agent": "halogen-timings-sidecar-release-notes"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    headers.update(extra or {})
    with urllib.request.urlopen(
        urllib.request.Request(url, headers=headers), timeout=45
    ) as r:
        return r.read(), dict(r.headers)


def git(*args: str) -> str:
    """Run git in the checkout; empty string when it cannot (shallow clone,
    missing tag, or no git at all)."""
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=30, check=True
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 - notes must not fail on a git lookup
        return ""


def split_tag(tag: str) -> tuple[str, str] | None:
    """0.20.6-sidecar2 -> ("0.20.6", "2"); None if unparseable, which also
    swallows an API error body leaking in through --prev-tag."""
    m = re.fullmatch(r"(\d+\.\d+\.\d+)-sidecar(\d+)", tag or "")
    return (m.group(1), m.group(2)) if m else None


def ghcr_digest(image_ref: str) -> str:
    """Digest of the manifest just published, so the notes cite the real
    artifact rather than the tag name that could be moved later. Uses an
    anonymous pull token, so it only resolves for a public image; CI passes
    the digest build-push-action already knows and skips this probe."""
    try:
        repo = image_ref.split("/", 1)[1].split(":")[0]
        tag = image_ref.rsplit(":", 1)[1]
        token = json.loads(
            get(f"https://ghcr.io/token?scope=repository:{repo}:pull")[0]
        )["token"]
        _, headers = get(
            f"https://ghcr.io/v2/{repo}/manifests/{tag}",
            token,
            {
                "Accept": "application/vnd.oci.image.index.v1+json,"
                "application/vnd.docker.distribution.manifest.list.v2+json,"
                "application/vnd.oci.image.manifest.v1+json,"
                "application/vnd.docker.distribution.manifest.v2+json"
            },
        )
        return headers.get("Docker-Content-Digest", "unknown")
    except Exception as exc:  # noqa: BLE001
        return f"unavailable ({exc.__class__.__name__})"


def changelog_summary(text: str, version: str) -> str | None:
    """The quoted summary paragraph upstream wrote for `version`, or None when
    the changelog has no such heading."""
    lines = text.splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip() == f"## {version}")
    except StopIteration:
        return None
    body = []
    for ln in lines[start + 1 :]:
        if ln.startswith("## "):
            break
        if body and not ln.strip():
            break
        if ln.strip() or body:
            body.append(ln)
    kept = body[:MAX_SUMMARY_LINES]
    quoted = "\n".join("> " + ln if ln.strip() else ">" for ln in kept)
    if len(body) > MAX_SUMMARY_LINES:
        quoted += "\n... truncated; the full entry is at the link above."
    return quoted


def upstream_section(version: str, token: str | None) -> list[str]:
    """Upstream's own summary paragraph for this version, quoted, or a bare
    changelog link when it cannot be fetched or has no entry yet."""
    link = f"- Changelog: https://github.com/{BASE_REPO}/blob/main/CHANGELOG.md"
    try:
        text = get(
            f"https://raw.githubusercontent.com/{BASE_REPO}/main/CHANGELOG.md", token
        )[0].decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return [link]
    summary = changelog_summary(text, version)
    if summary is None:
        return [link]
    return [
        "Upstream's own summary for this version, from its changelog "
        f"(https://github.com/{BASE_REPO}/blob/main/CHANGELOG.md):\n",
        summary,
    ]


def sidecar_commits(prev: str | None) -> list[str]:
    """Our own commits since the previous release, minus the base-pin chores
    the base-version line already states."""
    if not prev:
        return []
    log = git("log", "--pretty=%h %s", f"{prev}..HEAD")
    if not log:
        return []
    out = []
    for line in log.splitlines():
        subject = line.split(" ", 1)[1] if " " in line else line
        if PIN_CHORE.match(subject):
            continue
        out.append(f"- {line}")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tag", required=True)
    p.add_argument("--halogen", required=True, help="base halogen-flash-server version")
    p.add_argument("--sidecar", required=True, help="sidecar build number for it")
    p.add_argument("--image", required=True, help="full published image ref incl. tag")
    p.add_argument("--repo", default="", help="owner/name of this repo")
    p.add_argument("--prev-tag", default="", help="previously published tag, if any")
    p.add_argument("--tests", default="", help="test summary line from CI")
    p.add_argument(
        "--digest",
        default="",
        help="published manifest digest; probed from GHCR when omitted",
    )
    p.add_argument("--run-url", default="", help="URL of the build run")
    a = p.parse_args()

    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    repo = a.repo or git("config", "--get", "remote.origin.url").removeprefix(
        "https://github.com/"
    ).removesuffix(".git")
    commit = git("rev-parse", "--short", "HEAD") or "unknown"
    prev = split_tag(a.prev_tag)

    out: list[str] = []
    out.append(
        f"Timings sidecar for halogen-flash-server {a.halogen}: a stdlib-only "
        "proxy in front of `/v1` that puts llama.cpp-shaped `timings` on the "
        "final SSE chunk, so llama-swap reports real prefill and decode rates "
        "for Halogen.\n"
    )

    out.append("## What moved\n")
    if prev:
        moved = [f"base image {prev[0]} → {a.halogen}"]
        if prev[1] != a.sidecar:
            moved.append(f"sidecar build #{prev[1]} → #{a.sidecar}")
        out.append(f"Since {a.prev_tag}: " + "; ".join(moved) + ".\n")
        ours = sidecar_commits(a.prev_tag)
        if ours:
            out.append("Sidecar commits since then:\n")
            out.extend(ours)
            out.append("")
        else:
            out.append(
                "Nothing of ours moved since then beyond the base bump, so this "
                "proxy is byte-identical to the previous release.\n"
            )
    else:
        out.append(
            "First release published here, so there is no previous tag to diff "
            "against. The base is "
            f"[{BASE_REPO}:{a.halogen}](https://github.com/{BASE_REPO}/tree/"
            f"{a.halogen}).\n"
        )

    out.append("## Base image\n")
    out.append(
        f"`{BASE_IMAGE}:{a.halogen}`, baked in at build time (never a floating tag).\n"
    )
    out.extend(upstream_section(a.halogen, token))
    out.append("")

    out.append("## Provenance\n")
    out.append("| item | value |")
    out.append("|---|---|")
    out.append(f"| image | `{a.image}` |")
    out.append(f"| digest | `{a.digest or ghcr_digest(a.image)}` |")
    out.append(f"| base | `{BASE_IMAGE}:{a.halogen}` |")
    out.append(f"| sidecar commit | `{commit}` |")
    if a.tests:
        out.append(f"| tests | {a.tests} |")
    if a.run_url:
        out.append(f"| build | {a.run_url} |")
    out.append("")
    out.append(
        "Run it with the same devices and env as the base image and point "
        "llama-swap at `:8731`. README and compose topology: "
        f"https://github.com/{repo}/tree/main"
    )
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except urllib.error.HTTPError as exc:
        sys.exit(f"release notes generation failed: HTTP {exc.code} for {exc.url}")
    except urllib.error.URLError as exc:
        sys.exit(f"release notes generation failed: {exc.reason}")
