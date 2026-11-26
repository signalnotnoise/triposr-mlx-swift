#!/usr/bin/env python3
"""Check docs citations against the working tree.

verified_commit is informational metadata. This script does not resolve it
and does not require it to be an ancestor of HEAD, so a squash or rebase
merge still passes. Each knowledge-graph range is locked by content_sha256
of those exact lines. Markdown citations must fall inside a locked range.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GRAPH = ROOT / "docs" / "knowledge-graph.json"
MARKDOWN = [
    ROOT / "AGENTS.md",
    ROOT / "docs" / "engineering_standards.md",
    ROOT / "docs" / "knowledge-graph.md",
    *sorted((ROOT / "docs" / "wiki").glob("*.md")),
]

CITE = re.compile(
    r"`([^`\s:][^`]*?):(\d+)(?:-(\d+))?`"
)
LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def read_lines(rel: str) -> list[str] | None:
    path = ROOT / rel
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8").splitlines()


def check_span(rel: str, start: int, end: int, anchor: str | None, label: str, errors: list[str]) -> None:
    lines = read_lines(rel)
    if lines is None:
        errors.append(f"{label}: missing file {rel}")
        return
    if start < 1 or end < start or end > len(lines):
        errors.append(
            f"{label}: {rel}:{start}-{end} is outside 1..{len(lines)}"
        )
        return
    if anchor and anchor not in "\n".join(lines[start - 1 : end]):
        errors.append(
            f"{label}: anchor {anchor!r} not in {rel}:{start}-{end}"
        )


def iter_locations(value):
    if isinstance(value, dict) and "file_path" in value:
        yield value
    elif isinstance(value, dict) and "locations" in value:
        for item in value["locations"]:
            yield item


def load_graph() -> dict:
    return json.loads(GRAPH.read_text(encoding="utf-8"))


def content_sha256(lines: list[str]) -> str:
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def check_graph(doc: dict, errors: list[str], cited: set[str]) -> list[dict]:
    meta = doc["metadata"]
    commit = meta.get("verified_commit", "")
    if commit and not re.fullmatch(r"[0-9a-f]{40}", commit):
        errors.append(f"metadata.verified_commit is not a 40-character sha: {commit}")
    locks: list[dict] = []
    for key, value in doc["location_index"].items():
        for loc in iter_locations(value):
            cited.add(loc["file_path"])
            check_span(
                loc["file_path"],
                loc["line_start"],
                loc["line_end"],
                loc.get("anchor"),
                f"location_index.{key}",
                errors,
            )
            check_hash(loc, f"location_index.{key}", errors)
            locks.append(loc)
    for node in doc["nodes"]:
        cited.add(node["file_path"])
        check_span(
            node["file_path"],
            node["line_start"],
            node["line_end"],
            node.get("anchor"),
            f"node.{node['id']}",
            errors,
        )
        check_hash(node, f"node.{node['id']}", errors)
        locks.append(node)
    return locks


def check_hash(entry: dict, label: str, errors: list[str]) -> None:
    expected = entry.get("content_sha256")
    if not expected:
        errors.append(f"{label}: missing content_sha256")
        return
    lines = read_lines(entry["file_path"])
    if lines is None:
        return
    start, end = entry["line_start"], entry["line_end"]
    if start < 1 or end < start or end > len(lines):
        return
    actual = content_sha256(lines[start - 1 : end])
    if actual != expected:
        errors.append(
            f"{label}: {entry['file_path']}:{start}-{end} content_sha256 "
            f"is {actual}, expected {expected}"
        )


def citation_is_locked(rel: str, start: int, end: int, locks: list[dict]) -> bool:
    return any(
        lock["file_path"] == rel
        and lock["line_start"] <= start
        and lock["line_end"] >= end
        for lock in locks
    )


def check_markdown(errors: list[str], cited: set[str], locks: list[dict]) -> None:
    for path in MARKDOWN:
        if not path.is_file():
            errors.append(f"missing markdown file {path.relative_to(ROOT)}")
            continue
        text = path.read_text(encoding="utf-8")
        rel_md = path.relative_to(ROOT)
        for match in CITE.finditer(text):
            rel, start_s, end_s = match.group(1), match.group(2), match.group(3)
            # Skip prose that happens to use backticks around a non-path.
            if not (
                rel.startswith("Sources/")
                or rel.startswith("Tests/")
                or rel.startswith("docs/")
                or rel.startswith(".github/")
                or rel in {"Package.swift", "README.md", ".gitignore", "AGENTS.md"}
            ):
                continue
            start = int(start_s)
            end = int(end_s) if end_s else start
            cited.add(rel)
            check_span(rel, start, end, None, f"{rel_md}", errors)
            if not citation_is_locked(rel, start, end, locks):
                errors.append(
                    f"{rel_md}: {rel}:{start}-{end} is not inside a content-hashed range"
                )
        for match in LINK.finditer(text):
            target = match.group(1).strip()
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            target = target.split("#", 1)[0]
            if not target:
                continue
            resolved = (path.parent / target).resolve()
            try:
                resolved.relative_to(ROOT.resolve())
            except ValueError:
                errors.append(f"{rel_md}: link escapes the repo: {target}")
                continue
            if not resolved.exists():
                errors.append(f"{rel_md}: broken link {target}")


def main() -> int:
    errors: list[str] = []
    if not GRAPH.is_file():
        print(f"missing {GRAPH}", file=sys.stderr)
        return 1
    cited: set[str] = set()
    doc = load_graph()
    locks = check_graph(doc, errors, cited)
    check_markdown(errors, cited, locks)
    if errors:
        print(f"{len(errors)} documentation check(s) failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"docs citations match the working tree by content hash ({len(cited)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
