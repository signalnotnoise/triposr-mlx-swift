#!/usr/bin/env python3
"""Check docs citations against the working tree and verified_commit.

Line numbers in the docs were read from metadata.verified_commit. This script
fails when a cited file, line range, or anchor does not match the working tree,
when a relative Markdown link is broken, or when a cited source file differs
from that commit.
"""

from __future__ import annotations

import json
import re
import subprocess
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


def check_graph(doc: dict, errors: list[str], cited: set[str]) -> str:
    meta = doc["metadata"]
    commit = meta["verified_commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        errors.append(f"metadata.verified_commit is not a 40-character sha: {commit}")
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
    return commit


def check_markdown(errors: list[str], cited: set[str]) -> None:
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


def collect_spans(doc: dict) -> dict[str, list[tuple[int, int]]]:
    spans: dict[str, list[tuple[int, int]]] = {}

    def add(rel: str, start: int, end: int) -> None:
        spans.setdefault(rel, []).append((start, end))

    for value in doc["location_index"].values():
        for loc in iter_locations(value):
            add(loc["file_path"], loc["line_start"], loc["line_end"])
    for node in doc["nodes"]:
        add(node["file_path"], node["line_start"], node["line_end"])
    return spans


def check_commit(commit: str, doc: dict, errors: list[str]) -> None:
    present = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if present.returncode != 0:
        errors.append(
            f"verified_commit {commit} is not in this checkout. "
            "Fetch enough history (actions/checkout fetch-depth: 0) and retry."
        )
        return
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=ROOT,
    )
    if ancestor.returncode != 0:
        errors.append(f"verified_commit {commit} is not an ancestor of HEAD")
        return
    for rel, spans in collect_spans(doc).items():
        show = subprocess.run(
            ["git", "show", f"{commit}:{rel}"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if show.returncode != 0:
            errors.append(f"{rel} is not in verified_commit {commit}")
            continue
        old = show.stdout.splitlines()
        new = read_lines(rel)
        if new is None:
            continue
        for start, end in spans:
            old_slice = old[start - 1 : end]
            new_slice = new[start - 1 : end]
            if old_slice != new_slice:
                errors.append(
                    f"{rel}:{start}-{end} differs from verified_commit {commit}. "
                    "Re-read those lines and update metadata.verified_commit."
                )


def main() -> int:
    errors: list[str] = []
    if not GRAPH.is_file():
        print(f"missing {GRAPH}", file=sys.stderr)
        return 1
    cited: set[str] = set()
    doc = load_graph()
    commit = check_graph(doc, errors, cited)
    check_markdown(errors, cited)
    if commit:
        check_commit(commit, doc, errors)
    if errors:
        print(f"{len(errors)} documentation check(s) failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(
        f"docs citations match the working tree and verified_commit {commit} "
        f"({len(cited)} files)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
