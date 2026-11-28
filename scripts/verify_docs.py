#!/usr/bin/env python3
"""Check docs citations against the working tree.

verified_commit is informational metadata. This script does not resolve it
and does not require it to be an ancestor of HEAD, so a squash or rebase
merge still passes.

Line-range citations and knowledge-graph ranges are locked by a SHA-256 of
those exact lines. A citation may instead name a Swift symbol
(``Sources/Foo.swift#Type.member``). Symbol citations and graph entries
that name a symbol are resolved in the file and fail only when that symbol
is gone or renamed. Markdown line-range citations must fall inside a locked
range or a symbol's current span.

  python3 scripts/verify_docs.py
  python3 scripts/verify_docs.py --fix
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import doc_citations as cites

ROOT = Path(__file__).resolve().parents[1]
GRAPH = ROOT / "docs" / "knowledge-graph.json"
MARKDOWN = [
    ROOT / "AGENTS.md",
    ROOT / "docs" / "engineering_standards.md",
    ROOT / "docs" / "knowledge-graph.md",
    *sorted((ROOT / "docs" / "wiki").glob("*.md")),
]
CITATION = cites.CITATION
LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
PATH_PREFIXES = ("Sources/", "Tests/", "docs/", ".github/")
PATH_NAMES = {"Package.swift", "README.md", ".gitignore", "AGENTS.md"}


def is_tracked_path(rel: str) -> bool:
    return rel.startswith(PATH_PREFIXES) or rel in PATH_NAMES or Path(rel).name in PATH_NAMES


def read_lines(rel: str) -> list[str] | None:
    path = ROOT / rel
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8").splitlines()


def load_graph() -> dict:
    return json.loads(GRAPH.read_text(encoding="utf-8"))


def save_graph(graph: dict) -> None:
    GRAPH.write_text(json.dumps(graph, indent=2) + "\n", encoding="utf-8")


def resolve_symbol(rel: str, name: str) -> tuple[cites.Symbol | None, str | None]:
    lines = read_lines(rel)
    if lines is None:
        return None, f"missing file {rel}"
    if not rel.endswith(".swift"):
        return None, "not swift"
    return cites.find_symbol(cites.index_swift("\n".join(lines) + "\n"), name)


def symbols_in(rel: str) -> list[cites.Symbol]:
    lines = read_lines(rel)
    if lines is None or not rel.endswith(".swift"):
        return []
    return cites.index_swift("\n".join(lines) + "\n")


def iter_locations(value):
    if isinstance(value, dict) and "file_path" in value:
        yield value
    elif isinstance(value, dict) and "locations" in value:
        for item in value["locations"]:
            yield item


def check_span(rel: str, start: int, end: int, anchor: str | None, label: str, errors: list[str]) -> None:
    lines = read_lines(rel)
    if lines is None:
        errors.append(f"{label}: missing file {rel}")
        return
    if start < 1 or end < start or end > len(lines):
        errors.append(f"{label}: {rel}:{start}-{end} is outside 1..{len(lines)}")
        return
    if anchor and anchor not in "\n".join(lines[start - 1 : end]):
        errors.append(f"{label}: anchor {anchor!r} not in {rel}:{start}-{end}")


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
    actual = cites.hash_lines(lines, start, end)
    if actual != expected:
        errors.append(
            f"{label}: {entry['file_path']}:{start}-{end} content_sha256 "
            f"is {actual}, expected {expected} (content changed or moved; "
            f"run python3 scripts/verify_docs.py --fix if the cited text "
            f"moved unchanged in this file)"
        )


def check_entry(entry: dict, label: str, errors: list[str], locks: list[dict], cited: set[str]) -> None:
    rel = entry["file_path"]
    cited.add(rel)
    symbol = entry.get("symbol")
    anchor = entry.get("anchor")
    if symbol:
        found, err = resolve_symbol(rel, symbol)
        if not found:
            if err and err != "not swift":
                errors.append(f"{label}: {rel}: {err}")
            else:
                # Non-Swift or a token, not a declaration.
                check_span(rel, entry["line_start"], entry["line_end"], symbol if not anchor else anchor, label, errors)
            return
        if anchor and anchor not in "\n".join(
            (read_lines(rel) or [])[found.start - 1 : found.end]
        ):
            errors.append(f"{label}: anchor {anchor!r} not in {rel}#{symbol}")
        locks.append({"file_path": rel, "line_start": found.start, "line_end": found.end})
        return
    check_span(rel, entry["line_start"], entry["line_end"], anchor, label, errors)
    check_hash(entry, label, errors)
    locks.append(
        {
            "file_path": rel,
            "line_start": entry["line_start"],
            "line_end": entry["line_end"],
        }
    )


def check_graph(doc: dict, errors: list[str], cited: set[str]) -> list[dict]:
    meta = doc["metadata"]
    commit = meta.get("verified_commit", "")
    if commit and not re.fullmatch(r"[0-9a-f]{40}", commit):
        errors.append(f"metadata.verified_commit is not a 40-character sha: {commit}")
    locks: list[dict] = []
    for key, value in doc["location_index"].items():
        for loc in iter_locations(value):
            check_entry(loc, f"location_index.{key}", errors, locks, cited)
    for node in doc["nodes"]:
        check_entry(node, f"node.{node['id']}", errors, locks, cited)
    return locks


def citation_is_locked(rel: str, start: int, end: int, locks: list[dict]) -> bool:
    return any(
        lock["file_path"] == rel and lock["line_start"] <= start and lock["line_end"] >= end
        for lock in locks
    )


def collect_markdown_cites() -> dict[str, cites.Citation]:
    found: dict[str, cites.Citation] = {}
    for path in MARKDOWN:
        if not path.is_file():
            continue
        for match in CITATION.finditer(path.read_text(encoding="utf-8")):
            cite = cites.citation_from_match(match)
            if not is_tracked_path(cite.path):
                continue
            found[cite.key] = cite
    return found


def check_markdown(errors: list[str], cited: set[str], locks: list[dict]) -> None:
    for path in MARKDOWN:
        if not path.is_file():
            errors.append(f"missing markdown file {path.relative_to(ROOT)}")
            continue
        text = path.read_text(encoding="utf-8")
        rel_md = path.relative_to(ROOT)
        for match in CITATION.finditer(text):
            cite = cites.citation_from_match(match)
            if not is_tracked_path(cite.path):
                continue
            cited.add(cite.path)
            if cite.is_symbol:
                found, err = resolve_symbol(cite.path, cite.symbol or "")
                if not found:
                    errors.append(f"{rel_md}: {cite.key}: {err}")
                continue
            start, end = cite.line_start, cite.line_end
            check_span(cite.path, start, end, None, str(rel_md), errors)
            if not citation_is_locked(cite.path, start, end, locks):
                errors.append(
                    f"{rel_md}: {cite.key} is not inside a content-hashed range or symbol span"
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


def rewrite_text(text: str, replacements: dict[str, str]) -> tuple[str, int]:
    count = 0

    def repl(match: re.Match) -> str:
        nonlocal count
        cite = cites.citation_from_match(match)
        if not is_tracked_path(cite.path):
            return match.group(0)
        new = replacements.get(cite.key)
        if not new or new == cite.key:
            return match.group(0)
        count += 1
        return new

    return CITATION.sub(repl, text), count


def apply_replacements(replacements: dict[str, str]) -> int:
    total = 0
    for path in MARKDOWN:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        new, n = rewrite_text(text, replacements)
        if n:
            path.write_text(new, encoding="utf-8")
            total += n
    return total


def upgrade_markdown(cite: cites.Citation) -> cites.Citation | None:
    if cite.is_symbol or not cite.path.endswith(".swift"):
        return None
    symbol = cites.best_symbol_upgrade(symbols_in(cite.path), cite.line_start, cite.line_end)
    if symbol is None:
        return None
    return cites.Citation(path=cite.path, symbol=symbol.name)


def apply_symbol_to_entry(entry: dict, symbol: cites.Symbol) -> None:
    entry["symbol"] = symbol.name
    entry["line_start"] = symbol.start
    entry["line_end"] = symbol.end
    entry.pop("content_sha256", None)


def fix_entries(entries: list[tuple[str, dict]], unresolved: list[str]) -> int:
    changed = 0
    for label, entry in entries:
        rel = entry["file_path"]
        if entry.get("symbol"):
            found, _ = resolve_symbol(rel, entry["symbol"])
            if found and (entry.get("line_start") != found.start or entry.get("line_end") != found.end):
                entry["line_start"] = found.start
                entry["line_end"] = found.end
                changed += 1
            continue
        lines = read_lines(rel)
        if lines is None:
            unresolved.append(f"{label}: missing file {rel}")
            continue
        start, end = entry["line_start"], entry["line_end"]
        expected = entry.get("content_sha256")
        in_range = 1 <= start <= end <= len(lines)
        current = cites.hash_lines(lines, start, end) if in_range else None
        if expected and (not in_range or current != expected):
            matches = cites.find_hash_windows(lines, expected, end - start + 1)
            if len(matches) == 1:
                entry["line_start"], entry["line_end"] = matches[0]
                entry["content_sha256"] = expected
                start, end = matches[0]
                print(f"retarget {label} {rel}:{start}-{end}")
                changed += 1
            elif len(matches) == 0:
                # Content changed. Upgrade only if the (possibly stale) range
                # still clearly names a symbol, or the original start line
                # still starts one.
                symbol = cites.best_symbol_upgrade(symbols_in(rel), start, end) if in_range else None
                if symbol:
                    print(f"migrate  {label} {rel}:{start}-{end} -> #{symbol.name} (content changed; symbol still found)")
                    apply_symbol_to_entry(entry, symbol)
                    changed += 1
                    continue
                unresolved.append(f"{label}: {rel}:{start}-{end} cited text is gone or changed; not guessing")
                continue
            else:
                windows = ", ".join(f"{a}-{b}" for a, b in matches)
                unresolved.append(
                    f"{label}: {rel}:{start}-{end} cited text occurs {len(matches)} times ({windows}); not guessing"
                )
                continue
        symbol = cites.best_symbol_upgrade(symbols_in(rel), start, end)
        if symbol:
            print(f"migrate  {label} {rel}:{start}-{end} -> #{symbol.name}")
            apply_symbol_to_entry(entry, symbol)
            changed += 1
        elif in_range:
            entry["content_sha256"] = current
    return changed


def all_graph_entries(graph: dict) -> list[tuple[str, dict]]:
    entries: list[tuple[str, dict]] = []
    for key, value in graph["location_index"].items():
        for loc in iter_locations(value):
            entries.append((f"location_index.{key}", loc))
    for node in graph["nodes"]:
        entries.append((f"node.{node['id']}", node))
    return entries


def fix(graph: dict) -> int:
    unresolved: list[str] = []
    graph_changes = fix_entries(all_graph_entries(graph), unresolved)
    if graph_changes:
        print(f"updated {graph_changes} knowledge-graph entr(y/ies)")

    citations = collect_markdown_cites()
    replacements: dict[str, str] = {}
    for key, cite in sorted(citations.items()):
        if cite.is_symbol:
            continue
        lines = read_lines(cite.path)
        if lines is None:
            unresolved.append(f"{key}: missing file")
            continue
        start, end = cite.line_start, cite.line_end
        in_range = 1 <= start <= end <= len(lines)
        # Markdown line cites are not stored in a global hash table; they are
        # locked by graph ranges. Retarget is therefore only a symbol upgrade
        # or a no-op (graph --fix already moved locks).
        if in_range:
            upgraded = upgrade_markdown(cite)
            if upgraded:
                print(f"migrate  {key} -> {upgraded.key}")
                replacements[key] = upgraded.key

    apply_replacements(replacements)
    save_graph(graph)
    if unresolved:
        print("unresolved citations:", file=sys.stderr)
        for item in unresolved:
            print(f"  {item}", file=sys.stderr)
        return 1
    remaining = collect_markdown_cites()
    print(
        f"fixed citations; "
        f"{sum(1 for c in remaining.values() if not c.is_symbol)} line-range citations, "
        f"{sum(1 for c in remaining.values() if c.is_symbol)} symbol citations"
    )
    return 0


def main() -> int:
    if not GRAPH.is_file():
        print(f"missing {GRAPH}", file=sys.stderr)
        return 1
    graph = load_graph()
    if "--fix" in sys.argv:
        status = fix(graph)
        if status != 0:
            return status
        graph = load_graph()

    errors: list[str] = []
    cited: set[str] = set()
    locks = check_graph(graph, errors, cited)
    check_markdown(errors, cited, locks)
    if errors:
        print(f"{len(errors)} documentation check(s) failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"docs citations match the working tree ({len(cited)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
