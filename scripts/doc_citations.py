#!/usr/bin/env python3
"""Shared citation parsing, Swift symbol lookup, and --fix helpers.

Citation forms, all valid in docs and knowledge-graph keys:

    Sources/Foo.swift:12
    Sources/Foo.swift:12-40
    Sources/Foo.swift#Type
    Sources/Foo.swift#Type.member

Line-range citations are checked against a stored SHA-256 of the cited
lines (a trailing newline is included). Symbol citations are resolved by
parsing the Swift file; they fail only when the symbol is gone, renamed,
or ambiguous. Body edits do not fail a symbol citation.

``--fix`` retargets a line range when the stored hash still matches a
unique same-length window in the same file. It does not guess when there
are zero or multiple matches.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

CITATION = re.compile(
    r"(?P<path>"
    r"(?:Sources|Tests|Examples|docs|Vendor|\.github|scripts)/[A-Za-z0-9_./-]+"
    r"|[A-Za-z0-9_.-]+\.(?:swift|py|yml|yaml|md|sh|metal|txt|json)"
    r"|Package\.swift|Package\.resolved|README\.md|run_tests\.sh|\.gitignore|AGENTS\.md"
    r")"
    r"(?:"
    r":(?P<start>\d+)(?:-(?P<end>\d+))?"
    r"|\#(?P<symbol>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)"
    r")"
)

# Access, attributes, and modifiers that may precede a declaration keyword.
_PREFIX = r"""
    (?:
        (?:@[\w.]+(?:\s*\([^)]*\))?|\b(?:public|private|internal|open|fileprivate|package)\b
        |\b(?:final|static|class|mutating|nonmutating|override|indirect|nonisolated
            |async|convenience|required|lazy|weak|unowned|dynamic|optional|consuming
            |borrowing|isolated|distributed|infix|prefix|postfix|discardableResult)\b)
        \s+
    )*
"""

DECL_LINE = re.compile(
    rf"""^
    (?P<indent>\s*)
    {_PREFIX}
    (?P<kind>enum|struct|class|actor|protocol|extension|func|var|let|typealias|init|subscript|case)
    (?:
        \s+(?P<name>[_A-Za-z][_A-Za-z0-9]*
            |`[^`]+`
            |init
            |==|!=|~=|\+|-|\*|/|%|&|\||\^|<<|>>|<> )
    )?
    """,
    re.VERBOSE,
)

# A "case" inside a switch looks like `case .foo`, `case let`, `case 0`, `default`.
SWITCH_CASE = re.compile(r"^\s*case\s+(?:\.(?:[_A-Za-z])|let\b|var\b|_|[0-9\"'(]|nil\b)")


@dataclass(frozen=True)
class Symbol:
    name: str
    kind: str
    start: int
    end: int
    decl_line: int


@dataclass(frozen=True)
class Citation:
    path: str
    start: int | None = None
    end: int | None = None
    symbol: str | None = None

    @property
    def key(self) -> str:
        if self.symbol:
            return f"{self.path}#{self.symbol}"
        assert self.start is not None
        if self.end is None or self.end == self.start:
            return f"{self.path}:{self.start}"
        return f"{self.path}:{self.start}-{self.end}"

    @property
    def is_symbol(self) -> bool:
        return bool(self.symbol)

    @property
    def line_start(self) -> int:
        return int(self.start or 0)

    @property
    def line_end(self) -> int:
        return int(self.end or self.start or 0)


def parse_citation(text: str) -> Citation | None:
    match = CITATION.search(text)
    if not match:
        return None
    return citation_from_match(match)


def citation_from_match(match: re.Match[str]) -> Citation:
    symbol = match.group("symbol")
    if symbol:
        return Citation(path=match.group("path"), symbol=symbol)
    start = int(match.group("start"))
    end_s = match.group("end")
    end = int(end_s) if end_s else start
    return Citation(path=match.group("path"), start=start, end=end)


def format_citation(cite: Citation) -> str:
    return cite.key


def hash_lines(lines: list[str], start: int, end: int) -> str:
    payload = ("\n".join(lines[start - 1 : end]) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def find_hash_windows(lines: list[str], expected: str, length: int) -> list[tuple[int, int]]:
    if length < 1 or length > len(lines):
        return []
    matches: list[tuple[int, int]] = []
    for start in range(1, len(lines) - length + 2):
        end = start + length - 1
        if hash_lines(lines, start, end) == expected:
            matches.append((start, end))
    return matches


def _strip_name(name: str | None) -> str | None:
    if not name:
        return None
    if name.startswith("`") and name.endswith("`") and len(name) >= 2:
        return name[1:-1]
    return name


def _leading_doc_start(lines: list[str], decl_index: int) -> int:
    """1-based start including contiguous doc comments and attributes above decl."""
    i = decl_index  # 0-based
    while i > 0:
        prev = lines[i - 1].strip()
        if not prev:
            break
        if (
            prev.startswith("///")
            or prev.startswith("//")
            or prev.startswith("/*")
            or prev.startswith("*")
            or prev.startswith("@")
            or prev.startswith("#")
        ):
            i -= 1
            continue
        break
    return i + 1


def _code_mask(text: str) -> str:
    """Replace string and comment contents with spaces, preserving newlines and length."""
    out: list[str] = []
    i = 0
    n = len(text)

    def space_out(chunk: str) -> str:
        return "".join("\n" if ch == "\n" else " " for ch in chunk)

    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if ch == "/" and nxt == "/":
            j = i
            while j < n and text[j] != "\n":
                j += 1
            out.append(space_out(text[i:j]))
            i = j
            continue
        if ch == "/" and nxt == "*":
            j = text.find("*/", i + 2)
            if j == -1:
                out.append(space_out(text[i:]))
                break
            out.append(space_out(text[i : j + 2]))
            i = j + 2
            continue
        if ch == '"':
            # Raw string: #"..."# or #"""..."""#
            hashes = 0
            k = i - 1
            while k >= 0 and text[k] == "#":
                hashes += 1
                k -= 1
            if nxt == '"' and i + 2 < n and text[i + 2] == '"':
                # multiline
                close = '"' * 3 + "#" * hashes
                j = text.find(close, i + 3)
                if j == -1:
                    out.append(space_out(text[i:]))
                    break
                out.append(space_out(text[i : j + len(close)]))
                i = j + len(close)
                continue
            # simple string, skip interpolation coarsely by tracking depth
            j = i + 1
            while j < n:
                if text[j] == "\\" and hashes == 0:
                    j += 2
                    continue
                if text[j] == '"' and text[j + 1 : j + 1 + hashes] == "#" * hashes:
                    j += 1 + hashes
                    break
                j += 1
            out.append(space_out(text[i:j]))
            i = j
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def index_swift(text: str) -> list[Symbol]:
    """Return declarations in a Swift source string, outermost first."""
    lines = text.splitlines()
    masked = _code_mask(text)
    masked_lines = masked.splitlines()
    # Reconstruct a view that keeps newlines if the last line had no trailing newline
    if text.endswith("\n") and (not masked_lines or masked_lines[-1] != ""):
        pass

    symbols: list[Symbol] = []
    # stack entries: kind, qualified name, brace_depth at which this decl opened, start, decl_line
    stack: list[tuple[str, str, int, int, int]] = []
    depth = 0

    # Join continued declaration lines: if a decl header has no `{` and no terminator
    # we still register it on the line with the keyword.

    i = 0
    while i < len(masked_lines):
        raw = lines[i] if i < len(lines) else ""
        mask = masked_lines[i]
        # count braces on this line
        opens = mask.count("{")
        closes = mask.count("}")

        kind = None
        name = None
        match = DECL_LINE.match(mask if mask.strip() else raw)
        # Match against the masked line first (keywords aren't in strings).
        # If the line was fully masked (comment), skip.
        if mask.strip():
            match = DECL_LINE.match(lines[i]) if False else DECL_LINE.match(mask)
            # DECL_LINE uses ^ and indent; masked line keeps indent and keywords.
            if match is None:
                match = DECL_LINE.match(raw) if not _line_is_blank_mask(mask) else None
            if match:
                kind = match.group("kind")
                name = _strip_name(match.group("name"))
                # Don't treat switch-cases as enum cases.
                if kind == "case":
                    parent_kind = stack[-1][0] if stack else ""
                    if parent_kind != "enum" or SWITCH_CASE.match(raw):
                        kind = None
                elif kind in {"func", "var", "let", "typealias", "init", "subscript", "case",
                              "enum", "struct", "class", "actor", "protocol", "extension"}:
                    parent_kind = stack[-1][0] if stack else ""
                    if parent_kind in {"func", "init", "var", "let", "subscript"}:
                        # Local declarations inside a body are not citation targets.
                        kind = None
                if kind == "init":
                    name = "init"
                if kind == "subscript":
                    name = "subscript"
                if kind == "extension":
                    # extension Foo: Protocol  or extension Foo
                    if not name:
                        rest = mask[match.end() :]
                        ext = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)", rest)
                        if ext:
                            name = ext.group(1)
                if kind in {"func", "var", "let", "enum", "struct", "class", "actor",
                            "protocol", "extension", "typealias"} and not name:
                    kind = None

        if kind and name:
            parent_qual = stack[-1][1] if stack else ""
            if kind == "extension":
                qual = name
            elif parent_qual:
                qual = f"{parent_qual}.{name}"
            else:
                qual = name
            start = _leading_doc_start(lines, i)
            # If this declaration has a body, end is set when its braces close.
            # If not (case, typealias, protocol requirement), end is this line
            # or the last continued line before a brace-less terminator.
            end_line = i + 1
            def _body_opens_later() -> bool:
                if opens > 0:
                    return True
                j = i
                paren = mask.count("(") - mask.count(")")
                while j + 1 < len(masked_lines) and paren > 0:
                    j += 1
                    paren += masked_lines[j].count("(") - masked_lines[j].count(")")
                    if "{" in masked_lines[j]:
                        return True
                return False

            if kind in {"enum", "struct", "class", "actor", "protocol", "extension",
                        "func", "var", "init", "subscript"} and (opens > 0 or _body_opens_later()):
                # Body is on this line or a later continuation line.
                stack.append((kind, qual, depth if opens > 0 else -1, start, i + 1))
            else:
                # No brace body: enum case, stored property, typealias, etc.
                j = i
                paren = mask.count("(") - mask.count(")")
                while j + 1 < len(masked_lines) and paren > 0:
                    j += 1
                    paren += masked_lines[j].count("(") - masked_lines[j].count(")")
                end_line = j + 1 if j > i else i + 1
                symbols.append(Symbol(name=qual, kind=kind, start=start, end=end_line, decl_line=i + 1))

        # Brace accounting, including closing decls.
        # If something is waiting for '{', the first '{' at this nesting binds it.
        if stack and stack[-1][2] == -1 and opens > 0:
            kind_s, qual, _, start, decl_line = stack[-1]
            stack[-1] = (kind_s, qual, depth, start, decl_line)
            # this '{' opens the pending decl; depth increment happens below

        new_depth = depth + opens - closes
        # Close any decls whose open_depth >= new_depth (after this line's closes)
        while stack and stack[-1][2] >= 0 and new_depth <= stack[-1][2]:
            kind_s, qual, open_depth, start, decl_line = stack.pop()
            symbols.append(Symbol(name=qual, kind=kind_s, start=start, end=i + 1, decl_line=decl_line))
        depth = new_depth
        if depth < 0:
            depth = 0
        i += 1

    # Unclosed decls run to EOF
    while stack:
        kind_s, qual, open_depth, start, decl_line = stack.pop()
        if open_depth == -1:
            # never saw a body; end on the decl line
            symbols.append(Symbol(name=qual, kind=kind_s, start=start, end=decl_line, decl_line=decl_line))
        else:
            symbols.append(Symbol(name=qual, kind=kind_s, start=start, end=len(lines), decl_line=decl_line))
    symbols.sort(key=lambda s: (s.start, s.decl_line, s.end, s.name))
    return symbols


def _line_is_blank_mask(mask: str) -> bool:
    return not mask.strip()


def symbols_by_name(symbols: list[Symbol]) -> dict[str, list[Symbol]]:
    found: dict[str, list[Symbol]] = {}
    for symbol in symbols:
        found.setdefault(symbol.name, []).append(symbol)
        # Also index the bare last component when unique enough to look up later
        short = symbol.name.rsplit(".", 1)[-1]
        if short != symbol.name:
            found.setdefault(short, []).append(symbol)
    return found


def find_symbol(symbols: list[Symbol], name: str) -> tuple[Symbol | None, str | None]:
    """Return (symbol, error). error is set when missing or ambiguous."""
    exact = [s for s in symbols if s.name == name]
    if len(exact) == 1:
        return exact[0], None
    if len(exact) > 1:
        # Overloads: treat as one logical symbol covering all if they share a kind
        # but for a span, they are ambiguous.
        return None, (
            f"symbol {name!r} is ambiguous "
            f"({len(exact)} declarations at lines "
            + ", ".join(str(s.decl_line) for s in exact)
            + ")"
        )
    # Allow a unique short-name match only when one declaration has that exact unqualified name
    short = [s for s in symbols if s.name == name or s.name.rsplit(".", 1)[-1] == name]
    # Prefer exact already handled. Unique fully-qualified ending.
    qualified = [s for s in symbols if s.name.rsplit(".", 1)[-1] == name]
    if len(qualified) == 1:
        return qualified[0], None
    if len(qualified) > 1:
        options = ", ".join(sorted({s.name for s in qualified}))
        return None, f"symbol {name!r} is ambiguous (did you mean {options}?)"
    return None, f"symbol {name!r} not found in file (gone or renamed)"


def unique_symbol_for_range(symbols: list[Symbol], start: int, end: int) -> Symbol | None:
    """A single symbol whose declaration uniquely owns this line range."""
    matches: list[Symbol] = []
    for symbol in symbols:
        if symbol.start == start and symbol.end == end:
            matches.append(symbol)
        elif symbol.decl_line == start and symbol.end == end:
            matches.append(symbol)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # Prefer the qualified / longer name at the same span (type over nothing)
        matches.sort(key=lambda s: (-len(s.name), s.name))
        # If two different symbols share the exact span, don't guess
        if len({s.name for s in matches}) > 1 and {s.kind for s in matches} != {"extension"}:
            return None
        return matches[0]
    return None


def lookup_in_source(text: str, symbol_name: str) -> tuple[Symbol | None, str | None]:
    return find_symbol(index_swift(text), symbol_name)


def best_symbol_upgrade(symbols: list[Symbol], start: int, end: int) -> Symbol | None:
    """Map a line range to one Swift symbol when the range clearly is that symbol.

    Exact span matches win. Otherwise the range must start at the symbol's
    start or declaration line, must not run more than two lines past the
    symbol, and must cover at least 80% of the declaration body. This will
    not turn a file-spanning cite that starts at a small type into that type.
    """
    exact = unique_symbol_for_range(symbols, start, end)
    if exact:
        return exact
    kinds = {"enum", "struct", "class", "actor", "protocol", "extension", "func", "var", "init", "case", "typealias"}
    types = [
        s
        for s in symbols
        if s.kind in kinds and (s.start == start or s.decl_line == start)
    ]
    candidates: list[Symbol] = []
    for symbol in types:
        if end > symbol.end + 2:
            continue
        typelen = symbol.end - symbol.decl_line + 1
        span = end - start + 1
        if typelen <= 0 or span < max(1, int(0.8 * typelen)):
            continue
        candidates.append(symbol)
    if len(candidates) == 1:
        return candidates[0]
    return None
