"""Solver for 'cipher': character-level substitution cipher. Recover the
substitution table from example (plaintext, ciphertext) pairs, then decode the
query. Tries both directions (the prompt may list pairs as plain->cipher while
asking to decode, or vice versa). Answer: decoded text, exact format.
"""
from .common import extract_pair_lines, extract_quoted, lines_of


def _build_map(pairs):
    """Char map src->dst from aligned string pairs; None on conflict."""
    table = {}
    for src, dst in pairs:
        if len(src) != len(dst):
            return None
        for a, b in zip(src, dst):
            if a == " " and b == " ":
                continue
            if table.get(a, b) != b:
                return None
            table[a] = b
    return table


def _apply(table, text):
    out = []
    for c in text:
        if c == " ":
            out.append(" ")
        elif c in table:
            out.append(table[c])
        else:
            return None
    return "".join(out)


def _candidate_query(prompt, pair_strings):
    """The query is typically the last quoted/standalone string that is not
    part of an example pair."""
    used = set(pair_strings)
    cands = [q for q in extract_quoted(prompt) if q not in used]
    if cands:
        return cands[-1]
    # fallback: last non-empty line's tail after a colon
    for ln in reversed(lines_of(prompt)):
        if ":" in ln:
            tail = ln.rsplit(":", 1)[1].strip().strip("\"'")
            if tail and tail not in used:
                return tail
    return None


def solve(prompt: str):
    pairs = extract_pair_lines(prompt)
    # strip surrounding quotes from pair members
    pairs = [(a.strip("\"'“”‘’ "), b.strip("\"'“”‘’ ")) for a, b in pairs]
    pairs = [(a, b) for a, b in pairs if a and b and len(a) == len(b)]
    if not pairs:
        return None
    flat = [s for p in pairs for s in p]
    query = _candidate_query(prompt, flat)
    if query is None:
        return None
    # direction 1: examples are cipher->plain, query is cipher
    fwd = _build_map(pairs)
    if fwd is not None:
        res = _apply(fwd, query)
        if res is not None:
            return res
    # direction 2: examples are plain->cipher, query is cipher -> invert
    rev = _build_map([(b, a) for a, b in pairs])
    if rev is not None:
        res = _apply(rev, query)
        if res is not None:
            return res
    return None
