"""Solver for 'equations': symbols substituted for digits/operators in
arithmetic expressions; infer the mapping from solved examples, then evaluate
the query. Answer: symbol string or integer.

This is the most format-dependent type. The implementation below assumes
example lines like  <symbolic expr> = <result>  and a final symbolic query.
It learns a char->char map from symbols to '0123456789+-*/' by constrained
backtracking, verifying every example equation. MUST be re-validated /
reverse-engineered against real train.csv prompts (see validate_solvers.py).
"""
import re

from .common import lines_of

_CANON = "0123456789+-*/"


def _safe_eval(expr: str):
    if not re.fullmatch(r"[\d+\-*/() ]+", expr):
        return None
    expr = re.sub(r"\b0+(\d)", r"\1", expr)  # leading zeros break eval()
    try:
        val = eval(expr, {"__builtins__": {}}, {})  # noqa: S307 - charset-restricted
    except (SyntaxError, ZeroDivisionError, ValueError):
        return None
    return val


def _extract(prompt):
    """Returns (examples, query): examples = [(symbolic_lhs, numeric_result)],
    query = symbolic expression with unknown result."""
    examples, query = [], None
    for ln in lines_of(prompt):
        m = re.match(r"(.+?)\s*=\s*(-?\d+)\s*$", ln)
        if m:
            examples.append((m.group(1).strip(), int(m.group(2))))
            continue
        m = re.match(r"(.+?)\s*=\s*\?\s*$", ln)
        if m:
            query = m.group(1).strip()
    if query is None:
        # fallback: last line that contains symbols but no '=' result
        for ln in reversed(lines_of(prompt)):
            if "=" not in ln and re.search(r"\S\s*[^\w\s]\s*\S", ln):
                query = ln.strip()
                break
    return examples, query


def _symbols(exprs):
    syms = set()
    for e in exprs:
        for c in e:
            if c not in " ()" and not c.isdigit() and c not in "+-*/":
                syms.add(c)
    return sorted(syms)


def _substitute(expr, table):
    return "".join(table.get(c, c) for c in expr)


def solve(prompt: str, max_symbols=8):
    examples, query = _extract(prompt)
    if not examples or query is None:
        return None
    syms = _symbols([e for e, _ in examples] + [query])
    if not syms:
        # expressions are already plain arithmetic
        val = _safe_eval(query)
        return None if val is None else str(int(val))
    if len(syms) > max_symbols:
        return None

    # backtracking: assign symbols one at a time; as soon as every symbol of an
    # example is assigned, verify it — prunes the vast majority of branches.
    sym_sets = [(set(c for c in lhs if c in syms), lhs, res) for lhs, res in examples]

    def backtrack(i, table, used):
        if i == len(syms):
            val = _safe_eval(_substitute(query, table))
            return None if val is None else str(int(val))
        s = syms[i]
        for c in _CANON:
            if c in used:
                continue
            table[s] = c
            assigned = set(table)
            if all(_check(es, lhs, res, table, assigned) for es, lhs, res in sym_sets):
                used.add(c)
                out = backtrack(i + 1, table, used)
                if out is not None:
                    return out
                used.discard(c)
            del table[s]
        return None

    def _check(es, lhs, res, table, assigned):
        if not es <= assigned:
            return True  # not fully concrete yet
        val = _safe_eval(_substitute(lhs, table))
        return val is not None and val == res

    return backtrack(0, {}, set())
