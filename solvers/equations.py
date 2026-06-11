"""Solver for 'equations' (reverse-engineered from train.csv):

The plain equation (e.g. "44-43 = 1") is encoded by (a) REVERSING each side's
string and (b) substituting characters injectively (digits usually map to
themselves; operators map to symbols like / | \\ ` ! ...). Example:
  plain  25-96 = -71
  encode reverse+subst -> "69/52 = 17/"   ('/' encodes '-')

So: reverse the encoded strings, backtrack over symbol -> canonical
assignments (digit 0-9 or one of the ops), verify every example equation,
then evaluate the query and re-encode its result (reversed, sign included).

Value ordering prefers the identity for digit symbols and common ops first,
which empirically matches the generator's choices when several assignments
fit. "+1" (a+b+1) is included because train instances require it.
"""
import re

from .common import lines_of

OPS = {
    "-": lambda a, b: a - b,
    "+": lambda a, b: a + b,
    "*": lambda a, b: a * b,
    "abs-": lambda a, b: abs(a - b),
    "//": lambda a, b: a // b if b else None,
    "%": lambda a, b: a % b if b else None,
    "+1": lambda a, b: a + b + 1,
}
_OP_ORDER = ["-", "+", "*", "abs-", "//", "%", "+1"]
_DIGITS = "0123456789"


def _extract(prompt):
    examples, query = [], None
    for ln in lines_of(prompt):
        if ln.lower().startswith("now"):
            m = re.search(r":\s*(.+?)\s*$", ln)
            if m:
                query = m.group(1)
            continue
        if ln.endswith(":"):
            continue
        parts = re.split(r"\s=\s", ln)
        if len(parts) == 2:
            examples.append((parts[0].strip()[::-1], parts[1].strip()[::-1]))
    return examples, (query[::-1] if query else None)


def _decode_num(s, table):
    """Decode a reversed-side number: in reversed orientation the minus sign
    sits at the END."""
    neg = False
    if s and table.get(s[-1]) == "-":
        neg, s = True, s[:-1]
    if not s:
        return None
    digs = [table.get(c) for c in s]
    if any(d is None or d not in _DIGITS for d in digs):
        return None
    return -int("".join(digs)) if neg else int("".join(digs))


def _eval_lhs(s, table):
    ops_pos = [i for i, c in enumerate(s) if table.get(c) in OPS]
    if len(ops_pos) != 1:
        return None
    i = ops_pos[0]
    if i == 0 or i == len(s) - 1:
        return None
    a = _decode_num(s[:i], table)
    b = _decode_num(s[i + 1:], table)
    if a is None or b is None:
        return None
    try:
        return OPS[table[s[i]]](a, b)
    except (ZeroDivisionError, ValueError):
        return None


def solve(prompt: str, node_budget=3_000_000, max_solutions=4):
    examples, query = _extract(prompt)
    if not examples or query is None:
        return None
    syms = sorted({c for lhs, rhs in examples for c in lhs + rhs} | set(query))

    # order symbols so the shortest examples complete (and prune) first
    order, seen = [], set()
    for lhs, rhs in sorted(examples, key=lambda e: len(e[0]) + len(e[1])):
        for c in lhs + rhs:
            if c not in seen:
                seen.add(c)
                order.append(c)
    for c in syms:
        if c not in seen:
            seen.add(c)
            order.append(c)

    ex_syms = [(frozenset(lhs + rhs), lhs, rhs) for lhs, rhs in examples]

    # positional domain restriction: a symbol that ever appears where only a
    # digit is legal can't be an operator. (Reversed orientation: a number's
    # minus sign is at the END of the rhs string; expression edges are digits.)
    digit_only = set()
    for lhs, rhs in examples + [(query, "")]:
        if lhs:
            digit_only.add(lhs[0])
            digit_only.add(lhs[-1])
        digit_only.update(rhs[:-1])

    def values_for(c):
        """Candidate canonical values, identity / common-ops first."""
        vals = []
        if c in _DIGITS:
            vals.append(c)  # identity bias
        vals.extend(d for d in _DIGITS if d != c)
        if c not in digit_only:
            vals.extend(_OP_ORDER)
        return vals

    def consistent(table, assigned):
        for es, lhs, rhs in ex_syms:
            if not es <= assigned:
                continue
            lv = _eval_lhs(lhs, table)
            rv = _decode_num(rhs, table)
            if lv is None or rv is None or lv != rv:
                return False
        return True

    budget = [node_budget]
    solutions = []

    def backtrack(i, table, used, assigned):
        if budget[0] <= 0 or len(solutions) >= max_solutions:
            return
        if i == len(order):
            val = _eval_lhs(query, table)
            if val is None:
                return
            inv = {v: k for k, v in table.items()}
            s = str(val)
            out = []
            for c in s:
                enc = inv.get(c) if c != "-" else inv.get("-")
                if enc is None:
                    return
                out.append(enc)
            solutions.append("".join(out)[::-1])
            return
        c = order[i]
        for v in values_for(c):
            if v in used:
                continue
            budget[0] -= 1
            if budget[0] <= 0:
                return
            table[c] = v
            assigned.add(c)
            if consistent(table, assigned):
                used.add(v)
                backtrack(i + 1, table, used, assigned)
                used.discard(v)
            assigned.discard(c)
            del table[c]

    backtrack(0, {}, set(), set())
    if not solutions:
        return None
    return solutions[0]  # preference-ordered search: first hit is best guess
