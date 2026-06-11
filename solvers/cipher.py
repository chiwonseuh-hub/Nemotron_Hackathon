"""Solver for 'cipher': letter-substitution cipher with examples
'ciphertext -> plaintext' and a query 'Now, decrypt the following text: ...'.

The query may contain cipher letters never seen in the examples, so a pure
table lookup is insufficient. The plaintexts come from a tiny closed
vocabulary (77 words, extracted from train.csv into cipher_vocab.txt), so the
remaining letters are resolved by crossword-style search: each query word must
decode to a vocabulary word consistent with the (injective) partial mapping.
"""
import os
import re

from .common import lines_of

_VOCAB_PATH = os.path.join(os.path.dirname(__file__), "cipher_vocab.txt")
_vocab_cache = None


def _vocab():
    global _vocab_cache
    if _vocab_cache is None:
        with open(_VOCAB_PATH, encoding="utf-8") as f:
            _vocab_cache = [w.strip() for w in f if w.strip()]
    return _vocab_cache


def _extract(prompt):
    pairs, query = [], None
    for ln in lines_of(prompt):
        if "->" in ln:
            a, b = ln.split("->", 1)
            a, b = a.strip(), b.strip()
            if a and b and len(a) == len(b):
                pairs.append((a, b))
        else:
            m = re.search(r"(?:decrypt|decode)[^:]*:\s*(.+)$", ln, re.IGNORECASE)
            if m:
                query = m.group(1).strip()
    return pairs, query


def _build_map(pairs):
    fwd, used = {}, {}
    for c_txt, p_txt in pairs:
        for c, p in zip(c_txt, p_txt):
            if c == " " or p == " ":
                if (c == " ") != (p == " "):
                    return None
                continue
            if fwd.get(c, p) != p or used.get(p, c) != c:
                return None  # not consistent / not injective
            fwd[c] = p
            used[p] = c
    return fwd


def _candidates(cword, fwd, used, vocab):
    out = []
    for w in vocab:
        if len(w) != len(cword):
            continue
        local = {}
        ok = True
        for c, p in zip(cword, w):
            known = fwd.get(c) or local.get(c)
            if known is not None:
                if known != p:
                    ok = False
                    break
            else:
                if p in used or p in local.values():
                    ok = False
                    break
                local[c] = p
        if ok:
            out.append((w, local))
    return out


def solve(prompt: str):
    pairs, query = _extract(prompt)
    if not pairs or query is None:
        return None
    fwd = _build_map(pairs)
    if fwd is None:
        return None
    used = set(fwd.values())
    vocab = _vocab()
    cwords = query.split()

    # most-constrained-first: fewest candidate words
    order = sorted(range(len(cwords)),
                   key=lambda i: len(_candidates(cwords[i], fwd, used, vocab)))

    solution = [None] * len(cwords)

    def dfs(k, fwd, used):
        if k == len(cwords):
            return True
        i = order[k]
        for w, local in _candidates(cwords[i], fwd, used, vocab):
            solution[i] = w
            fwd2 = dict(fwd)
            fwd2.update(local)
            if dfs(k + 1, fwd2, used | set(local.values())):
                return True
        solution[i] = None
        return False

    if dfs(0, fwd, used):
        return " ".join(solution)
    return None
