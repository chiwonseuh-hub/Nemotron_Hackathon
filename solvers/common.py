"""Flexible parsing helpers for extracting example pairs / queries from prompts.

The real train.csv prompt formats are not yet visible in this repo, so these
parsers try several common notations. Once data is available, run
scripts/validate_solvers.py and tighten the patterns per type.
"""
import re

ARROW_RES = [
    re.compile(r"(.+?)\s*(?:->|=>|→|maps to|becomes|gives|:)\s*(.+)"),
]

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
BIN_RE = re.compile(r"\b[01]{4,32}\b")
QUOTED_RE = re.compile(r"[\"'“‘]([^\"'”’]+)[\"'”’]")


def lines_of(text):
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def extract_pair_lines(text):
    """Return list of (lhs, rhs) string pairs from lines containing an arrow-ish
    separator. Tries '->', '=>', unicode arrow, 'maps to', 'becomes', ':'."""
    pairs = []
    for ln in lines_of(text):
        for rx in ARROW_RES:
            m = rx.match(ln)
            if m:
                pairs.append((m.group(1).strip(), m.group(2).strip()))
                break
    return pairs


def extract_binary_pairs(text):
    """(input_bits, output_bits) pairs: lines with exactly two binary strings.
    Also returns the query: the last line with exactly one binary string."""
    pairs, query = [], None
    for ln in lines_of(text):
        bins = BIN_RE.findall(ln)
        if len(bins) == 2:
            pairs.append((bins[0], bins[1]))
        elif len(bins) == 1:
            query = bins[0]
    return pairs, query


def extract_number_pairs(text):
    """(x, y) float pairs from lines with exactly two numbers; query = the
    last line with exactly one number."""
    pairs, query = [], None
    for ln in lines_of(text):
        nums = NUM_RE.findall(ln)
        if len(nums) == 2:
            pairs.append((float(nums[0]), float(nums[1])))
        elif len(nums) == 1:
            query = float(nums[0])
    return pairs, query


def extract_quoted(text):
    return QUOTED_RE.findall(text)
