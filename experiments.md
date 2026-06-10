# Experiment Log — Nemotron Reasoning Challenge

Deadline: 2026-06-15 11:59PM UTC. D-1 (06-14): packaging + submission verified.

## 2026-06-10 — Pipeline scaffolding

- Built full pipeline (solvers, generators, eval harness, rejection sampling,
  SFT, packaging) — see README.md for run order.
- Solver↔generator round-trip self-test: **360/360** across 6 in-distribution
  types + 6 OOD variants (bits12/16, v=g·t, affine units, caesar, base-b
  numerals). Note: this validates internal consistency only — solvers still
  need validation against real `data/train.csv` (≥99% gate, step 1).
- Known provisional pieces (must be fixed against real data):
  - `templates/*.txt` are generic placeholders → replace via
    `scripts/extract_templates.py` output.
  - prompt parsers in `solvers/common.py` + per-type `_extract_*` try common
    notations (`->`, `=`, quoted strings) but were written blind.
  - `solvers/router.py` keyword type-detection needs tuning.
  - `scoring.py` numeric tolerance (rel/abs 1e-2) is a guess at the hidden
    metric — recalibrate if metric code is visible.
- Solver design notes:
  - bits: hypothesis search over composed shift/rotate/XOR/AND/OR/NOT/add +
    maj/ch of rotated copies; generator rejects rule-ambiguous puzzles.
  - gravity: consensus fit of g=2d/t² (robust to boilerplate numbers), both
    column orders, linear v=gt fallback; ties → original column order.
  - units: exact ratio → exact affine → consensus ratio (in that order, since
    affine data has near-equal ratios that consensus could wrongly accept).
  - cipher/numeral: substitution-table learning; generators guarantee query
    symbol coverage by the examples.
  - equations: char→(digit|op) backtracking with eval verification; leading
    zeros normalized; generator rejects ambiguous mappings.

## Baselines

| run | data | per-type acc | overall | notes |
|-----|------|--------------|---------|-------|
| base model | val_split | TBD | TBD | step 4 |

## SFT runs

| run | data size | lr | epochs | val acc | OOD val acc | notes |
|-----|-----------|----|--------|---------|-------------|-------|
