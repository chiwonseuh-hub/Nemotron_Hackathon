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

## 2026-06-11 — Solver validation against real train.csv

Data: 9,500 rows, 6 types (bits 1602 / gravity 1597 / units 1594 / cipher 1576
/ numeral 1576 / equations 1555). Router rewritten with exact first-line
signatures (each type has one fixed template).

Reverse-engineered formats:
- **gravity**: `For t = Xs, distance = Y m`; query embeds t in a 3-number line.
- **units**: `X m becomes Y` — pure ratio. 100% (1594/1594).
- **cipher**: cipher->plain examples, query may contain UNSEEN letters →
  closed 77-word vocabulary (cipher_vocab.txt) + crossword-style constraint
  search. 100% (1576/1576).
- **numeral**: standard Roman numerals. 100% (1576/1576).
- **gravity**: consensus fit. 100% (1597/1597).
- **equations**: encoding = REVERSE the string + injective char substitution
  (digits often identity; ops remapped, e.g. `/`->`-`). Example: plain
  `25-96 = -71` -> displayed `69/52 = 17/`. Op pool seen so far:
  +, -, *, abs-, and an odd `a+b+1` case; signed vs abs subtraction both
  occur. Solver: positional domains + backtracking + identity-first value
  ordering.
- **bits**: staged hypothesis search with per-stage majority voting:
  unary-chain/const -> binary over chain-2 -> 2-level xor-top trees ->
  maj/ch over chain-2 -> 3-leaf mixed-op trees (xor/and/or/add/sub over
  single/not operands, covers `or(xor(rotl1, not(shl3)), shr3)` forms).
  **Full train: 95.2%** (1525/1602, 53 none + 24 bad). Remainder are even
  deeper trees; probed 4-leaf shapes, maj/ch-inside, running-value
  pipelines, bit-permutations, affine mod 256 — no fit. Diminishing
  returns; rejection sampling uses gold answers anyway.

### equations deep-dive (status: ~70% of instances uncracked)
Confirmed mechanics (deep enumeration with gold-map recovery):
- both FORWARD and REVERSED encodings exist (per instance)
- digits keep identity when they appear as digits; substituted chars get
  weird symbols; canonical ops sometimes remapped ('/'->'-')
- op pool confirmed: +, -, *, abs(a-b) (02e871e4), a+b+1 (09302769:
  `21'16 = 38`, `90'91 -> 182`), signed minus with encoded sign (00d8b3db
  gold '17/' = rev('-71' encoded))
- signed-vs-abs subtraction is indistinguishable from the examples when
  results are positive; gold differs per instance -> irreducible ~coin-flip
  on a small slice.
Uncracked class (e.g. 00457d26, 02c15453, 0dce4039): exhaustively ruled out
within "single binary op + injective char map" paradigm: both directions,
op symbols unknown (any 1-3 chars), multi-op exprs with precedence,
digit-reversal/swap/abs flags, leading zeros, non-injective maps, bases
7-16, digit-wise mod-10 ops, a-b+k offsets. Each gives 0 fits or misses
gold. Parked — revisit only if time remains after SFT loop.

## Baselines

| run | data | per-type acc | overall | notes |
|-----|------|--------------|---------|-------|
| base model | val_split | TBD | TBD | step 4 |

## SFT runs

| run | data size | lr | epochs | val acc | OOD val acc | notes |
|-----|-----------|----|--------|---------|-------------|-------|
