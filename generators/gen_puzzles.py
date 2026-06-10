"""Synthetic puzzle generators for all 6 train types + OOD 'neighbor' variants.

Every generated item is verified with the corresponding solver before being
emitted, guaranteeing (prompt, answer) consistency.

IMPORTANT — prompt templates: real train.csv templates are not yet in this
repo. Each type reads its template from templates/<type>.txt with slots
{examples} and {query}. The bundled templates are GENERIC PLACEHOLDERS; once
train.csv is available, run scripts/extract_templates.py (or copy a real
prompt manually) and overwrite the template files so synthetic prompts match
the train distribution exactly.
"""
import os
import random
import string
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from solvers import bits as bits_solver  # noqa: E402
from solvers.numeral import to_roman  # noqa: E402

_TPL_DIR = os.path.join(os.path.dirname(__file__), "..", "templates")

WORDS = ("the quick brown fox jumps over lazy dog pack my box with five dozen "
         "liquor jugs how vexingly daft zebras time flies like an arrow fruit "
         "bright vixens jump dozy fowl quack waltz bad nymph for quick jigs vex "
         "sphinx of black quartz judge my vow").split()


def _tpl(name):
    with open(os.path.join(_TPL_DIR, f"{name}.txt"), encoding="utf-8") as f:
        return f.read()


def _render(name, examples, query):
    return _tpl(name).format(examples="\n".join(examples), query=query)


# ---------------------------------------------------------------- bits
def gen_bits(rng, width=8, n_examples=4):
    """Pick a random rule from the solver's own candidate library."""
    m = (1 << width) - 1
    for _ in range(50):
        xs = rng.sample(range(m + 1), n_examples + 1)
        seed_pairs = [(xs[0], rng.randrange(m + 1))]
        cands = list(bits_solver.candidate_functions(width, seed_pairs))
        _, fn = rng.choice(cands)
        pairs = [(x, fn(x)) for x in xs[:-1]]
        qx = xs[-1]
        examples = [f"{x:0{width}b} -> {y:0{width}b}" for x, y in pairs]
        prompt = _render("bits", examples, f"{qx:0{width}b}")
        ans = bits_solver.solve(prompt)
        # keep only unambiguous puzzles: solver's rule and the intended rule
        # must agree on the query
        if ans is not None and int(ans, 2) == fn(qx):
            return prompt, ans
    return None


# ---------------------------------------------------------------- gravity
def gen_gravity(rng, n_examples=3, linear=False):
    g = round(rng.uniform(1.5, 30.0), 2)
    ts = rng.sample([round(0.5 * i, 1) for i in range(1, 21)], n_examples + 1)
    law = (lambda t: g * t) if linear else (lambda t: 0.5 * g * t * t)
    examples = [f"t = {t} s, d = {law(t):.2f} m" for t in ts[:-1]]
    qt = ts[-1]
    prompt = _render("gravity", examples, f"t = {qt} s")
    return prompt, f"{law(qt):.2f}"


# ---------------------------------------------------------------- units
def gen_units(rng, n_examples=3, affine=False):
    r = round(rng.uniform(0.05, 50.0), 3)
    b = round(rng.uniform(-20, 20), 2) if affine else 0.0
    xs = rng.sample(range(1, 500), n_examples + 1)
    examples = [f"{x} zarks = {r * x + b:.2f} blims" for x in xs[:-1]]
    qx = xs[-1]
    prompt = _render("units", examples, f"{qx} zarks")
    return prompt, f"{r * qx + b:.2f}"


# ---------------------------------------------------------------- cipher
def gen_cipher(rng, n_examples=3, caesar=False):
    letters = string.ascii_lowercase
    if caesar:
        k = rng.randrange(1, 26)
        enc = {c: letters[(i + k) % 26] for i, c in enumerate(letters)}
    else:
        perm = list(letters)
        rng.shuffle(perm)
        enc = dict(zip(letters, perm))

    def encrypt(s):
        return "".join(enc.get(c, c) for c in s)

    phrases = [" ".join(rng.sample(WORDS, rng.randint(2, 4)))
               for _ in range(n_examples)]
    # query must only use letters covered by the examples, or it is unsolvable
    seen = set("".join(phrases))
    covered = [w for w in WORDS if set(w) <= seen]
    if len(covered) < 3:
        return None
    q = " ".join(rng.sample(covered, 3))
    examples = [f'"{encrypt(p)}" -> "{p}"' for p in phrases]
    prompt = _render("cipher", examples, f'"{encrypt(q)}"')
    return prompt, q


# ---------------------------------------------------------------- numeral
def gen_numeral(rng, base=None, n_examples=4):
    if base:  # OOD: base-b digits with custom symbols
        digits = "0123456789abcdef"[:base]
        symbols = rng.sample(string.ascii_uppercase, base)
        table = dict(zip(digits, symbols))

        def render(n):
            s = ""
            while n:
                s = "0123456789abcdef"[n % base] + s
                n //= base
            return "".join(table[c] for c in (s or "0"))
    else:  # roman numerals with substituted symbols
        canon = "IVXLCDM"
        symbols = rng.sample([c for c in string.ascii_uppercase if c not in canon], 7)
        table = dict(zip(canon, symbols))

        def render(n):
            return "".join(table[c] for c in to_roman(n))

    for _ in range(50):
        ns = rng.sample(range(1, 2500), n_examples + 1)
        examples = [f"{n} -> {render(n)}" for n in ns[:-1]]
        qn = ns[-1]
        # query must only use symbols covered by the examples
        if not set(render(qn)) <= set("".join(render(n) for n in ns[:-1])):
            continue
        prompt = _render("numeral", examples, str(qn))
        return prompt, render(qn)
    return None


# ---------------------------------------------------------------- equations
def gen_equations(rng, n_symbols=5, n_examples=4):
    from solvers.equations import solve as eq_solve

    for _ in range(50):
        digits = rng.sample("123456789", n_symbols - 1)  # avoid leading zeros
        canon = digits + [rng.choice("+-*")]
        symbols = rng.sample("@#$%&!?^~;", n_symbols)
        table = dict(zip(canon, symbols))

        def make_expr():
            a = "".join(rng.choices(digits, k=rng.randint(1, 2)))
            b = "".join(rng.choices(digits, k=rng.randint(1, 2)))
            op = canon[-1]
            return f"{a}{op}{b}", eval(f"{int(a)}{op}{int(b)}")

        examples, used = [], set()
        for _ in range(n_examples):
            expr, val = make_expr()
            used.update(expr)
            examples.append(f"{''.join(table[c] for c in expr)} = {val}")
        qexpr, qval = make_expr()
        if not set(qexpr) <= used:  # query symbols must appear in examples
            continue
        prompt = _render("equations", examples,
                         f"{''.join(table[c] for c in qexpr)} = ?")
        # reject ambiguous puzzles (multiple mappings fitting the examples)
        if eq_solve(prompt) == str(qval):
            return prompt, str(qval)
    return None


GENERATORS = {
    "bits": lambda rng: gen_bits(rng),
    "gravity": lambda rng: gen_gravity(rng),
    "units": lambda rng: gen_units(rng),
    "cipher": lambda rng: gen_cipher(rng),
    "numeral": lambda rng: gen_numeral(rng),
    "equations": lambda rng: gen_equations(rng),
}

OOD_GENERATORS = {
    "bits12": lambda rng: gen_bits(rng, width=12),
    "bits16": lambda rng: gen_bits(rng, width=16),
    "velocity": lambda rng: gen_gravity(rng, linear=True),
    "affine_units": lambda rng: gen_units(rng, affine=True),
    "caesar": lambda rng: gen_cipher(rng, caesar=True),
    "base_numeral": lambda rng: gen_numeral(rng, base=rng.choice([3, 5, 7, 8])),
}


def generate(kind, n, seed=0):
    """Yield n verified (kind, prompt, answer) tuples."""
    rng = random.Random(seed)
    gen = GENERATORS.get(kind) or OOD_GENERATORS[kind]
    made = 0
    while made < n:
        item = gen(rng)
        if item is None:
            continue
        prompt, ans = item
        yield kind, prompt, ans
        made += 1
