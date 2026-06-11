"""Synthetic puzzle generators matching the real train.csv formats exactly
(templates/<type>.txt are copied from real prompts), plus OOD 'neighbor'
variants. Every item is verified with the corresponding solver before being
emitted, so (prompt, answer) consistency is guaranteed.
"""
import os
import random
import string
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from solvers import bits as bits_solver  # noqa: E402
from solvers import cipher as cipher_solver  # noqa: E402
from solvers import equations as eq_solver  # noqa: E402
from solvers.numeral import to_roman  # noqa: E402

_TPL_DIR = os.path.join(os.path.dirname(__file__), "..", "templates")


def _tpl(name):
    with open(os.path.join(_TPL_DIR, f"{name}.txt"), encoding="utf-8") as f:
        return f.read()


def _render(name, examples, query):
    return _tpl(name).format(examples="\n".join(examples), query=query)


def _vocab():
    return cipher_solver._vocab()


# ---------------------------------------------------------------- bits
def gen_bits(rng, width=8, n_examples=8):
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
        if width != 8:
            prompt = prompt.replace("8-bit", f"{width}-bit")
        ans = bits_solver.solve(prompt)
        # keep only unambiguous puzzles: solver must agree with intended rule
        if ans is not None and int(ans, 2) == fn(qx):
            return prompt, ans
    return None


# ---------------------------------------------------------------- gravity
def gen_gravity(rng, n_examples=5, linear=False):
    g = round(rng.uniform(1.5, 30.0), 2)
    ts = [round(rng.uniform(1.0, 5.0), 2) for _ in range(n_examples + 1)]
    if len(set(ts)) != len(ts):
        ts = [round(t + i * 0.01, 2) for i, t in enumerate(ts)]
    law = (lambda t: g * t) if linear else (lambda t: 0.5 * g * t * t)
    examples = [f"For t = {t}s, distance = {law(t):.2f} m" for t in ts[:-1]]
    qt = ts[-1]
    prompt = _render("gravity", examples, f"{qt}")
    if linear:
        prompt = prompt.replace("d = 0.5*g*t^2", "d = g*t")
    return prompt, f"{law(qt):.2f}"


# ---------------------------------------------------------------- units
def gen_units(rng, n_examples=5, affine=False):
    r = round(rng.uniform(0.05, 5.0), 4)
    b = round(rng.uniform(-10, 10), 2) if affine else 0.0
    xs = [round(rng.uniform(5.0, 60.0), 2) for _ in range(n_examples + 1)]
    examples = [f"{x} m becomes {r * x + b:.2f}" for x in xs[:-1]]
    qx = xs[-1]
    prompt = _render("units", examples, f"{qx}")
    return prompt, f"{r * qx + b:.2f}"


# ---------------------------------------------------------------- cipher
def gen_cipher(rng, n_examples=5, caesar=False):
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

    vocab = _vocab()
    phrases = [" ".join(rng.sample(vocab, rng.randint(3, 5)))
               for _ in range(n_examples)]
    q = " ".join(rng.sample(vocab, rng.randint(3, 4)))
    examples = [f"{encrypt(p)} -> {p}" for p in phrases]
    prompt = _render("cipher", examples, encrypt(q))
    ans = cipher_solver.solve(prompt)
    if ans != q:
        return None  # ambiguous under vocab constraints; retry
    return prompt, q


# ---------------------------------------------------------------- numeral
def gen_numeral(rng, base=None, n_examples=5):
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
    else:  # in-distribution: standard roman numerals
        render = to_roman

    for _ in range(50):
        ns = rng.sample(range(1, 400), n_examples + 1)
        qn = ns[-1]
        if base and not set(render(qn)) <= set("".join(render(n) for n in ns[:-1])):
            continue
        examples = [f"{n} -> {render(n)}" for n in ns[:-1]]
        prompt = _render("numeral", examples, str(qn))
        return prompt, render(qn)
    return None


# ---------------------------------------------------------------- equations
_EQ_SYMBOL_POOL = "@#$%&!?^~;:<>(){}[]`'\"\\|/+-*0123456789"


def gen_equations(rng, n_examples=4, ops="+-*"):
    """Real encoding: substitute a random subset of canonical chars with
    distinct symbols, write each side REVERSED."""
    for _ in range(80):
        # build substitution: every canonical char maps somewhere (mostly id)
        canon = list("0123456789") + list(ops)
        n_sub = rng.randint(3, len(canon))
        subbed = rng.sample(canon, n_sub)
        pool = [s for s in _EQ_SYMBOL_POOL if s not in canon or s in subbed]
        rng.shuffle(pool)
        table = {c: c for c in canon}
        for c, s in zip(subbed, pool):
            table[c] = s
        if len(set(table.values())) != len(table):
            continue

        def enc(plain):
            return "".join(table[c] for c in plain)[::-1]

        lines, q = [], None
        ok = True
        for i in range(n_examples + 1):
            a = rng.randint(10, 99)
            b = rng.randint(10, 99)
            op = rng.choice(ops)
            val = eval(f"{a}{op}{b}")
            if i < n_examples:
                lines.append(f"{enc(f'{a}{op}{b}')} = {enc(str(val))}")
            else:
                q = (f"{a}{op}{b}", val)
        prompt = _render("equations", lines, enc(q[0]))
        gold = enc(str(q[1]))
        if eq_solver.solve(prompt) == gold:
            return prompt, gold
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
