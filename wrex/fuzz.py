"""Randomized differential fuzzer (spec 12.3 kind 2, scaled up): generates
small programs across the representable families (and some unrepresentable
ones), then runs each under Seq, Par, auto, GPU (when available) and the
generated C (when supported) and requires identical outputs and traps."""
import random
from .compiler import compile_source, make_interpreter
from .interp import Config
from .values import Trap, fmt

def _data(rnd, n, lo=-9, hi=9): return [rnd.randint(lo, hi) for _ in range(n)]

def gen_case(rnd):
    kind = rnd.choice(["affine_fold", "affine_scan", "sum_fold", "clamp_scan", "tropical",
                       "match_fold", "degree2_fold", "finite_fold", "pair_fold", "user_family_fold"])
    n = rnd.randint(0, 12)
    data = _data(rnd, n)
    lit = "[" + ", ".join(str(v) for v in data) + "]"
    if kind == "affine_fold":
        a, b = rnd.randint(-3, 3), rnd.randint(-2, 2); init = rnd.randint(-5, 5)
        src = f"fn f(xs: [int]) -> int:\n    fold(xs, {init}, (s, x) => s * {a} + x * {b})\nfn main(args: [Str]) -> Out:\n    print(f({lit}))\n"
        return src, ("fold", data, init)
    if kind == "affine_scan":
        a, b = rnd.randint(-2, 2), rnd.randint(-2, 2)
        src = f"fn f(xs: [int]) -> [int]:\n    scan(xs, 0, (s, x) => s * {a} + x + {b})\nfn main(args: [Str]) -> Out:\n    print(f({lit}))\n"
        return src, None
    if kind == "sum_fold":
        src = f"fn f(xs: [int]) -> int:\n    fold(xs, 0, (s, x) => s + x)\nfn main(args: [Str]) -> Out:\n    print(f({lit}))\n"
        return src, ("fold", data)
    if kind == "clamp_scan":
        cap = rnd.randint(0, 12)
        src = f"fn f(xs: [int]) -> [int]:\n    scan(xs, 0, (s, d) => clamp(s + d, 0, {cap}))\nfn main(args: [Str]) -> Out:\n    print(f({lit}))\n"
        return src, None
    if kind == "tropical":
        src = ("fn f(xs: [int]) -> int:\n    let (_, best) = fold(xs, (0, 0), ((cur, best), x) =>\n"
               "        let cur2 = if x > 0 then cur + 1 else 0\n        (cur2, max(best, cur2)))\n    best\n"
               f"fn main(args: [Str]) -> Out:\n    print(f({lit}))\n")
        return src, None
    if kind == "match_fold":
        src = (f"fn f(xs: [int]) -> int:\n    fold(xs, 0, (s, x) =>\n        match x:\n            0 => s\n            _ => s + x)\n"
               f"fn main(args: [Str]) -> Out:\n    print(f({lit}))\n")
        return src, ("fold", data)
    if kind == "degree2_fold":
        # keep the (intentionally unrepresentable) quadratic iteration short:
        # its magnitude roughly squares each step and otherwise explodes
        data = [max(-2, min(2, v)) for v in data[:4]]
        lit = "[" + ", ".join(str(v) for v in data) + "]"
        src = f"fn f(xs: [int]) -> int:\n    fold(xs, 1, (s, r) => r * s * (10 - s))\nfn main(args: [Str]) -> Out:\n    print(f({lit}))\n"
        return src, None
    if kind == "finite_fold":
        src = ("type Light = Red | Green\nfn f(xs: [int]) -> Light:\n    fold(xs, Red, (s, x) => if x > 0 then Green else Red)\n"
               f"fn main(args: [Str]) -> Out:\n    print(f({lit}))\n")
        return src, None
    if kind == "user_family_fold":
        small = [max(-3, min(3, v)) for v in data[:8]]
        lit2 = "[" + ", ".join(str(v) for v in small) + "]"
        src = ("family growth over int identity 0: (a, b) => a + b + a * b\n"
               f"fn f(xs: [int]) -> int:\n    fold(xs, 0, (s, x) => growth(s, x))\nfn main(args: [Str]) -> Out:\n    print(f({lit2}))\n")
        return src, ("fold", small, 0)
    src = (f"fn f(xs: [int]) -> (int, int):\n    fold(xs, (0, 1), ((a, b), x) => (a + x, b * {rnd.randint(1,3)}))\n"
           f"fn main(args: [Str]) -> Out:\n    print(f({lit}))\n")
    return src, None

def _run(src, plan, device="cpu"):
    prog, ck, an = compile_source(src)
    interp = make_interpreter(prog, an, Config(plan=plan, workers=4, device=device))
    from .interp import run_main
    try:
        return ("ok", fmt(run_main(interp, [])))
    except Trap as t:
        return ("trap", t.kind)

def _run_c(src, cinfo):
    import subprocess, os, tempfile
    from .codegen import generate_c, CodegenError
    data = cinfo[1]; s0 = cinfo[2] if len(cinfo) > 2 else 0
    prog, ck, an = compile_source(src)
    try:
        csrc = generate_c(prog, an, "f")
    except CodegenError:
        return None
    stdin = f"{len(data)} " + " ".join(str(v) for v in data) + f" {s0}"
    with tempfile.TemporaryDirectory() as d:
        cpath, binpath = os.path.join(d, "o.c"), os.path.join(d, "o")
        open(cpath, "w").write(csrc)
        r = subprocess.run(["cc", "-O2", "-o", binpath, cpath], capture_output=True, text=True)
        if r.returncode != 0: return ("c_compile_fail", r.stderr[:200])
        r = subprocess.run([binpath, "4"], input=stdin, capture_output=True, text=True)
        return ("ok", r.stdout.strip()) if r.returncode == 0 else ("trap", "c_exit")

def fuzz(iters=50, seed=0, gpu=False, include_c=True, log=None):
    rnd = random.Random(seed)
    failures = []
    for i in range(iters):
        src, cinfo = gen_case(rnd)
        expected = _run(src, "seq")
        got = {"par": _run(src, "par"), "auto": _run(src, "auto")}
        if gpu: got["gpu"] = _run(src, "par", device="metal")
        for plan, val in got.items():
            if val != expected:
                failures.append((i, plan, expected, val, src))
                if log: log(f"iter {i} {plan}: expected {expected} got {val}")
        if include_c and cinfo is not None:
            cval = _run_c(src, cinfo)
            if cval is not None and cval != expected:
                failures.append((i, "c", expected, cval, src))
                if log: log(f"iter {i} c: expected {expected} got {cval}")
    return failures

def main(argv=None):
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    iters, seed, gpu = 50, 0, "--gpu" in argv
    for a in argv:
        if a.startswith("--iters="): iters = int(a.split("=", 1)[1])
        if a.startswith("--seed="): seed = int(a.split("=", 1)[1])
    failures = fuzz(iters=iters, seed=seed, gpu=gpu, log=lambda m: print(m))
    print(f"fuzz: {iters} cases, {len(failures)} failures")
    return 1 if failures else 0

if __name__ == "__main__":
    raise SystemExit(main())
