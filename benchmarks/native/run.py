"""Times the Appendix B loops in Wrex (interpreter Seq and Par, generated C)
and in hand-written C, Rust, Go, Java and Python, on identical inputs.

Every time is the computation alone, with the input already in memory, best
of repeated runs. Each language's result is checked against the Wrex
reference interpreter before its time is reported.

Usage: python3 benchmarks/native/run.py [--n=50000]   (writes results.json)"""
import json, os, random, re, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from wrex.bench import _load, _time, BENCH_DIR
from wrex.codegen import generate_c
from wrex.interp import Config
from wrex.compiler import make_interpreter

BENCHES = [
    ("prefix_sum", "prefix_sum.wrx", "prefix_sum"),
    ("linear_recurrence", "linear_recurrence.wrx", "balances"),
    ("saturating_counters", "saturating_counters.wrx", "stock_levels"),
    ("tokenizer", "tokenizer.wrx", "word_count"),
    ("viterbi", "viterbi.wrx", "viterbi"),
]


def checksum(r):
    if isinstance(r, list): return f"{sum(r)}:{r[-1]}"
    if isinstance(r, tuple): return f"{r[0]}:{r[1]}"
    return str(r)


def sh(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0: raise SystemExit(f"{' '.join(cmd)} failed:\n{r.stderr}")
    return r.stdout


def parse(out):
    return {name: (int(ns) / 1e9, chk) for name, ns, chk in (l.split() for l in out.strip().splitlines())}


def wrex_interp(xs, text):
    args = {"prefix_sum": [list(xs)], "linear_recurrence": [[(1, x) for x in xs]],
            "saturating_counters": [list(xs), 50], "tokenizer": [list(text)], "viterbi": [list(xs)]}
    seq, par, ref = {}, {}, {}
    for key, fname, fn in BENCHES:
        prog, _, an = _load(fname)
        for plan, dst in (("seq", seq), ("par", par)):
            dt, r = _time(make_interpreter(prog, an, Config(plan=plan, workers=8)), fn, args[key], reps=3)
            dst[key] = (dt, checksum(r))
        ref[key] = seq[key][1]
    return seq, par, ref


def generated_c(xs, text, d, runs=3, workers=8, cflags=()):
    """Times the generated C's compute step through its WREX_BENCH hook, which
    reports the fastest of 200 in-process runs, the same as the hand-written loops."""
    n = len(xs)
    stdin = {"prefix_sum": f"{n} {' '.join(map(str, xs))} 0",
             "linear_recurrence": f"{n} " + " ".join(f"1 {x}" for x in xs) + " 0",
             "saturating_counters": f"50 {n} {' '.join(map(str, xs))} 0",
             "tokenizer": f"{n} {' '.join(map(str, text))} 1 0",
             "viterbi": f"{n} {' '.join(map(str, xs))} 0 0"}
    out = {}
    for key, fname, fn in BENCHES:
        prog, _, an = _load(fname)
        src = generate_c(prog, an, fn)
        c, b = os.path.join(d, key + ".c"), os.path.join(d, key)
        open(c, "w").write(src)
        sh(["cc", "-O2", "-DWREX_BENCH=200", *cflags, "-o", b, c])
        best, res = None, None
        for _ in range(runs):
            r = subprocess.run([b, str(workers)], input=stdin[key], capture_output=True, text=True)
            ns = int(r.stderr.strip()); best = ns if best is None else min(best, ns)
            res = list(map(int, r.stdout.split()))
        # Scans print the output array, the tokenizer its (mode, count) state.
        out[key] = (best / 1e9, checksum({"tokenizer": res[-1], "viterbi": tuple(res)}.get(key, res)))
    return out


def main():
    n = next((int(a.split("=")[1]) for a in sys.argv[1:] if a.startswith("--n=")), 50000)
    rnd = random.Random(42)  # same draws as wrex bench
    xs = [rnd.randint(-9, 9) for _ in range(n)]
    text = [rnd.choice(b"ab ") for _ in range(n)]

    results = {}
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "data.txt"), "w") as f:
            f.write(f"{n}\n{' '.join(map(str, xs))}\n{' '.join(map(str, text))}\n")
        print("Wrex interpreter ...", file=sys.stderr)
        results["Wrex Seq"], results["Wrex Par"], ref = wrex_interp(xs, text)
        print("Wrex generated C ...", file=sys.stderr)
        results["Wrex generated C"] = generated_c(xs, text, d)

        java = shutil.which("javac") and os.environ.get("JAVA", "java")
        langs = [
            ("C", ["cc", "-O2", "-o", "bench_c", os.path.join(HERE, "bench.c")], ["./bench_c"]),
            ("Rust", ["rustc", "-O", "-o", "bench_rs", os.path.join(HERE, "bench.rs")], ["./bench_rs"]),
            ("Go", ["go", "build", "-o", "bench_go", os.path.join(HERE, "bench.go")], ["./bench_go"]),
            ("Java", ["javac", "-d", d, os.path.join(HERE, "Bench.java")], [java or "java", "-cp", d, "Bench"]),
            ("Python", None, [sys.executable, os.path.join(HERE, "bench.py")]),
        ]
        for lang, build, run in langs:
            print(f"{lang} ...", file=sys.stderr)
            if build: sh(build, cwd=d)
            results[lang] = parse(sh(run, cwd=d))

    ok = True
    for impl, rows in results.items():
        for key, (_, chk) in rows.items():
            if chk != ref[key]:
                ok = False; print(f"MISMATCH {impl} {key}: {chk} != {ref[key]}", file=sys.stderr)
    keys = [k for k, _, _ in BENCHES]
    print("| impl | " + " | ".join(keys) + " |"); print("|---" * (len(keys) + 1) + "|")
    for impl, rows in results.items():
        print(f"| {impl} | " + " | ".join(f"{rows[k][0]*1e3:.3f} ms" for k in keys) + " |")
    json.dump({"n": n, "agree": ok, "seconds": {i: {k: r[k][0] for k in keys} for i, r in results.items()}},
              open(os.path.join(HERE, "results.json"), "w"), indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
