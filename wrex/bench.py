"""Appendix B benchmark suite: times each benchmark under Seq, Par, and
(when tinygrad is present) GPU plans. Baselines in the spec are hand-tuned
native implementations; here the Seq column is the reference interpreter,
which is the honest in-repo baseline for relative plan speedups."""
import os, random, time
from .compiler import compile_source, make_interpreter
from .interp import Config
from .values import ADT

BENCH_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks")

def _load(name):
    with open(os.path.join(BENCH_DIR, name)) as f: return compile_source(f.read())

def _time(interp, fn, args, reps=2):
    best = None
    for _ in range(reps):
        t0 = time.perf_counter(); r = interp.call_fn(fn, args); dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    return best, r

def run_bench(n=2000, gpu=False):
    rnd = random.Random(42)
    i64s = [rnd.randint(-9, 9) for _ in range(n)]
    rows = []
    def case(label, fname, fn, args):
        prog, ck, an = _load(fname)
        fam = next((r["cert"].describe() for r in an.records if "cert" in r), "none")
        out = {}
        results = {}
        for plan, dev in (("seq", "cpu"), ("par", "cpu")) + ((("par", "metal"),) if gpu else ()):
            interp = make_interpreter(prog, an, Config(plan=plan, workers=8, device=dev))
            dt, r = _time(interp, fn, args)
            out[dev if dev != "cpu" else plan] = dt; results[plan + dev] = r
        vals = list(results.values())
        rows.append((label, fam, n, out, all(v == vals[0] for v in vals)))
    case("Prefix sum", "prefix_sum.wrx", "prefix_sum", [list(i64s)])
    case("Linear recurrence", "linear_recurrence.wrx", "balances", [[(1, x) for x in i64s]])
    case("Saturating counters", "saturating_counters.wrx", "stock_levels", [list(i64s), 50])
    case("Tokenizer with counters", "tokenizer.wrx", "word_count", [[rnd.choice(b"ab ") for _ in range(n)]])
    case("Viterbi decoding (tropical)", "viterbi.wrx", "viterbi", [list(i64s)])
    # expression trees: balanced and list-shaped
    prog, ck, an = _load("expr_tree.wrx")
    def balanced(d):
        if d == 0: return ADT("Num", (1,), "Expr")
        return ADT("Add", (balanced(d - 1), balanced(d - 1)), "Expr")
    def chain(m):
        t = ADT("Num", (1,), "Expr")
        for _ in range(m): t = ADT("Add", (t, ADT("Num", (1,), "Expr")), "Expr")
        return t
    for label, tree in (("Expr tree (balanced)", balanced(10)), ("Expr tree (list-shaped)", chain(1000))):
        out = {}; results = {}
        for plan in ("seq", "par", "auto"):
            interp = make_interpreter(prog, an, Config(plan=plan, workers=8))
            dt, r = _time(interp, "eval", [tree])
            out[plan] = dt; results[plan] = r
        rows.append((label, "contexts F1", getattr(tree, 'size', 0), out, len(set(results.values())) == 1))
    return rows

def main(argv=None):
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    n = 2000; gpu = "--gpu" in argv
    for a in argv:
        if a.startswith("--n="): n = int(a.split("=", 1)[1])
    rows = run_bench(n=n, gpu=gpu)
    print(f"| benchmark | family | n | " + " | ".join(["seq (s)", "par (s)"] + (["gpu (s)"] if gpu else [])) + " | plans agree |")
    print("|---|---|---|---|" + "---|" * (3 if gpu else 2))
    for label, fam, nn, out, agree in rows:
        cells = [f"{out.get(k, float('nan')):.4f}" for k in (("seq", "par", "metal") if gpu else ("seq", "par"))]
        print(f"| {label} | {fam} | {nn} | " + " | ".join(cells) + f" | {'yes' if agree else 'NO'} |")
    return 0 if all(r[4] for r in rows) else 1

if __name__ == "__main__":
    raise SystemExit(main())
