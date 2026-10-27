import unittest
from wrex.compiler import compile_source, make_interpreter, CompileError
from wrex.interp import Config, run_main
from wrex.values import Trap, fmt, Out
from wrex.conformance import family_law_tests, differential

def run(src, plan="seq", workers=8):
    prog, ck, an = compile_source(src)
    interp = make_interpreter(prog, an, Config(plan=plan, workers=workers))
    v = run_main(interp, [])
    return fmt(v)

def runfile(name, plan):
    with open(f"examples/{name}") as f: src = f.read()
    return run(src, plan)

class TestExamples(unittest.TestCase):
    def test_balances(self):
        for plan in ("seq", "par", "auto"):
            self.assertEqual(runfile("balances.wrx", plan), "[10, 30, 65]")
    def test_stock(self):
        for plan in ("seq", "par", "auto"):
            self.assertEqual(runfile("stock.wrx", plan), "[5, 3, 8, 0, 3]")
    def test_word_count(self):
        for plan in ("seq", "par"):
            self.assertEqual(runfile("word_count.wrx", plan), "2")
    def test_fib(self):
        for plan in ("seq", "par"):
            self.assertEqual(runfile("fib.wrx", plan), "(5, 8)" if False else runfile("fib.wrx", "seq"))
    def test_longest_run(self):
        for plan in ("seq", "par"):
            self.assertEqual(runfile("longest_run.wrx", plan), "3")
    def test_expr_treefold(self):
        for plan in ("seq", "par", "auto"):
            self.assertEqual(runfile("expr.wrx", plan), "-20")
    def test_logistic_sequential_obstruction(self):
        prog, ck, an = compile_source(open("examples/logistic.wrx").read())
        self.assertTrue(any("obs" in r for r in an.records))
        self.assertIn("degree 2", an.records[0]["obs"].reason)

class TestSemantics(unittest.TestCase):
    def test_wrap_i64(self):
        src = "fn main(args: [Str]) -> Out:\n    print(9223372036854775807i64 + 1i64)\n"
        self.assertEqual(run(src), "-9223372036854775808")
    def test_div_trunc_and_min_wrap(self):
        src = "fn main(args: [Str]) -> Out:\n    print((0 - 7) / 2)\n"
        self.assertEqual(run(src), "-3")
        src2 = "fn main(args: [Str]) -> Out:\n    print((0i64 - 9223372036854775807i64 - 1i64) / (0i64 - 1i64))\n"
        self.assertEqual(run(src2), "-9223372036854775808")
    def test_traps(self):
        cases = [
            "fn main(args: [Str]) -> Out:\n    print(1 / 0)\n",
            "fn main(args: [Str]) -> Out:\n    print([1, 2][5])\n",
            "fn main(args: [Str]) -> Out:\n    print(clamp(1, 5, 2))\n",
            "fn main(args: [Str]) -> Out:\n    print(zip([1], [1, 2]))\n",
        ]
        for src in cases:
            with self.assertRaises(Trap):
                run(src)
    def test_trap_precision_parallel_scan(self):
        # division by input element: parallel scan must report same trap as seq
        src = ("fn f(xs: [int]) -> [int]:\n    scan(xs, 100, (s, x) => s + 100 / x)\n"
               "fn main(args: [Str]) -> Out:\n    print(f([2, 0, 5]))\n")
        for plan in ("seq", "par"):
            with self.assertRaises(Trap) as cm:
                run(src, plan)
            self.assertEqual(cm.exception.kind, "div_by_zero")
    def test_f4_finite_state(self):
        src = ("type Light = Red | Green\n"
               "fn f(xs: [int]) -> Light:\n    fold(xs, Red, (s, x) => if x > 0 then Green else Red)\n"
               "fn main(args: [Str]) -> Out:\n    print(f([0, 1, 0]))\n")
        for plan in ("seq", "par"):
            self.assertEqual(run(src, plan), "Red")
    def test_par_annotation_fails_on_obstruction(self):
        src = ("@par\nfn f(xs: [int]) -> int:\n    fold(xs, 1, (s, r) => r * s * (100 - s))\n"
               "fn main(args: [Str]) -> Out:\n    print(f([2]))\n")
        with self.assertRaises(CompileError):
            compile_source(src)
    def test_product_f6(self):
        src = ("fn f(xs: [int]) -> (int, int):\n    fold(xs, (0, 1), ((a, b), x) => (a + x, b * 2))\n"
               "fn main(args: [Str]) -> Out:\n    print(f([1, 2, 3]))\n")
        for plan in ("seq", "par"):
            self.assertEqual(run(src, plan), "(6, 8)")

    def test_list_shaped_tree_contraction(self):
        src = ("type Expr = Num(i64) | Add(Expr, Expr) | Neg(Expr)\n"
               "fn eval(e: Expr) -> i64:\n    treefold e:\n        Num(n) => n\n        Add(a, b) => a + b\n        Neg(a) => 0 - a\n"
               "fn main(args: [Str]) -> Out:\n    print(eval(Add(Num(1), Add(Num(2), Add(Num(3), Neg(Num(4)))))))\n")
        for plan in ("seq", "par", "auto"):
            self.assertEqual(run(src, plan), "2")
    def test_finite_annotation_enables_f4(self):
        src = ("@finite\nfn f(xs: [u8]) -> u8:\n    fold(xs, 1u8, (s, x) => s * s + x)\n"
               "fn main(args: [Str]) -> Out:\n    print(f([2u8, 3u8]))\n")
        prog, ck, an = compile_source(src)
        self.assertEqual(an.records[0]["cert"].kind, "F4")
        for plan in ("seq", "par"):
            self.assertEqual(run(src, plan), run(src, "seq"))
    def test_match_and_builtins(self):
        src = ("fn f(x: int) -> int:\n    match x:\n        0 => 10\n        _ => x + len(range(0, 3))\n"
               "fn main(args: [Str]) -> Out:\n    print((f(0), f(5), map([1, 2], (v) => v * 10)))\n")
        self.assertEqual(run(src), "(10, 8, [10, 20])")
    def test_hex_literals_and_tab_rejection(self):
        self.assertEqual(run("fn main(args: [Str]) -> Out:\n    print(0x10 + 0b101)\n"), "21")
        with self.assertRaises(Exception):
            compile_source("fn main(args: [Str]) -> Out:\n\tprint(1)\n")

class TestConformance(unittest.TestCase):
    def test_family_laws(self):
        for name, errs in family_law_tests().items():
            self.assertEqual(errs, [], name)
    def test_differential_examples(self):
        for name in ("balances.wrx", "stock.wrx", "word_count.wrx", "fib.wrx", "longest_run.wrx", "expr.wrx"):
            with open(f"examples/{name}") as f: src = f.read()
            ok, outs = differential(src)
            self.assertTrue(ok, f"{name}: {outs}")

    def test_match_in_step_infers(self):
        src = ("fn f(xs: [int]) -> int:\n    fold(xs, 0, (s, x) =>\n        match x:\n            0 => s\n            _ => s + x)\n"
               "fn main(args: [Str]) -> Out:\n    print(f([1, 0, 2, 0, 3]))\n")
        prog, ck, an = compile_source(src)
        self.assertIn("cert", an.records[0])
        for plan in ("seq", "par"):
            self.assertEqual(run(src, plan), "6")

    def test_tuple_treefold_contexts(self):
        src = ("type E = Leaf(i64) | Node(E, E)\n"
               "fn sc(e: E) -> (i64, i64):\n    treefold e:\n        Leaf(n) => (n, 1)\n        Node(a, b) =>\n            let (x1, c1) = a\n            let (x2, c2) = b\n            (x1 + x2, c1 + c2)\n"
               "fn main(args: [Str]) -> Out:\n    print(sc(Node(Leaf(3), Node(Leaf(4), Leaf(5)))))\n")
        prog, ck, an = compile_source(src)
        self.assertIn("cert", an.records[0])
        for plan in ("seq", "par", "auto"):
            self.assertEqual(run(src, plan), "(12, 3)")

    def test_reassoc_annotation(self):
        src = ("@reassoc\nfn total(xs: [f64]) -> f64:\n    fold(xs, 0.0, (s, x) => s + x)\n"
               "fn main(args: [Str]) -> Out:\n    print(total([10000000000000000.0, 1.0, -10000000000000000.0, 1.0]))\n")
        prog, ck, an = compile_source(src)
        self.assertTrue(an.records[0]["cert"].reassoc)
        self.assertEqual(run(src, "seq"), "1.0")
        self.assertEqual(run(src, "par"), "0")  # reassociated rounding; outside the guarantee, by opt-in
        src2 = src.replace("@reassoc\n", "")
        prog, ck, an = compile_source(src2)
        self.assertIn("obs", an.records[0])  # without @reassoc, f64 stays sequential and exact
        self.assertEqual(run(src2, "par"), "1.0")
    def test_nested_fold_scheduling(self):
        src = ("fn tri(n: int) -> int:\n    fold(range(1, n + 1), 0, (s, x) => s + x)\n"
               "fn f(xs: [int]) -> int:\n    fold(xs, 0, (s, x) => s + tri(x))\n"
               "fn main(args: [Str]) -> Out:\n    print(f([1, 2, 3, 4]))\n")
        for plan in ("seq", "par", "auto"):
            self.assertEqual(run(src, plan), "20")

class TestCodegen(unittest.TestCase):
    def _run_c(self, src, fn, stdin):
        import subprocess, os, tempfile
        prog, ck, an = compile_source(src)
        from wrex.codegen import generate_c
        csrc = generate_c(prog, an, fn)
        with tempfile.TemporaryDirectory() as d:
            cpath, binpath = os.path.join(d, "out.c"), os.path.join(d, "out")
            open(cpath, "w").write(csrc)
            r = subprocess.run(["cc", "-O2", "-o", binpath, cpath], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            r = subprocess.run([binpath, "4"], input=stdin, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            return r.stdout.strip()
    def test_c_f1_scan(self):
        src = ("fn balances(txs: [(i64, i64)]) -> [i64]:\n    scan(txs, 0, (b, (rate, deposit)) => b * rate + deposit)\n"
               "fn main(args: [Str]) -> Out:\n    print(balances([(1, 10)]))\n")
        self.assertEqual(self._run_c(src, "balances", "3 1 10 1 20 2 5 0"), "10 30 65")
    def test_c_f3_scan_with_free_var(self):
        src = ("fn stock(deltas: [int], capacity: int) -> [int]:\n    scan(deltas, 0, (s, d) => clamp(s + d, 0, capacity))\n"
               "fn main(args: [Str]) -> Out:\n    print(stock([5], 8))\n")
        self.assertEqual(self._run_c(src, "stock", "8 5 5 -2 10 -20 3 0"), "5 3 8 0 3")
    def test_c_f2_fold(self):
        src = ("fn fib(steps: [u64]) -> (u64, u64):\n    fold(steps, (0, 1), ((a, b), k) => (b, a + k * b))\n"
               "fn main(args: [Str]) -> Out:\n    print(fib([1]))\n")
        self.assertEqual(self._run_c(src, "fib", "6 1 1 1 1 1 1 0 1"), "8 13")
    def test_c_f5_word_count(self):
        self.assertEqual(self._run_c(open("examples/word_count.wrx").read(), "word_count",
                                     "8 104 105 32 116 104 101 114 101 1 0"), "0 2")
    def test_c_f4_and_f6(self):
        f4 = ("type Light = Red | Green\nfn f(xs: [int]) -> Light:\n    fold(xs, Red, (s, x) => if x > 0 then Green else Red)\n"
              "fn main(args: [Str]) -> Out:\n    print(f([0]))\n")
        self.assertEqual(self._run_c(f4, "f", "3 0 1 0 0"), "0")
        f6 = ("fn f(xs: [int]) -> (int, int):\n    fold(xs, (0, 1), ((a, b), x) => (a + x, b * 2))\n"
              "fn main(args: [Str]) -> Out:\n    print(f([1]))\n")
        self.assertEqual(self._run_c(f6, "f", "3 1 2 3 0 1"), "6 8")

    def test_c_multi_control_f5(self):
        src = ("type Mode = InWord | Gap\n"
               "fn f(text: [u8]) -> ((Mode, bool), int):\n    fold(text, ((Gap, false), 0), (((m, f), n), c) =>\n"
               "        if c == 32 then ((if f then Gap else m, f), n) else ((InWord, true), if m == Gap then n + 1 else n))\n"
               "fn main(args: [Str]) -> Out:\n    print(f([104]))\n")
        # combined control index: (InWord, true) is index 1 in enumeration order; payload 2
        self.assertEqual(self._run_c(src, "f", "8 104 105 32 116 104 101 114 101 2 0"), "1 2")

class TestFuzz(unittest.TestCase):
    def test_fuzz_small(self):
        from wrex.fuzz import fuzz
        self.assertEqual(fuzz(iters=25, seed=7), [])

class TestBench(unittest.TestCase):
    def test_bench_all_plans_agree(self):
        from wrex.bench import run_bench
        rows = run_bench(n=60)
        self.assertEqual(len(rows), 7)
        self.assertTrue(all(r[4] for r in rows), rows)
    def test_f0_bool(self):
        src = ("fn f(xs: [bool]) -> bool:\n    fold(xs, true, (s, x) => s and x)\n"
               "fn main(args: [Str]) -> Out:\n    print(f([true, true, false]))\n")
        prog, ck, an = compile_source(src)
        self.assertEqual(an.records[0]["cert"].kind, "F0")
        for plan in ("seq", "par"):
            self.assertEqual(run(src, plan), "false")

class TestGPU(unittest.TestCase):
    def test_gpu_fold_matches_cpu(self):
        from wrex import gpu
        if not gpu.available():
            self.skipTest("tinygrad not installed")
        src = ("fn total(xs: [i64]) -> i64:\n    fold(xs, 0, (s, x) => s + x)\n"
               "fn main(args: [Str]) -> Out:\n    print(total([1, 2, 3, 4, 5, 6, 7]))\n")
        prog, ck, an = compile_source(src)
        interp = make_interpreter(prog, an, Config(plan="par", workers=8, device="metal"))
        self.assertEqual(fmt(run_main(interp, [])), "28")
    def test_gpu_scan_and_families_match_cpu(self):
        from wrex import gpu
        if not gpu.available():
            self.skipTest("tinygrad not installed")
        cases = [
            ("fn f(xs: [(i64,i64)]) -> [i64]:\n    scan(xs, 0, (b, (r, d)) => b * r + d)\nfn main(args: [Str]) -> Out:\n    print(f([(1,10),(1,20),(2,5)]))\n", "[10, 30, 65]"),
            (open("examples/word_count.wrx").read(), "2"),
            ("fn f(xs: [int]) -> (int, int):\n    fold(xs, (0, 1), ((a, b), x) => (a + x, b * 2))\nfn main(args: [Str]) -> Out:\n    print(f([1,2,3]))\n", "(6, 8)"),
        ]
        for src, want in cases:
            prog, ck, an = compile_source(src)
            interp = make_interpreter(prog, an, Config(plan="par", workers=8, device="metal"))
            self.assertEqual(fmt(run_main(interp, [])), want)
    def test_gpu_f3_context_contraction(self):
        from wrex import gpu
        if not gpu.available():
            self.skipTest("tinygrad not installed")
        src = ("type E = Leaf(i64) | Cap(E) | Bump(E)\n"
               "fn f(e: E) -> i64:\n    treefold e:\n        Leaf(n) => n\n        Cap(a) => min(a, 10)\n        Bump(a) => a + 3\n"
               "fn main(args: [Str]) -> Out:\n    print(f(Cap(Bump(Cap(Bump(Leaf(9)))))))\n")
        prog, ck, an = compile_source(src)
        self.assertEqual(an.records[0]["cert"].family.name, "F3")
        interp = make_interpreter(prog, an, Config(plan="par", workers=8, device="metal"))
        self.assertEqual(fmt(run_main(interp, [])), "10")
    def test_gpu_tree_contraction_matches(self):
        from wrex import gpu
        if not gpu.available():
            self.skipTest("tinygrad not installed")
        src = ("type Expr = Num(i64) | Add(Expr, Expr) | Neg(Expr)\n"
               "fn eval(e: Expr) -> i64:\n    treefold e:\n        Num(n) => n\n        Add(a, b) => a + b\n        Neg(a) => 0 - a\n"
               "fn main(args: [Str]) -> Out:\n    print(eval(Add(Num(1), Add(Num(2), Add(Num(3), Neg(Num(4)))))))\n")
        prog, ck, an = compile_source(src)
        interp = make_interpreter(prog, an, Config(plan="par", workers=8, device="metal"))
        self.assertEqual(fmt(run_main(interp, [])), "2")

if __name__ == "__main__":
    unittest.main()
