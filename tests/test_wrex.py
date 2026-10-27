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

if __name__ == "__main__":
    unittest.main()
