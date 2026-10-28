# Wrex

Wrex is an experiment in a question that I think is underrated: how much parallelism can a compiler *prove* into existence, instead of asking the programmer to assert it?

The starting observation is a very old piece of algebra. Suppose you have a loop:

```
state = step(state, x_i)    for each x_i in some sequence
```

For each input `x`, define `t_x` to be the function `s ↦ step(s, x)`. Then the whole loop is just function composition:

```
fold = t_x1 ; t_x2 ; ... ; t_xn        (";" = "apply left, then right")
```

Composition is associative. It does not care how you parenthesize. So in principle, *every* sequential loop is a parallel reduction waiting to happen — this is basically Cayley's theorem applied to your for-loop.

In principle. The problem is that "functions from states to states" is a terrible data structure. You cannot store `t_x` in a register. The entire content of Wrex is the search for cases where you don't have to: cases where the functions your loop actually generates live inside some small family that (i) is closed under composition, and (ii) has a composition operation that is cheap. Call such a family a **representation**. If your loop has one, it runs as a parallel scan. If it doesn't, it runs sequentially, and the compiler tells you exactly which subterm is responsible. Either way, the answer is the same as the sequential program, bit for bit, including which error fires first. That last part is not a nicety; a parallelization that changes your outputs is a different program.

## The families

Everything rests on a fixed, deliberately boring set of families. For a step function, a representation is `(R, ⋄, ι, ρ, ev)` where `ρ` maps inputs into `R`, `ev` applies an element of `R` to a state, and three laws hold, of which only the third — `ev(ρ(x), s) = step(s, x)` — depends on your program. The others are proven once, for the family, forever.

| Family | Idea | ⋄ cost |
|---|---|---|
| F0 | The step is already `s ⊕ g(x)` for an associative `⊕` | 1 |
| F1 | Affine maps `s ↦ a·s + b` over a ring | 3 |
| F2 | Linear maps on vectors, over a ring or a tropical (max-plus / min-plus) semiring | (k+1)³ |
| F3 | Shifted clamps `s ↦ min(h, max(l, s + a))` | 6 |
| F4 | Finite state? Just tabulate the whole transition function | \|Q\| |
| F5 | A finite control state carrying a represented payload — this is the wreath product, and it's where the name comes from | \|Q\| · cost(payload) |
| F6 | Independent components, product of the above | sum |

Note what is *not* here. Polynomials of degree ≥ 2 are not closed under composition (`s ↦ s²` composes to `s⁴`, then `s⁸`...), so `fold(rs, 1, (s, r) => r * s * (100 - s))` gets no representation, and Wrex says so, in roughly those words. Floating-point state gets no representation either, because floating-point addition is not associative and no amount of engineering enthusiasm changes that. I generally think systems should be honest about the boundary of what they can guarantee, rather than quietly being approximately correct inside it. Wrex's version of that principle: parallelize exactly, or explain yourself.

The interesting family is F5. A tokenizer that counts words looks inherently sequential — whether you're "in a word" determines what the next character does. But split the state into a finite control part (in-word / in-gap) that depends only on itself and the input, and a counter that updates affinely *for each fixed control state*, and the whole thing is again a scan, now over a wreath product. A fair amount of "inherently sequential" code has this shape once you look at it as algebra instead of as control flow.

## How it runs

Compilation is: parse, type check, then for each `fold`, `scan`, and `treefold`, attempt representation inference by symbolic normalization — ring normal forms for F0–F2, lattice normal forms for F3 and the tropical cases. Because the normal forms are canonical, checking law L3 is just checking that two normal forms are equal. This is a property I value a lot: inference can be as heuristic and dumb as it wants, because the *checker* is decidable and small. A bug in inference loses you performance. It cannot lose you correctness.

Execution then picks a plan per call site with a cost model — sequential, blocked parallel scan (summarize blocks, combine summaries, replay), balanced fold reduction, fork-join tree evaluation, or tree contraction with composed contexts for list-shaped trees. The guard is evaluated on the actual input size at runtime, because a representation existing and a representation being *worth it* are different claims, and conflating them is how parallel runtimes end up slower than the loop they replaced. Traps are carried through the parallel plans as values indexed by input position, so the error you get is the error the sequential program would have given you, at the same place.

## Try it

```sh
python3 -m venv .venv && .venv/bin/pip install -e .
.venv/bin/wrex run examples/word_count.wrx        # 2
.venv/bin/wrex explain examples/balances.wrx      # F1 affine, plan + threshold
.venv/bin/wrex check examples/logistic.wrx        # sequential, and why
.venv/bin/wrex test examples/stock.wrx            # family laws + differential tests
```

`wrex explain` is, to me, the most important command in the project. A system that makes decisions on your behalf owes you a legible account of them: which family, what dimension, what the scan costs, where the crossover point is — or, if it gave up, the smallest subterm that made it give up.

## Status and honest limitations

Version 0.9 finalizes user-defined families and closes out the 0.x line. User families (0.8) now generate C like the built-in families, serve as tree-fold contexts, and are exercised against the generated C by the differential fuzzer; only the GPU backend still falls back to exact CPU plans for them, because an arbitrary user operator has no tensor form. What remains genuinely open is stated as such in the spec: nested-parallel flattening beyond the outer-owns rule, user families with a representation distinct from the state, mechanized law proofs, and a checked per-program `@reassoc` bound. Version 0.8 added user-defined families (spec Section 6.8, open question 4) and a stated `@reassoc` error bound (open question 5). Declare a monoid with `family growth over int identity 0: (a, b) => a + b + a * b` and a step `(s, x) => growth(s, x)` folds and scans in parallel under it: the compiler machine-checks the family's laws at the declaration — exhaustively when the state domain is small, otherwise on edge values and 800 seeded-random triples — and a family that fails (subtraction, say) is rejected with the counterexample, runs sequentially, and stays callable as an ordinary operator, so the sequential answer never changes. Certified user families compose into F6 products, serve as F5 payloads and tree-fold contexts, and generate C; only the GPU backend falls back to exact CPU plans for them, and `wrex explain` says how each family was checked. `wrex explain` now also prints the `@reassoc` bound (`|error| <= ceil(log2 n) * 2^-53 * sum |terms|`, balanced pairing, sequential still exact). See `examples/growth.wrx`. Before that, version 0.7 implemented the fusion rules of Section 8.3 and finished the release pipeline. A `map` feeding a fold or scan is fused into the step — the mapped array is never materialized — with trap order preserved by construction (the map's traps are checked in input order before the fold, exactly as the strict reference semantics evaluates them; values are then recomputed lazily, which is safe because programs are pure). Consecutive scans over the same input fuse into one product scan, falling back to the sequential pair if a step traps so the first scan's trap always wins. `wrex explain` now prints the symbolic `ρ` for each certificate, the finite families' laws are joined by exhaustive bounded-domain checks for F0–F3 including the infinity boundaries, and pushing a version tag builds the wheel and sdist and attaches them to the release automatically. Generated C scans now run all three phases on pthreads — parallel summarize, prefix starts, and parallel per-block replay, with per-state traps recorded by index so the first trap still wins. The conformance suite verifies the finite families *exhaustively* (every element of F4 over three states, every F5 element over a small payload), not just by sampling. Inference completeness for the affine and clamp families is characterized by tests that enumerate the normal forms and the exact boundary cases outside them (spec open question 1). `wrex bench --c` reports generated-C timings as the native baseline, and the release ships a wheel that CI installs into a clean venv and runs. `@reassoc` on a function permits parallel `f64` folds and scans there: they reassociate (balanced pairing, left operand first), so rounding may differ from the sequential order — Section 11 of the spec states this precisely, sequential plans stay exact, and `wrex explain` marks such certificates. GPU tree contraction now covers F0/F1/F3/F2 contexts, not just affine ones. Nested folds have an explicit scheduling rule: the outer plan owns the parallelism, and anything inside a step it is evaluating runs sequentially. And there is CI now, running the test suite, fuzzer, benchmarks, and a codegen smoke test on every push, with and without tinygrad installed. Tree folds may now return tuples — contexts are inferred componentwise (products and coupled linear forms both work, so a tree fold computing `(sum, count)` contracts like any other). The C backend handles F5 states with several control components, encoding the combined control state densely. `wrex fuzz --iters=200` generates random programs across the families and requires Seq, Par, auto, GPU, and the generated C to agree on every case; it is the cheapest confidence per line in the repo. And plan selection now refuses parallel plans for representations whose compose cost passes a calibrated benefit threshold (spec open question 2): a big-enough table is a sequential loop with extra steps, and the compiler should say so by its choices.

- **C backend.** `wrex codegen file.wrx -o out.c` emits standalone C (pthreads, three-phase blocked scan / balanced fold) for certified folds and scans in *all* families F0–F6, including captured scalar parameters and scalar helper functions used by coefficients. Compile it with any C compiler; it reads any free parameters, then `n`, the inputs, and the initial state on stdin, exchanging finite states as dense indices. Generated code is tested byte-for-byte against the interpreter.
- **GPU backend.** With tinygrad installed (`pip install -e '.[gpu]'`), folds and scans for *all* families F0–F6 lower to tensor operations — batched matrix products for F2, gathers for F4/F5 control, componentwise composition for F6 — with operand order preserved, because ⋄ is not commutative and pretending otherwise is how you get wrong answers quickly. Tree folds with F1 contexts also contract on the GPU: each round's independent context compositions are batched into a single tensor operation. Any failure falls back to the exact CPU plan, and in `auto` mode the GPU is only used when the cost model says it wins.
- **Benchmarks.** `wrex bench` runs the Appendix B suite (prefix sum, linear recurrence, saturating counters, tokenizer, Viterbi-style tropical fold, balanced and list-shaped expression trees) under every plan and reports whether they agree — they must — alongside timings. At small `n` the parallel and GPU plans lose to sequential, exactly as the cost model predicts; that crossover behavior is the point of having one.
- Steps may now use `match` on the input (literal, wildcard, and nullary-constructor arms), and the inference engine normalizes it into the same conditional normal forms as `if`. See the open questions in [spec.md](spec.md) — representation-size-vs-benefit thresholds and user-defined families with machine-checked laws were the two I'd most like to see attacked — both have since landed (0.7 and 0.8); nested-parallel flattening is the one I'd point at next.

The full design — laws, proofs sketches, trap semantics, cost formulas, and the phase plan — is in [spec.md](spec.md). Tests (`python3 -m unittest discover -s tests`) include randomized checks that each family's laws actually hold, and differential tests that every plan agrees with the reference interpreter. Those tests exist because of the obvious meta-point: a language whose entire pitch is "trust me, same answer" should be required to demonstrate it, mechanically, on every change.
