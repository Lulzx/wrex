# Wrex Language Specification

**Version:** 0.3
**Status:** Implemented. The reference implementation covers Sections 1–14: parser, type checker, reference interpreter, representation inference for F0–F6, CPU plans, tree contraction, the tinygrad backend, a C code generator for certified folds and scans in all families F0-F6, and GPU tree contraction for F1 contexts (see the repository README for the exact backend coverage and remaining gaps).

---

## 1. Overview

Wrex is a small, pure, functional language for programs whose parallelism is not visible in their syntax. The programmer writes ordinary sequential folds, scans, and tree recursions. The compiler finds a parallel execution by recognizing the algebraic structure of the computation, and it proves that the parallel execution produces exactly the same result as the sequential one.

The core observation is classical. Every sequential loop is a monoid action: a sequence of state transformations applied one after another. Function composition is associative, so any such loop can in principle be evaluated as a parallel reduction or scan. The obstacle is that arbitrary state transformations are too large to store and compose. The Wrex compiler performs **representation inference**: it searches for a compact, closed family of transformations that contains every step of the loop, together with an efficient composition operator. When it finds one, the loop runs as a parallel scan over that family. When it does not, the loop runs sequentially and the compiler explains why.

The name comes from the **wreath product**, the construction from algebra that combines a finite state machine with the transformations it carries along (family F5, Section 6.6). It is what lets Wrex parallelize loops that look inherently sequential, such as tokenizers that count as they go. The underlying fact that makes every loop a composable transformation is Cayley's theorem: every monoid embeds in a monoid of functions under composition.

### 1.1 Goals

1. **Exactness.** Every execution plan produces results identical to the reference sequential semantics, bit for bit, including which error is reported when a program traps.
2. **Implicit parallelism.** Programs contain no threads, tasks, kernels, locks, or synchronization.
3. **Irregular parallelism.** Parallel evaluation applies to state-dependent loops and to unbalanced trees, not only to maps and reductions with an obvious associative operator.
4. **Explainability.** For every fold, scan, and tree fold, the compiler can report the inferred representation, its size, the chosen plan, and the estimated cost, or the precise reason no representation was found.
5. **A small trusted core.** Correctness rests on a fixed set of representation families whose laws are proven once, plus a decidable per-program check.

### 1.2 Non-goals for version 0.1

- Distributed execution.
- Tree contraction on GPUs. Tree folds run on the CPU runtime (Section 13.5.1).
- Parallel reassociation of floating-point arithmetic.
- General-purpose effects, mutation, or concurrency primitives.
- User-defined representation families.

---

## 2. Mathematical foundation

### 2.1 Loops as monoid actions

Let `S` be a state type, `X` an input type, and `step : S × X → S`. For an input sequence `x₁ … xₙ` and initial state `s₀`, the sequential fold is

```
fold(x, s₀, step) = step(… step(step(s₀, x₁), x₂) …, xₙ)
```

For each input `x`, write `tₓ : S → S` for the map `s ↦ step(s, x)`. Writing `f ; g` for "apply `f`, then `g`", the fold equals `(tₓ₁ ; tₓ₂ ; … ; tₓₙ)(s₀)`. The operator `;` is associative and has the identity function as its unit, so the transformations form a monoid. Any parenthesization of the composition gives the same function, which permits a balanced, logarithmic-depth evaluation.

### 2.2 Representations

A **representation** of a step function `step : S × X → S` is a tuple `(R, ⋄, ι, ρ, ev)` where

- `(R, ⋄, ι)` is a monoid: `⋄` is associative and `ι` is its identity,
- `ρ : X → R` maps each input to an element of `R`,
- `ev : R × S → S` applies an element of `R` to a state,

and the following laws hold for all `r₁, r₂ ∈ R`, `s ∈ S`, `x ∈ X`:

| Law | Statement | Meaning |
|---|---|---|
| L1 | `ev(ι, s) = s` | The identity does nothing |
| L2 | `ev(r₁ ⋄ r₂, s) = ev(r₂, ev(r₁, s))` | `⋄` means "first `r₁`, then `r₂`" |
| L3 | `ev(ρ(x), s) = step(s, x)` | `ρ` faithfully represents each step |

**Theorem 1 (Fold correctness).** If `(R, ⋄, ι, ρ, ev)` satisfies L1–L3, then for every sequence `x₁ … xₙ` and every parenthesization `P` of the product `ρ(x₁) ⋄ … ⋄ ρ(xₙ)`,

```
fold(x, s₀, step) = ev(P, s₀).
```

*Proof sketch.* By associativity, every parenthesization of the product denotes the same element `r`. By induction on `n` using L2 and L3, `ev(ρ(x₁) ⋄ … ⋄ ρ(xₙ), s₀)` equals the sequential fold. The case `n = 0` is L1. ∎

**Theorem 2 (Scan correctness).** Under the same hypotheses, the `i`-th state of the sequential scan equals `ev(ρ(x₁) ⋄ … ⋄ ρ(xᵢ), s₀)`. Any parallel prefix algorithm over `(R, ⋄)` therefore computes the exact scan.

Note that `R` need not represent `S → S` faithfully. Laws L1–L3 are the only requirements.

### 2.3 Where obligations live

Laws L1, L2, and the associativity of `⋄` are properties of a **family** and are proven once, as part of the Wrex trusted kernel (Section 6). Law L3 depends on the program. It is the only per-program proof obligation, and for every family in this specification it is decidable (Section 7.4).

### 2.4 The closure boundary

A family is useful only if it is closed under `⋄` and its elements stay small. Some natural families are not closed. For example, polynomials of degree `d ≥ 2` in the state compose to degree `d²`, so a step such as `s ↦ s·s + x` has no fixed-size polynomial representation. Wrex does not approximate in such cases. It falls back to sequential execution and reports the obstruction.

---

## 3. Lexical structure

- **Encoding:** UTF-8 source files with the extension `.wrx`.
- **Comments:** `#` to end of line.
- **Layout:** Blocks are introduced by `:` at the end of a line and delimited by indentation, as in Python. Tabs are rejected.
- **Identifiers:** `[a-z_][A-Za-z0-9_]*` for values and functions, `[A-Z][A-Za-z0-9_]*` for types and constructors.
- **Integer literals:** decimal, `0x` hexadecimal, `0b` binary, with optional `_` separators. A suffix such as `42i64` or `7u8` fixes the type; unsuffixed literals take their type from context.
- **Float literals:** `1.5`, `2e-3`, `0.25f64`.
- **Keywords:** `fn type let if then else match treefold true false and or not`.

---

## 4. Syntax

### 4.1 Grammar

```ebnf
program     = { decl } ;
decl        = typedecl | fndecl ;

typedecl    = "type" TypeName "=" variant { "|" variant } ;
variant     = Ctor [ "(" type { "," type } ")" ] ;

fndecl      = { annotation } "fn" name "(" [ params ] ")" "->" type ":" block ;
annotation  = "@par" | "@seq" | "@finite" ;
params      = param { "," param } ;
param       = name ":" type ;

type        = "int" | "i64" | "u64" | "i32" | "u32" | "u8" | "bool" | "f64"
            | TypeName
            | "(" type "," type { "," type } ")"
            | "[" type "]" ;

block       = expr | INDENT { binding } expr DEDENT ;
binding     = "let" pattern "=" expr ;

expr        = lambda | ifexpr | matchexpr | treefoldexpr | binexpr ;
lambda      = "(" [ pattern { "," pattern } ] ")" "=>" block ;
ifexpr      = "if" expr "then" expr "else" expr ;
matchexpr   = "match" expr ":" INDENT { arm } DEDENT ;
treefoldexpr= "treefold" expr ":" INDENT { arm } DEDENT ;
arm         = pattern "=>" block ;

pattern     = name | "_" | literal
            | "(" pattern "," pattern { "," pattern } ")"
            | Ctor [ "(" pattern { "," pattern } ")" ] ;

binexpr     = unary { binop unary } ;
binop       = "+" | "-" | "*" | "/" | "%" | "==" | "!=" | "<" | "<=" | ">" | ">="
            | "and" | "or" | "&" | "|" | "^" | "<<" | ">>" ;
unary       = [ "-" | "not" ] postfix ;
postfix     = atom { "(" [ args ] ")" | "[" expr "]" } ;
atom        = literal | name | Ctor | "(" expr { "," expr } ")" | "[" [ args ] "]" ;
args        = expr { "," expr } ;
```

### 4.2 Built-in combinators

| Combinator | Type | Meaning |
|---|---|---|
| `map(xs, f)` | `[A], (A → B) → [B]` | Apply `f` to each element |
| `zip(xs, ys)` | `[A], [B] → [(A, B)]` | Pair elements; traps if `len(xs) != len(ys)` |
| `range(a, b)` | `int, int → [int]` | `a, a+1, …, b-1` (empty if `a ≥ b`) |
| `len(xs)` | `[A] → int` | Length |
| `xs[i]` | `[A], int → A` | Indexing; traps if out of bounds |
| `fold(xs, s₀, step)` | `[X], S, (S, X → S) → S` | Left fold (Section 5.3) |
| `scan(xs, s₀, step)` | `[X], S, (S, X → S) → [S]` | Inclusive scan; result has length `len(xs)` |
| `treefold t: arms` | `T → V` | Catamorphism over an algebraic type (Section 9) |
| `min`, `max` | `int, int → int` (and other ordered types) | Minimum and maximum |
| `clamp(v, lo, hi)` | `int, int, int → int` | `min(hi, max(lo, v))`; requires `lo ≤ hi` |

Lambdas appear only as arguments to combinators. They cannot be stored in data structures or returned from functions. This restriction keeps every program first-order after inlining.

### 4.3 Annotations

- `@par` on a function asserts that every `fold`, `scan`, and `treefold` in its body has a representation. If one does not, compilation fails with the obstruction report.
- `@seq` forces sequential plans for the function's body.
- `@finite` on a function permits the compiler to treat machine-integer state components as finite control (Section 7.2), enabling tabulation.

---

## 5. Static and dynamic semantics

### 5.1 Types

| Type | Values | Arithmetic |
|---|---|---|
| `int` | Mathematical integers | Exact; never overflows |
| `i64`, `u64`, `i32`, `u32`, `u8` | Machine integers | Wrapping, i.e. arithmetic in the ring ℤ/2ᵏ |
| `bool` | `true`, `false` | Boolean algebra |
| `f64` | IEEE 754 binary64 | Round-to-nearest-even; never reassociated |
| user types | Constructors of algebraic data types | — |
| `(A, B, …)` | Tuples | — |
| `[A]` | Immutable arrays | — |

Integer division and remainder truncate toward zero. Division or remainder by zero traps. Signed machine integers use two's complement; on `i32::MIN` and `i64::MIN`, `/` by `-1` wraps to `MIN` and `%` by `-1` yields `0`.

A user type is **finite** if all of its constructors have only finite fields. `bool`, finite user types, and tuples of finite types are finite. Machine integers are finite in principle but are treated as numeric unless `@finite` applies.

### 5.2 Purity and effects

The core of Wrex is pure. The only effects are traps (Section 5.4) and the input/output performed by the entry point:

```
fn main(args: [Str]) -> Out
```

where `Out` is built from `print(v)`, `write(path, v)`, and sequencing with `;`. Inputs are read with `read_i64s(path) -> [i64]` and similar functions, evaluated before the pure program runs. Effects never occur inside folds, scans, or tree folds.

### 5.3 Reference semantics

The reference semantics is a strict, left-to-right, call-by-value interpreter. In particular:

```
fold([], s, f)        = s
fold([x] ++ xs, s, f) = fold(xs, f(s, x), f)

scan(xs, s, f)[i]     = fold(xs[0..=i], s, f)

treefold t: arms      = evaluate children left to right, then apply the arm
```

The reference interpreter is part of the specification. Every execution plan must agree with it.

### 5.4 Traps

A trap is a runtime error: division or remainder by zero, out-of-bounds indexing, a `clamp` with `lo > hi`, a `zip` of arrays with unequal lengths, or a failed `match`. Under the reference semantics, a program that traps reports the trap reached first in left-to-right evaluation order.

**Trap precision.** Every execution plan reports exactly the trap the reference semantics reports, or no trap if the reference semantics reports none. Section 8.4 describes how parallel plans achieve this.

### 5.5 The plan-independence guarantee

> For every program `P` and input `I`, every execution plan the compiler selects for `P` produces the same output on `I` as the reference interpreter, including identical bit patterns for all values and the identical trap, if any.

This guarantee is the defining property of the language. Representation inference (Section 7) and the plan selector (Section 10) operate entirely within it.

---

## 6. Representation families

Each family below is specified by its elements, identity, composition, evaluation, and representation cost (the cost of one `⋄`, in scalar operations). The laws L1, L2, and associativity are part of the trusted kernel and should be mechanized in a proof assistant before version 1.0.

Throughout, `r₁ ⋄ r₂` means "first `r₁`, then `r₂`".

### 6.1 F0: Direct monoid

**Applies when** `step(s, x) = s ⊕ g(x)`, where `⊕` is one of the following associative operators with identity `e`: `+`, `*` on `int` or on a wrapping integer type; `min`, `max` on `int` (with identities `+inf`, `-inf`); `and`, `or` on `bool`; `&`, `|`, `^` on machine integers.

| | |
|---|---|
| Elements | values of `S` |
| `ι` | `e` |
| `r₁ ⋄ r₂` | `r₁ ⊕ r₂` |
| `ev(r, s)` | `s ⊕ r` |
| `ρ(x)` | `g(x)` |
| Cost | 1 |

`int` values here range over the extended integers ℤ ∪ {−∞, +∞} only where `min` and `max` require them. Operations between infinities of opposite sign never arise in this family.

### 6.2 F1: Affine maps over a commutative ring

**Applies when** the state is a single ring element and `step(s, x) = a(x)·s + b(x)`. The ring is `int` or a wrapping integer type ℤ/2ᵏ. Both are commutative rings, so the family is exact.

| | |
|---|---|
| Elements | pairs `(a, b)` representing `s ↦ a·s + b` |
| `ι` | `(1, 0)` |
| `(a₁, b₁) ⋄ (a₂, b₂)` | `(a₂·a₁, a₂·b₁ + b₂)` |
| `ev((a, b), s)` | `a·s + b` |
| Cost | 3 |

### 6.3 F2: Linear maps over a semiring

**Applies when** the state is a vector `v ∈ Kᵏ` and each new component is an affine combination of old components, with coefficients that depend only on the input:

```
v'ᵢ = (⊕ⱼ Aᵢⱼ(x) ⊗ vⱼ) ⊕ bᵢ(x)
```

`K` is either a commutative ring as in F1 (with `⊕ = +`, `⊗ = ·`), or one of the tropical semirings over extended `int`:

- **max-plus**: `⊕ = max`, `⊗ = +`, zero `−∞`, one `0`,
- **min-plus**: `⊕ = min`, `⊗ = +`, zero `+∞`, one `0`.

Tropical semirings are admitted only over `int`, because wrapping addition does not distribute over `max` or `min`.

| | |
|---|---|
| Elements | `(k+1)×(k+1)` matrices `M = [[A, b], [0_K, 1_K]]` over `K` |
| `ι` | identity matrix |
| `M₁ ⋄ M₂` | `M₂ ⊗ M₁` (matrix product over `K`) |
| `ev(M, v)` | `M ⊗ [v; 1_K]`, dropping the last coordinate |
| Cost | `(k+1)³` |

The default dimension limit is `k ≤ 8`. F1 is the special case `k = 1` over a ring and is preferred when it applies, because its cost is lower.

### 6.4 F3: Shifted clamps

**Applies when** the state is a single `int` and the step is a composition of additions of input-dependent constants with `min`, `max`, and `clamp` against input-dependent bounds. This covers saturating counters, running maxima with resets, and bounded buffers.

Elements are triples `(a, l, h)` with `l ≤ h`, over extended integers, representing

```
s ↦ min(h, max(l, s + a))
```

| | |
|---|---|
| `ι` | `(0, −∞, +∞)` |
| `(a₁, l₁, h₁) ⋄ (a₂, l₂, h₂)` | `(a₁ + a₂, min(max(l₁ + a₂, l₂), h₂), max(min(h₁ + a₂, h₂), l₂))` |
| `ev((a, l, h), s)` | `min(h, max(l, s + a))` |
| Cost | 6 |

*Derivation.* Applying the second clamp after the first gives `clamp(clamp(s + a₁ + a₂, l₁ + a₂, h₁ + a₂), l₂, h₂)`. Two nested clamps with `l₁' ≤ h₁'` and `l₂ ≤ h₂` equal a single clamp to the interval `[min(max(l₁', l₂), h₂), max(min(h₁', h₂), l₂)]`. This covers the overlapping case (the intersection) and both disjoint cases (a constant at `l₂` or `h₂`). The resulting interval always satisfies `l ≤ h`. Infinite bounds follow the rule `±∞ + a = ±∞` for finite `a`.

Common steps and their representations:

| Step | `ρ(x)` |
|---|---|
| `s + x` | `(x, −∞, +∞)` |
| `max(s, x)` | `(0, x, +∞)` |
| `min(s + x, cap)` | `(x, −∞, cap)` |
| reset to `c` | `(0, c, c)` |

### 6.5 F4: Finite transformations

**Applies when** the entire state type `Q` is finite. Every step function on a finite state is representable by tabulation.

| | |
|---|---|
| Elements | arrays `r : Q → Q ∪ {⊥ᵢ}` |
| `ι` | `q ↦ q` |
| `(r₁ ⋄ r₂)(q)` | `r₂(r₁(q))`, where `⊥` propagates (Section 8.4) |
| `ev(r, q)` | `r(q)` |
| `ρ(x)` | `q ↦ step(q, x)`, evaluated for every `q ∈ Q` |
| Cost | `|Q|` |

The default limit is `|Q| ≤ 256`. States are numbered densely so that tables are flat arrays.

### 6.6 F5: Finite control with a represented payload

**Applies when** the state is a pair `(q, v)` where `q` ranges over a finite type `Q`, the new control state depends only on the old control state and the input, and for each fixed control state the payload update lies in a family `R` constructed from F0–F3 and F6:

```
q' = δ(q, x)
v' = ev(σ(q, x), v)      where σ(q, x) ∈ R
```

This is a wreath product of `R` with the transformation monoid of `Q`. It covers state machines that carry counters or accumulators, such as tokenizers that count tokens or parsers that track depth.

| | |
|---|---|
| Elements | arrays `r : Q → (Q × R) ∪ {⊥ᵢ}` |
| `ι` | `q ↦ (q, ι_R)` |
| `(r₁ ⋄ r₂)(q)` | let `(q₁, m₁) = r₁(q)` and `(q₂, m₂) = r₂(q₁)` in `(q₂, m₁ ⋄_R m₂)` |
| `ev(r, (q, v))` | let `(q', m) = r(q)` in `(q', ev_R(m, v))` |
| `ρ(x)` | `q ↦ (δ(q, x), σ(q, x))` |
| Cost | `|Q| · cost(⋄_R)` |

*Associativity.* Both groupings of three elements map `q` to the control state reached after all three steps, paired with the `⋄_R`-product of the three payload transformations taken in order. These agree because `⋄_R` is associative.

### 6.7 F6: Products

**Applies when** the state can be partitioned into sub-tuples whose components update independently of one another: each sub-tuple's new value depends only on its own old value and the input. Each sub-tuple receives its own family, and

```
(r₁, r₂) ⋄ (r₁', r₂') = (r₁ ⋄ r₁', r₂ ⋄ r₂')
```

Components within a sub-tuple that depend on each other linearly are handled by F2 instead. Components that depend on each other in other ways (aside from finite control in F5) are not representable.

---

## 7. Representation inference

### 7.1 Input and output

Inference runs on each `fold`, `scan`, and tree-fold arm after inlining and type checking. Its output is either

- a **representation certificate**: the chosen family, its dimension, compiled functions for `ρ`, `⋄`, and `ev`, and a discharged L3 obligation, or
- an **obstruction report**: the smallest subterm of the step function that prevents every family from applying, and the reason.

### 7.2 State decomposition

1. Flatten the state type into components.
2. Classify each component as **control** (finite type, or a machine integer under `@finite`) or **numeric** (`int`, machine integers, `f64`).
3. Build the dependency graph: an edge from component `c` to component `d` means the new value of `d` reads the old value of `c`.
4. Partition the dependency graph into connected components. Independent connected components form a product under F6 and are inferred separately.
5. Within a connected component, if any control component depends on a numeric component, family F5 cannot apply. If the whole component is finite, F4 still applies. If control components depend only on themselves and the input, F5 reduces inference to the numeric payload components, which may themselves decompose into independent sub-components via F6.

### 7.3 Symbolic normalization

For each assignment of control states (the elements of `Q`, or a single dummy state if there are no control components), evaluate the step body symbolically. Numeric state components become symbolic variables. Input-dependent subterms become opaque coefficients. Branches whose condition depends only on the input become conditional coefficients. Branches whose condition depends on numeric state are permitted only in the clamp and tropical normal forms.

To preserve exact trap semantics (Section 8.4):
- If any potentially trapping operation (divisor of `/` or `%`, array index, `clamp` bounds, or `match` scrutinee) depends on a numeric state variable, numeric families (F0–F3) are rejected before algebraic simplification, so that `0 * (1 / s)` is never simplified to `0`.
- Input-only subterms that may trap are retained and evaluated in `ρ(x)` even if their algebraic coefficient simplifies to `0`.

The normalizer rewrites each numeric component or coupled numeric group into one of two normal forms:

- **Ring normal form:** a polynomial in the state variables with coefficients built from input terms, using the commutative-ring laws. The step is in F1 or F2 over a ring if and only if every polynomial has total degree at most 1 in the state variables.
- **Lattice normal form** (over `int` only): an expression built from `max`, `min`, and addition of input terms, normalized with the distributive and absorption laws of the max-plus and min-plus semirings. The step is in F3 if its normal form is a single shifted clamp of one variable, and in a tropical F2 if each component is a max (or min) of state variables plus coefficients.

For a single numeric component, candidate families are tried in order of increasing cost: F0, F1, F3. For coupled numeric components, F2 over a ring is tried before F2 tropical. When all components of a connected group are finite, F4 is available as a fallback when no cheaper family applies. The plan selector (Section 10) may still choose a sequential plan.

### 7.4 Discharging law L3

The obligation `ev(ρ(x), s) = step(s, x)` is checked as follows:

- **F0–F3, F6:** Normalize both sides symbolically with `s` and `x` as variables. Ring normal forms and lattice normal forms are canonical, so equality of normal forms decides the obligation.
- **F4:** L3 holds by construction, because `ρ(x)` is defined by evaluating `step` at every control state.
- **F5:** By construction for the control part. The payload part is checked per control state as above.

The checker is part of the trusted kernel. It does not depend on the inference heuristics: a bug in inference can only cause a missed representation, never an incorrect one.

### 7.5 Obstruction reports

When no family applies, the report names the offending subterm and the violated condition. Examples:

```
fold at stats.wrx:14 runs sequentially.
  The new value of `acc` contains `acc * acc`, which has degree 2 in the state.
  Polynomials of degree 2 or more are not closed under composition.
```

```
fold at lexer.wrx:31 runs sequentially.
  The control state `mode` depends on the numeric state `depth` (line 35).
  Finite-control families require the control state to depend only on itself and the input.
```

---

## 8. Parallel evaluation of folds and scans

### 8.1 Three-phase blocked scan

For `scan(xs, s₀, step)` with `n` elements on `p` workers, the runtime uses a three-phase algorithm:

1. **Summarize.** Split the input into `p` contiguous blocks. Each worker computes the product `ρ(x_first) ⋄ … ⋄ ρ(x_last)` for its block.
2. **Combine.** Compute the exclusive prefix product of the `p` block summaries, and from it the starting state of each block: `start_b = ev(prefix_b, s₀)`.
3. **Replay.** Each worker runs the original `step` sequentially through its block from `start_b`, writing the scan output.

Phase 3 uses `step` directly rather than `ev`, so the representation is needed only to summarize blocks. The total work is about `n·(c_ρ + c_⋄ + c_step)`.

### 8.2 Folds

For `fold`, phase 3 is unnecessary. The result is `ev(summary_1 ⋄ … ⋄ summary_p, s₀)`, with the `p` summaries combined by a balanced tree.

### 8.3 Fusion

A `map` that feeds a `fold` or `scan` is fused into `ρ`, so the mapped array is never materialized. Consecutive scans whose states form a product are fused into a single scan over the product family (F6).

### 8.4 Trap precision in parallel plans

Representation elements can carry a trap value `⊥ᵢ`, where `i` is the index of the input element that caused it:

- If computing `ρ(xᵢ)` traps in a way that does not depend on the state, `ρ(xᵢ) = ⊥ᵢ`.
- In F4 and F5, if `step(q, xᵢ)` traps for a particular control state `q`, only the table entry for `q` becomes `⊥ᵢ`. The trap is raised only if that control state is actually reached at that position, exactly as in the sequential semantics.
- `⋄` propagates traps. When both operands carry traps, the one with the smaller index wins.

Because the sequential semantics reports the first trap in input order, and `⋄` preserves the leftmost trap, parallel plans report the same trap as the reference interpreter. The replay phase (Section 8.1) re-executes `step`, so it encounters any trap at its true position.

---

## 9. Tree folds

### 9.1 Semantics

For an algebraic type `T`, a `treefold` gives one arm per constructor. In each arm, pattern variables bound to recursive fields stand for the **results** of folding those children, not the children themselves:

```
type Expr = Num(i64) | Add(Expr, Expr) | Mul(Expr, Expr) | Neg(Expr)

fn eval(e: Expr) -> i64:
    treefold e:
        Num(n)    => n
        Add(a, b) => a + b
        Mul(a, b) => a * b
        Neg(a)    => 0 - a
```

The reference semantics evaluates children left to right, then the arm.

### 9.2 Contexts

Fix a node with constructor `C`, payload `p`, and child results known for every recursive field except field `j`. The arm then defines a function `V → V` of the missing result. Such a function is called a **context**. Contexts compose: plugging one context into the hole of another gives a context.

A tree fold has a **context representation** if there is a family `(R, ⋄, ι, ev)` over the result type `V` satisfying L1 and L2, and a function `ρ_ctx(C, p, j, known results) ∈ R` such that

```
ev(ρ_ctx(C, p, j, known), v) = arm_C(p, known with v placed at field j)
```

for all arguments. This is the tree analogue of law L3 and is checked the same way.

For `eval` above, every context is affine in the missing result: `v ↦ c + v` for `Add`, `v ↦ c·v` for `Mul`, `v ↦ 0 − v` for `Neg`. The arm `a * b` is not affine in the pair `(a, b)`, but it is affine in each argument separately. **Multilinear arms are sufficient**, because contexts always have exactly one hole.

### 9.3 Tree contraction

When a context representation exists, the runtime evaluates the tree by parallel tree contraction, in rounds:

- **Rake:** A node whose children are all known evaluates its arm and becomes known.
- **Unary conversion:** A node with exactly one unknown child becomes a context `ρ_ctx(…) ∈ R`.
- **Compress:** Along chains of context nodes, adjacent contexts are merged with `⋄`, using random mate selection to choose independent pairs. Merging a child context `r_child` into its parent context `r_parent` computes `r_child ⋄ r_parent` (child applied first, parent applied second, matching L2).

For trees of bounded arity, contraction performs `O(n)` work and completes in `O(log n)` rounds with high probability, regardless of the tree's shape. A tree shaped like a list, which provides no parallelism to fork-join evaluation, contracts in logarithmic depth. The default arity bound is 8; constructors with more recursive fields are binarized by the compiler.

### 9.4 Fork-join evaluation and trap ordering

When a tree is balanced, ordinary fork-join recursion over children is cheaper than contraction because it needs no contexts. Every tree node allocated by the runtime records its size and height at construction, at constant cost per node. The plan selector reads these values at the call site (Section 10.2).

Fork-join evaluation needs no representation and is always available. Contraction is available only when a context representation exists.

To preserve trap precision (Section 5.4) in `ForkJoin` and `Contract`, if any worker encounters a trap during parallel tree evaluation, the runtime resolves the exact first trap in left-to-right post-order (either by comparing post-order node ranks derived from subtree `size` fields or by re-running the sequential reference traversal on the trapping tree).

---

## 10. Execution plans and cost model

### 10.1 Plans

| Plan | Applies to | Requires |
|---|---|---|
| `Seq` | everything | nothing |
| `ParFold`, `ParScan` | `fold`, `scan` | a representation |
| `ForkJoin` | `treefold` | nothing |
| `Contract` | `treefold` | a context representation |
| `ParMap` | `map`, `zip`, `range` | nothing |
| `GpuFold`, `GpuScan`, `GpuMap` | `fold`, `scan`, `map` | a representation (except `GpuMap`) and a device that supports the state's types (Section 13.5.6) |

### 10.2 Cost model

Let `n` be the input size, `p` the number of workers, `c_step`, `c_ρ`, `c_⋄`, `c_ev` the estimated costs of one step, one `ρ`, one `⋄`, and one `ev`, and `c_sync` the cost of one synchronization. The estimates come from static operation counts multiplied by per-operation costs calibrated on the target machine at installation.

```
T_Seq     = n · c_step
T_ParFold = (n/p)·(c_ρ + c_⋄) + log₂p·(c_⋄ + c_sync) + c_ev
T_ParScan = (n/p)·(c_ρ + c_⋄ + c_step) + p·c_⋄ + 2·c_sync
```

For trees with size `n` and height `h`:

```
T_ForkJoin ≈ (n/p)·c_arm + h·c_sync
T_Contract ≈ κ·(n/p)·(c_ctx + c_⋄) + κ'·log₂n·c_sync
```

where `κ` and `κ'` are calibrated constants of the contraction implementation.

For GPU plans, let `B` be the bytes moved between host and device, `β` the transfer bandwidth, `c_launch` the cost of one kernel launch, and `c_⋄^dev` the device cost of one `⋄`:

```
T_GpuFold = B/β + ⌈log₂n⌉·c_launch + (n/P_dev)·(c_ρ + c_⋄)^dev
T_GpuScan = B/β + 2·⌈log₂n⌉·c_launch + (2n/P_dev)·(c_ρ + c_⋄ + c_ev)^dev
```

where `P_dev` is the device's effective parallel width. Kernel fusion performed by the backend reduces the launch term; the calibration step measures it rather than assuming it.

### 10.3 Selection

The compiler emits code for every applicable plan and a guard at the call site that evaluates the cost formulas using the actual `n` (and `h`, for trees) and selects the cheapest plan. Because all plans satisfy the plan-independence guarantee (Section 5.5), selection affects only speed.

For example, a tokenizer with 12 control states and a counter payload (F5 over F0) has `c_⋄ ≈ 12`, and computing `ρ(x)` evaluates the step once per control state, so `c_ρ ≈ 12·c_step`. With `c_step ≈ 5`, the parallel scan wins only when `(n/p)·(60 + 12 + 5) < 5n`, that is, roughly `p ≥ 16`. On smaller machines the guard keeps this scan sequential. This illustrates why representation size, not just its existence, decides the plan.

---

## 11. Floating point

`f64` values may appear anywhere, but no family in version 0.1 admits `f64` state, because floating-point addition and multiplication are not associative and law L2 fails. Folds and scans with `f64` state always use `Seq`. Maps over `f64` are parallel, since they involve no reassociation.

A future version may add an explicit `reassoc` annotation that permits parallel plans for `f64` reductions. Such programs would fall outside the plan-independence guarantee, so the annotation would have to state an error bound and the compiler would have to document the exact summation order of each plan.

---

## 12. Toolchain

### 12.1 Commands

| Command | Purpose |
|---|---|
| `wrex run file.wrx` | Compile and run, selecting plans automatically |
| `wrex run --plan=seq file.wrx` | Force sequential plans everywhere |
| `wrex run --device=auto\|cpu\|metal\|cuda\|webgpu file.wrx` | Restrict or choose the execution device |
| `wrex calibrate` | Measure cost-model constants for the CPU and each available device |
| `wrex check file.wrx` | Type-check, infer representations, discharge obligations |
| `wrex explain file.wrx` | Report the representation and plan for each fold, scan, and tree fold |
| `wrex test file.wrx` | Run differential tests (Section 12.3) |

### 12.2 Explain output

```
balances.wrx:4  scan
  state       b : i64
  family      F1 affine over ℤ/2⁶⁴   (dimension 1, ⋄ cost 3)
  ρ(r, d)     (r, d)
  L3          discharged by ring normalization
  plan        ParScan when n ≥ 41 000 on 16 workers, else Seq   (illustrative thresholds)

expr.wrx:9  treefold Expr → i64
  contexts    F1 affine over ℤ/2⁶⁴
  plan        Contract when h > 4·log₂n, else ForkJoin
```

### 12.3 Conformance testing

Every conforming implementation must pass three kinds of tests:

1. **Semantic tests:** expected outputs and traps for programs under the reference interpreter.
2. **Differential tests:** each program runs under every applicable plan on randomized inputs, and all outputs and traps must match the reference interpreter exactly.
3. **Family law tests:** randomized checks of associativity, L1, and L2 for each family, including infinite bounds and wrapping edge cases. These complement, and do not replace, the mechanized proofs.

---

## 13. Compiler architecture

### 13.1 Pipeline

```
source → parse → type check → inline and defunctionalize lambdas
       → Core IR → representation inference → obligation checking
       → plan generation → fusion → backend
                                       ├─ CPU:  generated C + work-stealing runtime
                                       └─ GPU:  tinygrad tensor graphs
```

### 13.2 Core IR

Core IR is first-order, typed, and in A-normal form. Its distinctive nodes are:

```
Fold    { input, init, step, repr: Option<Certificate> }
Scan    { input, init, step, repr: Option<Certificate> }
TreeFold{ tree, arms, ctx_repr: Option<Certificate> }
Guard   { plans: [(Plan, CostExpr)] }
```

A `Certificate` records the family, dimension, compiled `ρ`, `⋄`, and `ev`, and the hash of the discharged obligation.

### 13.3 Trusted computing base

Correctness depends on:

- the family implementations of `⋄`, `ι`, and `ev`, and their laws,
- the ring and lattice normalizers used to check L3,
- the runtime's scan, fold, and contraction algorithms,
- the backends: the C compiler and work-stealing runtime for CPU plans, and tinygrad for GPU plans.

Representation inference, the cost model, and plan selection lie outside the trusted base. Errors in them affect performance only.

### 13.4 Backends

Version 0.1 has two backends that consume the same certificates:

- **CPU:** generated C with a work-stealing runtime. It implements every plan, including tree folds.
- **GPU:** tinygrad (Section 13.5). It implements `GpuMap`, `GpuFold`, and `GpuScan` for families F0–F6.

The representation families are chosen so that `⋄` and `ev` are fixed-size, branch-light arithmetic on small arrays. This is what makes them expressible as tensor operations.

**Implementation language.** The reference compiler is written in Python, so that it can construct tinygrad graphs directly. The CPU backend emits C, which is compiled with the system C compiler. The tinygrad version is pinned per Wrex release.

### 13.5 The tinygrad backend

The GPU backend lowers certified folds and scans to tinygrad tensor graphs. tinygrad then fuses, compiles, and runs them on the selected device. Wrex generates no device code of its own.

#### 13.5.1 Scope

| Construct | GPU support |
|---|---|
| `map`, `zip`, `range` | Yes |
| `fold`, `scan` with a certificate in F0–F6 | Yes |
| `fold`, `scan` without a certificate | No; runs `Seq` on the CPU |
| `treefold` | No; runs `ForkJoin` or `Contract` on the CPU |
| `f64` state | No (Section 11) |

#### 13.5.2 Data layout

Elements of a family are tuples of scalars. A sequence of `n` elements is stored as a **structure of arrays**: one tensor per scalar field, each with leading dimension `n`.

| Family | Tensors for `n` elements |
|---|---|
| F0 | one tensor of shape `(n,)` |
| F1 | `a`, `b`, each `(n,)` |
| F2 | one tensor of shape `(n, k+1, k+1)` |
| F3 | `a`, `l`, `h`, each `(n,)`, plus infinity flags (Section 13.5.4) |
| F4 | one integer tensor of shape `(n, |Q|)` |
| F5 | control tensor `(n, |Q|)` plus payload tensors with leading shape `(n, |Q|)` |
| F6 | the concatenation of its components' tensors |

Input arrays are uploaded once. Fused `map`s run on the device and are never materialized on the host.

#### 13.5.3 Lowering

- **`ρ`** depends only on the input, so it lowers to elementwise tensor operations over the input tensors. Branches on the input become `where` selections. For F4 and F5, the step is evaluated for every control state at once by broadcasting over a `|Q|` axis.
- **`⋄`** lowers per family:
  - F0, F1, F3: elementwise arithmetic on the fields, following the formulas in Section 6.
  - F2 over a ring: batched matrix multiplication, `M₂ · M₁`.
  - F2 tropical: broadcast addition followed by a `max` (or `min`) reduction over the inner index.
  - F4: a gather, `(r₁ ⋄ r₂)[q] = r₂[r₁[q]]`, taken along the `|Q|` axis.
  - F5: the F4 gather on the control tensor, then the payload family's `⋄` on gathered payloads.
- **`ev`** lowers to elementwise operations applied after the scan, producing the output states.

`⋄` is not commutative. Every lowering preserves operand order exactly; this is part of the conformance tests.

**Fold.** The input is padded to a power of two with `ι` and reduced by `⌈log₂n⌉` rounds. Each round reshapes the current tensors to pair adjacent elements and combines each pair with `⋄`, left operand first.

**Scan.** The backend uses a work-efficient inclusive up-sweep/down-sweep scan over the padded input, expressed as `⌈log₂n⌉` rounds in each direction using strided views and `⋄` (combining exclusive prefixes with `ρ(xᵢ)` to produce inclusive prefixes before applying `ev`). For small `n`, the backend may instead use a step-efficient inclusive scan (`⌈log₂n⌉` rounds of "shift by 2ᵈ and combine"), which does `O(n log n)` work but fewer, larger operations. The choice is a cost-model decision; both are exact.

The three-phase blocked scan of Section 8.1 is used only on the CPU, because its sequential replay phase does not map onto tensor operations.

#### 13.5.4 Exactness rules

The plan-independence guarantee (Section 5.5) applies to GPU plans without exception. The backend enforces it as follows:

- **Wrapping integers** are computed in the unsigned tensor type of the same width, and reinterpreted as signed only when results return to the host. This avoids relying on any backend's treatment of signed overflow.
- **Division and remainder** in `ρ` (and in generated C on the CPU backend) are lowered with explicit guards so that truncation toward zero and the `MIN / -1` and `MIN % -1` cases match Section 5.1 on every device.
- **`int` (mathematical integers)** has no GPU representation. A fold or scan over `int` runs on the GPU only if the planner proves, from the input length `n` and a static range bound or a cheap runtime bound on the input elements, that every intermediate value of `ρ`, `⋄`, and `ev` fits in a signed 64-bit integer. Otherwise it runs on the CPU.
- **Infinities** in F0, F2 (tropical), and F3 are represented by a boolean flag tensor alongside each value. Every addition and comparison is lowered with `where` so that `±∞ + a = ±∞` holds exactly.
- **Traps** are carried as an integer tensor holding the index of the earliest trap, with `n` meaning "no trap." For F0–F3, the trap tensor has shape `(n,)` and `⋄` combines trap tensors with elementwise `min`. For F4 and F5, the trap tensor has shape `(n, |Q|)` so that `⋄` propagates traps only along reached control states (`where(trap₁[q] < n, trap₁[q], trap₂[r₁[q]])`), and `ev` selects the trap entry for the initial control state `q₀`, matching Section 8.4. After the computation, the host checks the resulting trap index and raises the earliest trap.

#### 13.5.5 Device selection

The planner treats each available tinygrad device as a separate set of plans and adds the GPU cost formulas from Section 10.2. Host-device transfer is part of the cost, so small inputs, and folds whose result is immediately consumed by sequential CPU code, stay on the CPU.

#### 13.5.6 Device capabilities

| Device | Integer widths available to families | Notes |
|---|---|---|
| CPU via tinygrad (C or LLVM) | 8, 32, 64 | Useful for testing the backend without a GPU |
| Metal | 8, 32, 64 | — |
| CUDA | 8, 32, 64 | — |
| WebGPU | 8, 32 | WGSL has no native 64-bit integers; 64-bit families run on the CPU |

A family is offloaded only if every scalar type in its elements is available on the device.

#### 13.5.7 Trust and testing

tinygrad's code generator and runtime become part of the trusted computing base (Section 13.3) for GPU plans. To compensate, the conformance suite (Section 12.3) runs every differential test on every available device, and each Wrex release pins and tests a specific tinygrad version.

---

## 14. Examples

### 14.1 Account balances with interest (F1)

```
fn balances(txs: [(i64, i64)]) -> [i64]:
    scan(txs, 0, (b, (rate, deposit)) => b * rate + deposit)
```

### 14.2 Saturating inventory (F3)

```
fn stock_levels(deltas: [int], capacity: int) -> [int]:
    scan(deltas, 0, (s, d) => clamp(s + d, 0, capacity))
```

### 14.3 Word count with a state machine (F5 over F0)

```
type Mode = InWord | Gap

fn is_alpha(c: u8) -> bool:
    (c >= 65 and c <= 90) or (c >= 97 and c <= 122)

@par
fn word_count(text: [u8]) -> int:
    let (_, n) = fold(text, (Gap, 0), ((m, n), c) =>
        if is_alpha(c) then
            (InWord, if m == Gap then n + 1 else n)
        else
            (Gap, n))
    n
```

The control state `m` depends only on itself and the input. For each control state, the counter update is `n + w(m, c)` with `w ∈ {0, 1}`, which lies in F0 over `int`.

### 14.4 Two-term linear recurrence (F2 over a ring)

```
fn fib_mod(steps: [u64]) -> (u64, u64):
    fold(steps, (0, 1), ((a, b), k) => (b, a + k * b))
```

### 14.5 Longest run of positive values (tropical F2)

```
fn longest_positive_run(xs: [int]) -> int:
    let (_, best) = fold(xs, (0, 0), ((cur, best), x) =>
        let cur2 = if x > 0 then cur + 1 else 0
        (cur2, max(best, cur2)))
    best
```

Here `best` depends on `cur`, and `cur` depends on itself through both `+1` and reset. The pair forms a max-plus linear system over `int`:

```
cur'  = (x > 0) ? cur + 1 : 0
best' = max(best, cur')
```

Both components are max-plus affine in `(cur, best)` with input-dependent coefficients, so the step lies in tropical F2 with `k = 2`.

### 14.6 Not representable

```
fn logistic(rs: [int]) -> int:
    fold(rs, 1, (s, r) => r * s * (100 - s))
```

The step has degree 2 in `s`, so it runs sequentially, and `wrex explain` reports the obstruction from Section 7.5.

---

## 15. Related work

Wrex combines several established results into a single language mechanism:

- Ladner and Fischer (1980) and Blelloch (1990) on parallel prefix computation and on evaluating linear recurrences as scans.
- Gibbons (1996), the third homomorphism theorem, on when list functions are parallelizable homomorphisms.
- Miller and Reif (1985) on parallel tree contraction.
- Mytkowicz, Musuvathi, and Schulte (2014) on data-parallel execution of finite-state machines.
- Raychev, Musuvathi, and Mytkowicz (2015) on parallelizing user-defined aggregations by symbolic execution.
- The Krohn–Rhodes theory of finite automata, for wreath-product decompositions such as F5.
- Scheduling languages such as Halide and data-parallel functional languages such as Futhark, which separate algorithms from execution strategy.

The contribution Wrex aims to make is the combination: representation inference as the language's single source of parallelism, uniform across sequences and trees, with decidable correctness obligations, exact trap semantics, and a cost model derived from representation size.

---

## 16. Open questions

1. **Inference completeness.** For which classes of step functions is the search in Section 7 complete? A characterization for F1–F3 relative to the normal forms should be achievable.
2. **Representation size versus benefit.** F4 always applies to finite state, but tables of 256 entries may never pay off. Should the compiler stop searching above a cost threshold?
3. **Nested parallelism.** How should a parallel scan whose step contains its own fold be scheduled? A flattening transformation is likely needed.
4. **User-defined families.** Could users supply new families together with mechanized proofs of their laws, without enlarging the trusted base?
5. **Floating point.** What should a `reassoc` annotation promise, and how should its error bound be expressed and checked?
6. **GPU contraction.** Tree contraction relies on pointer-based random mating. Could a level-synchronous variant over node arrays be expressed as tinygrad gathers and still be competitive?
7. **Kernel fusion.** How many kernels does tinygrad produce for the up-sweep/down-sweep scan of each family, and would a single custom scan kernel per family be worth the added trusted code?

---

## Appendix A. Implementation phases

| Phase | Deliverable | Exit criterion |
|---|---|---|
| 1 | Parser, type checker, reference interpreter | All semantic tests pass |
| 2 | Ring normalizer, F0 and F1 inference, L3 checking | Obstruction reports for every failing example |
| 3 | Three-phase blocked scan on a work-stealing CPU runtime | Differential tests pass; speedup over sequential on 16 cores |
| 4 | F3, F4, F5, and trap-precise propagation | Word count and tokenizer benchmarks parallelize |
| 5 | Tree folds: contexts, contraction, fork-join, height-based guard | List-shaped expression trees evaluate in logarithmic depth |
| 6 | Tropical F2 and the calibrated cost model | Plan selection within 10% of the best plan on the benchmark suite |
| 7 | tinygrad backend for F0, F1, and F3 on CPU-via-tinygrad and Metal | Differential tests pass on every available device |
| 8 | tinygrad backend for F2, F4, F5, F6, plus CUDA and WebGPU | GPU speedup over the multicore CPU plan on large inputs |

## Appendix B. Benchmark suite

| Benchmark | Family | Baseline |
|---|---|---|
| Prefix sum | F0 | Hand-tuned parallel scan |
| Linear recurrence | F1, F2 | Hand-written blocked scan |
| Tokenizer with counters | F5 | Sequential state machine; hand-parallelized FSM |
| Saturating counters | F3 | Sequential loop |
| Expression-tree evaluation, balanced and list-shaped | contexts over F1 | Sequential recursion; fork-join recursion |
| Viterbi decoding | tropical F2 | Sequential dynamic programming |

Baselines must be competent, idiomatic implementations, compiled with full optimization.
