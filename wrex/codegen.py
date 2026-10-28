"""C backend (spec 13.4): emit standalone C for a certified fold/scan.

The generated program implements the three-phase blocked scan / balanced
fold of spec Section 8, using the same rho/compose/ev of the certificate.
Each worker runs LANES sub-blocks interleaved, so the composition law breaks
the step's dependency chain even on one core. Workers are threads only when
n is large enough to pay for them (GRAIN elements each). rho is applied as
elements are consumed, never stored. Traps are detected after the fast pass,
and any trap reruns the steps in order to report the first one exactly. Protocol on stdin: free scalar parameters first (captured
function variables), then n, then n * (input fields) integers, then the
initial state. Finite states (F4, and F5 control) are exchanged as dense
indices into the enumerated state set (spec 6.5). Output: the fold result,
or the scan states, whitespace separated.

All families F0-F6 are supported for folds; scans support every family
whose ev(rho(x), s) reproduces the step, which the certificate guarantees.
C `int64_t` stands in for Wrex `int`, so exactness holds within int64
range (the same bound the GPU backend uses, spec 13.5.4).
"""
from . import nodes as N
from .inference import pattern_names

class CodegenError(Exception): pass

# Wrex user function name -> generated C function name, for coefficient calls
FN_C = {}
# Family name -> FamilyDecl, populated by generate_c for FU codegen
FAM_C = {}

def _cexpr(e, env, ctor_map=None):
    if isinstance(e, N.Lit):
        if isinstance(e.value, bool): return "1" if e.value else "0"
        if isinstance(e.value, float): raise CodegenError("float coefficient not supported in C backend")
        return f"({e.value}LL)"
    if isinstance(e, N.Var):
        if e.name in env: return env[e.name]
        raise CodegenError(f"coefficient references unsupported name {e.name!r}")
    if isinstance(e, N.CtorExpr):
        if ctor_map and e.name in ctor_map and not e.args: return str(ctor_map[e.name])
        raise CodegenError(f"unsupported constructor {e.name} in generated C")
    if isinstance(e, N.Unary):
        return f"(!{_cexpr(e.e, env, ctor_map)})" if e.op == 'not' else f"(-{_cexpr(e.e, env, ctor_map)})"
    if isinstance(e, N.If):
        return f"({_cexpr(e.cond, env, ctor_map)} ? {_cexpr(e.then, env, ctor_map)} : {_cexpr(e.els, env, ctor_map)})"
    if isinstance(e, N.BinOp):
        l, r = _cexpr(e.l, env, ctor_map), _cexpr(e.r, env, ctor_map)
        if e.op in ('/', '%'):
            return f"{'wrex_sdiv' if e.op=='/' else 'wrex_smod'}({l}, {r})"
        op = {'and': '&&', 'or': '||'}.get(e.op, e.op)
        return f"({l} {op} {r})"
    if isinstance(e, N.Call) and isinstance(e.f, N.Var) and e.f.name in ('min', 'max', 'clamp'):
        a = [_cexpr(x, env, ctor_map) for x in e.args]
        if e.f.name == 'clamp': return f"wrex_clamp({a[0]}, {a[1]}, {a[2]})"
        return f"(({a[0]} {'<' if e.f.name=='min' else '>'} {a[1]}) ? {a[0]} : {a[1]})"
    if isinstance(e, N.Call) and isinstance(e.f, N.Var) and e.f.name in FN_C:
        a = [_cexpr(x, env, ctor_map) for x in e.args]
        return f"{FN_C[e.f.name]}({', '.join(a)})"
    raise CodegenError(f"unsupported coefficient expression {type(e).__name__}")

def _csym(s, env, varmap=None, ctor_map=None):
    k = s[0]
    if k == 'var':
        if varmap and s[1] in varmap: return varmap[s[1]]
        raise CodegenError("symbolic state variable in coefficient position")
    if k == 'leaf': return _cexpr(s[1], env, ctor_map)
    if k == 'nleaf':
        if s[1] in env: return env[s[1]]
        raise CodegenError(f"unsupported named coefficient {s[1]}")
    if k == 'inf': return "WREX_PINF" if s[1] > 0 else "WREX_NINF"
    if k == 'bin':
        if s[1] in ('/', '%'): return f"{'wrex_sdiv' if s[1]=='/' else 'wrex_smod'}({_csym(s[2], env, varmap, ctor_map)}, {_csym(s[3], env, varmap, ctor_map)})"
        op = {'and': '&&', 'or': '||'}.get(s[1], s[1])
        return f"({_csym(s[2], env, varmap, ctor_map)} {op} {_csym(s[3], env, varmap, ctor_map)})"
    if k == 'un': return f"(!{_csym(s[2], env, varmap, ctor_map)})" if s[1] == 'not' else f"(-{_csym(s[2], env, varmap, ctor_map)})"
    if k == 'if': return f"({_csym(s[1], env, varmap, ctor_map)} ? {_csym(s[2], env, varmap, ctor_map)} : {_csym(s[3], env, varmap, ctor_map)})"
    if k == 'call':
        a = [_csym(x, env, varmap, ctor_map) for x in s[2]]
        if s[1] == 'clamp': return f"wrex_clamp({a[0]}, {a[1]}, {a[2]})"
        return f"(({a[0]} {'<' if s[1]=='min' else '>'} {a[1]}) ? {a[0]} : {a[1]})"
    raise CodegenError(f"unsupported symbolic term {k}")

HELPERS = r"""
typedef long long i64;
#define WREX_PINF ((i64)4000000000000000000LL)
#define WREX_NINF ((i64)-4000000000000000000LL)
static int wrex_trap_capture; _Thread_local static int wrex_trap_flag;
static const char* wrex_trap_name[] = {"step_trap", "div_by_zero", "clamp_bounds", "step_trap"};
static i64 wrex_sdiv(i64 a, i64 b){ if(b==0){ if(wrex_trap_capture){ wrex_trap_flag=1; return 0; } fprintf(stderr,"trap: div_by_zero\n"); exit(3);} if(a==INT64_MIN && b==-1) return INT64_MIN; return a/b; }
static i64 wrex_smod(i64 a, i64 b){ if(b==0){ if(wrex_trap_capture){ wrex_trap_flag=1; return 0; } fprintf(stderr,"trap: div_by_zero\n"); exit(3);} if(a==INT64_MIN && b==-1) return 0; return a%b; }
static i64 wrex_clamp(i64 v, i64 lo, i64 hi){ if(lo>hi){ if(wrex_trap_capture){ wrex_trap_flag=2; return lo; } fprintf(stderr,"trap: clamp_bounds\n"); exit(3);} return v<lo?lo:(v>hi?hi:v); }
static inline i64 wrex_addinf(i64 a, i64 b){ i64 r = (i64)((unsigned long long)a + (unsigned long long)b); r = r > WREX_PINF ? WREX_PINF : r; r = r < WREX_NINF ? WREX_NINF : r; r = (a <= WREX_NINF || b <= WREX_NINF) ? WREX_NINF : r; return (a >= WREX_PINF || b >= WREX_PINF) ? WREX_PINF : r; }
"""

def _is_semiring_zero(sym, semiring):
    """True when a coefficient is statically the zero of its semiring:
    0 for the ring, -inf for max-plus, +inf for min-plus."""
    k = sym[0]
    if semiring == 'ring':
        return k == 'leaf' and isinstance(sym[1], N.Lit) and sym[1].value == 0 and not isinstance(sym[1].value, bool)
    sign = -1 if semiring == 'maxplus' else 1
    if k == 'inf': return sym[1] == sign
    if k == 'call' and sym[1] in ('max', 'min'): return all(_is_semiring_zero(x, semiring) for x in sym[2])
    return False

def _bound(sym):
    """What a coefficient can evaluate to: 'fin', '+inf', '-inf', or None
    when it cannot be decided statically."""
    k = sym[0]
    if k == 'inf': return '+inf' if sym[1] > 0 else '-inf'
    if k in ('leaf', 'nleaf', 'var'): return 'fin'
    if k == 'call' and sym[1] in ('max', 'min'):
        bs = [_bound(x) for x in sym[2]]
        absorb, drop = ('+inf', '-inf') if sym[1] == 'max' else ('-inf', '+inf')
        if absorb in bs: return absorb
        if None in bs: return None
        return 'fin' if 'fin' in bs else drop
    if k == 'call' and sym[1] == 'clamp':
        return 'fin' if all(_bound(x) == 'fin' for x in sym[2]) else None
    if k in ('bin', 'un', 'if'):
        subs = [x for x in sym[1:] if isinstance(x, tuple)]
        return 'fin' if all(_bound(x) == 'fin' for x in subs) else None
    return None

def _addinf(prefix, coeffs, lines):
    """Name of the tropical add for one element type. Wrex values are never
    infinite: the +-inf sentinels enter only through the identity and through
    coefficients. When every coefficient is a finite expression of the input,
    each sum the composition forms is clamped or maximized against a finite
    value, so a plain add gives the same result within the int64 range."""
    if any(_bound(c) != 'fin' for c in coeffs): return "wrex_addinf"
    lines.append(f"static inline i64 add_{prefix}(i64 a, i64 b){{ return (i64)((unsigned long long)a + (unsigned long long)b); }}")
    return f"add_{prefix}"

# Narrow storage for input fields: a [u8] input is stored as bytes, not int64.
C_STORAGE = {'bool': ('uint8_t', 1), 'u8': ('uint8_t', 1), 'i8': ('int8_t', 1), 'u16': ('uint16_t', 2),
             'i16': ('int16_t', 2), 'u32': ('uint32_t', 4), 'i32': ('int32_t', 4)}

def _input_storage(ty):
    """One C type wide enough for every scalar field of the input type."""
    fields = ty[1] if isinstance(ty, tuple) and ty and ty[0] == 'tuple' else [ty]
    if not all(isinstance(f, str) and f in C_STORAGE for f in fields): return "i64"
    signed = any(C_STORAGE[f][0].startswith('int') for f in fields)
    width = max(C_STORAGE[f][1] for f in fields)
    if signed and any(C_STORAGE[f][0] == f"uint{width * 8}_t" for f in fields): width *= 2
    if width > 4: return "i64"
    return f"{'int' if signed else 'uint'}{width * 8}_t"

CHEAP_F0 = ('+', '&', '|', '^', 'min', 'max', 'and', 'or')

def _lanes(cert):
    """Sub-blocks one worker interleaves. Interleaving pays when the step's
    dependency chain is slower than composing; it stops paying once the lanes'
    elements no longer fit in registers. Measured on the Appendix B loops."""
    k = cert.kind
    if k == 'F0' and cert.family.op in CHEAP_F0: return 1  # compose is the step
    if k in ('F4', 'F5'): return 1  # composing tables costs far more than one step
    size = {'F0': 1, 'FU': 1, 'F1': 2, 'F3': 3}.get(k)
    if k == 'F2': size = cert.family.k ** 2
    if k == 'F6': return 4
    if size is None: return 1
    return 8 if size <= 3 else 4 if size <= 8 else 2

def _collect_coeff_syms(cert, acc):
    for c in getattr(cert, 'coeffs', []) or []: acc.append(c)
    for c in getattr(cert, 'ctrl_syms', []) or []: acc.append(c)
    for sub in getattr(cert, 'subs', []) or []: _collect_coeff_syms(sub, acc)
    if getattr(cert, 'payload', None) is not None: _collect_coeff_syms(cert.payload, acc)

def _gen(cert, prefix, env, lines, fns, ctor_map=None):
    """Emit element type/functions for a certificate. Returns (type, state words)."""
    if cert.kind == 'FU':
        # emit the declared compose/identity as C; ev(r, s) = op(s, r)
        d = FAM_C.get(cert.family.fam_name)
        if d is None: raise CodegenError(f"C backend: family '{cert.family.fam_name}' declaration not found")
        if not all(isinstance(p, N.PVar) for p in d.compose.params):
            raise CodegenError(f"C backend: family '{d.name}' compose parameters must be simple names")
        opf = f"fam_{prefix}_op"
        senv = {d.compose.params[0].name: "a", d.compose.params[1].name: "b"}
        stmts = []
        body = d.compose.body
        if isinstance(body, N.Block):
            for pat, e in body.bindings:
                if not isinstance(pat, N.PVar): raise CodegenError(f"C backend: family '{d.name}' compose let pattern must be a simple name")
                stmts.append(f"i64 {pat.name} = {_cexpr(e, senv, ctor_map)};"); senv[pat.name] = pat.name
            stmts.append(f"return {_cexpr(body.expr, senv, ctor_map)};")
        else:
            stmts.append(f"return {_cexpr(body, senv, ctor_map)};")
        ident = _cexpr(d.identity, {}, ctor_map)
        g = _csym(cert.coeffs[0], env, ctor_map=ctor_map)
        lines += [f"static i64 {opf}(i64 a, i64 b){{ " + " ".join(stmts) + " }",
                  f"typedef i64 Elem_{prefix};",
                  f"static Elem_{prefix} elem_{prefix}_ident(void){{ return {ident}; }}",
                  f"static Elem_{prefix} elem_{prefix}_compose(Elem_{prefix} x, Elem_{prefix} y){{ return {opf}(x, y); }}",
                  f"static void elem_{prefix}_ev(Elem_{prefix} r, const i64* s, i64* out){{ out[0] = {opf}(s[0], r); }}",
                  f"static Elem_{prefix} elem_{prefix}_rho(const i64* in, i64 aux){{ (void)aux; return {g}; }}"]
        return f"Elem_{prefix}", 1
    T = f"Elem_{prefix}"; fam = cert.family; k = cert.kind
    idf, cof, evf, rhof = f"elem_{prefix}_ident", f"elem_{prefix}_compose", f"elem_{prefix}_ev", f"elem_{prefix}_rho"
    if k == 'F0':
        op = fam.op
        body = {'+': "a + b", '*': "a * b", 'min': "a<b?a:b", 'max': "a>b?a:b", '&': "a & b", '|': "a | b", '^': "a ^ b", 'and': "a && b", 'or': "a || b"}[op]
        ident = {'+': "0", '*': "1", 'min': "WREX_PINF", 'max': "WREX_NINF", '&': "-1", '|': "0", '^': "0", 'and': "1", 'or': "0"}[op]
        g = _csym(cert.coeffs[0], env, ctor_map=ctor_map)
        lines += [f"typedef i64 {T};",
                  f"static {T} {idf}(void){{ return {ident}; }}",
                  f"static {T} {cof}({T} a, {T} b){{ return {body}; }}",
                  f"static void {evf}({T} r, const i64* s, i64* out){{ out[0] = {cof}(s[0], r); }}",
                  f"static {T} {rhof}(const i64* in, i64 aux){{ (void)aux; return {g}; }}"]
        return T, 1
    if k == 'F1':
        a, b = _csym(cert.coeffs[0], env, ctor_map=ctor_map), _csym(cert.coeffs[1], env, ctor_map=ctor_map)
        lines += [f"typedef struct {{ i64 a, b; }} {T};",
                  f"static {T} {idf}(void){{ {T} e = {{1, 0}}; return e; }}",
                  f"static {T} {cof}({T} x, {T} y){{ {T} e = {{y.a * x.a, y.a * x.b + y.b}}; return e; }}",
                  f"static void {evf}({T} r, const i64* s, i64* out){{ out[0] = r.a * s[0] + r.b; }}",
                  f"static {T} {rhof}(const i64* in, i64 aux){{ (void)aux; {T} e = {{{a}, {b}}}; return e; }}"]
        return T, 1
    if k == 'F3':
        a, l, h = (_csym(c, env, ctor_map=ctor_map) for c in cert.coeffs)
        add = _addinf(prefix, cert.coeffs, lines)
        lines += [f"typedef struct {{ i64 a, l, h; }} {T};",
                  f"static {T} {idf}(void){{ {T} e = {{0, WREX_NINF, WREX_PINF}}; return e; }}",
                  f"static {T} {cof}({T} x, {T} y){{ i64 l = {add}(x.l, y.a); l = l<y.l?y.l:l; l = l>y.h?y.h:l; i64 h = {add}(x.h, y.a); h = h>y.h?y.h:h; h = h<y.l?y.l:h; {T} e = {{{add}(x.a, y.a), l, h}}; return e; }}",
                  f"static void {evf}({T} r, const i64* s, i64* out){{ i64 v = {add}(s[0], r.a); out[0] = v<r.l?r.l:(v>r.h?r.h:v); }}",
                  f"static {T} {rhof}(const i64* in, i64 aux){{ (void)aux; {T} e = {{{a}, {l}, {h}}}; return e; }}"]
        return T, 1
    if k == 'F2':
        kk = fam.k; n = kk + 1
        # The constant column carries the step's additive constants. When every
        # entry is the semiring zero, the map is linear and k x k suffices.
        linear = all(_is_semiring_zero(cert.coeffs[i * n + kk], fam.semiring) for i in range(kk))
        coeffs = [_csym(c, env, ctor_map=ctor_map) for c in cert.coeffs]
        used = list(cert.coeffs)
        if linear:
            n = kk
            coeffs = [coeffs[i * (kk + 1) + j] for i in range(kk) for j in range(kk)]
            used = [cert.coeffs[i * (kk + 1) + j] for i in range(kk) for j in range(kk)]
        add = _addinf(prefix, used, lines) if fam.semiring != 'ring' else "wrex_addinf"
        flat = ", ".join(coeffs)
        if fam.semiring == 'ring':
            combine = "__int128 acc=0; for(int t=0;t<N_;t++) acc += (__int128)y.m[i][t]*x.m[t][j]; e.m[i][j]=(i64)acc;"
            evcombine = "__int128 acc=0; for(int j=0;j<K_;j++) acc+=(__int128)r.m[i][j]*s[j]; acc += r.m[i][K_]; out[i]=(i64)acc;"
        elif fam.semiring == 'maxplus':
            combine = "i64 acc=WREX_NINF; for(int t=0;t<N_;t++){ i64 v=wrex_addinf(y.m[i][t], x.m[t][j]); acc = v>acc?v:acc; } e.m[i][j]=acc;"
            evcombine = "i64 acc=WREX_NINF; for(int j=0;j<K_;j++){ i64 v=wrex_addinf(r.m[i][j], s[j]); acc = v>acc?v:acc; } acc = r.m[i][K_] > acc ? r.m[i][K_] : acc; out[i]=acc;"
        else:
            combine = "i64 acc=WREX_PINF; for(int t=0;t<N_;t++){ i64 v=wrex_addinf(y.m[i][t], x.m[t][j]); acc = v<acc?v:acc; } e.m[i][j]=acc;"
            evcombine = "i64 acc=WREX_PINF; for(int j=0;j<K_;j++){ i64 v=wrex_addinf(r.m[i][j], s[j]); acc = v<acc?v:acc; } acc = r.m[i][K_] < acc ? r.m[i][K_] : acc; out[i]=acc;"
        if linear:
            evcombine = {'ring': "__int128 acc=0; for(int j=0;j<K_;j++) acc+=(__int128)r.m[i][j]*s[j]; out[i]=(i64)acc;",
                         'maxplus': "i64 acc=WREX_NINF; for(int j=0;j<K_;j++){ i64 v=wrex_addinf(r.m[i][j], s[j]); acc = v>acc?v:acc; } out[i]=acc;",
                         'minplus': "i64 acc=WREX_PINF; for(int j=0;j<K_;j++){ i64 v=wrex_addinf(r.m[i][j], s[j]); acc = v<acc?v:acc; } out[i]=acc;"}[fam.semiring]
        combine = combine.replace("N_", str(n)).replace("wrex_addinf(", add + "("); evcombine = evcombine.replace("K_", str(kk)).replace("wrex_addinf(", add + "(")
        lines += [f"#define K_{prefix} {kk}", f"#define N_{prefix} {n}", f"typedef struct {{ i64 m[{n}][{n}]; }} {T};",
                  f"static {T} {idf}(void){{ {T} e; for(int i=0;i<{n};i++) for(int j=0;j<{n};j++) e.m[i][j] = (i==j) ? {1 if fam.semiring=='ring' else 0} : {'0' if fam.semiring=='ring' else ('WREX_NINF' if fam.semiring=='maxplus' else 'WREX_PINF')}; return e; }}",
                  f"static {T} {cof}({T} x, {T} y){{ {T} e; for(int i=0;i<{n};i++) for(int j=0;j<{n};j++){{ {combine} }} return e; }}",
                  f"static void {evf}({T} r, const i64* s, i64* out){{ for(int i=0;i<{kk};i++){{ {evcombine} }} }}",
                  f"static {T} {rhof}(const i64* in, i64 aux){{ (void)aux; i64 flat[{n*n}] = {{{flat}}}; {T} e; for(int i=0;i<{n};i++) for(int j=0;j<{n};j++) e.m[i][j]=flat[i*{n}+j]; return e; }}"]
        return T, kk
    if k == 'F4':
        q = len(fam.q)
        ctor_map = {}
        for v, idx in fam.index.items():
            from .values import ADT
            if isinstance(v, ADT): ctor_map[v.ctor] = idx
        step_fn = _gen_step_fn(cert, prefix, env, lines, fns, ctor_map)
        lines += [f"typedef struct {{ int t[{q}]; }} {T};",
                  f"static {T} {idf}(void){{ {T} e; for(int i=0;i<{q};i++) e.t[i]=i; return e; }}",
                  f"static {T} {cof}({T} x, {T} y){{ {T} e; for(int i=0;i<{q};i++) e.t[i] = x.t[i]<0 ? -1 : y.t[x.t[i]]; return e; }}",
                  f"static void {evf}({T} r, const i64* s, i64* out){{ int v = r.t[s[0]]; if(v<0){{ if(wrex_trap_capture){{ wrex_trap_flag=3; out[0]=s[0]; return; }} fprintf(stderr,\"trap: step_trap\\n\"); exit(3); }} out[0] = v; }}",
                  f"static {T} {rhof}(const i64* in, i64 aux){{ (void)aux; {T} e; int saved = wrex_trap_flag, cap = wrex_trap_capture; wrex_trap_capture=1; for(int i=0;i<{q};i++){{ wrex_trap_flag=0; i64 v = {step_fn}(i, in); e.t[i] = wrex_trap_flag ? -1 : (int)v; }} wrex_trap_flag=saved; wrex_trap_capture=cap; return e; }}"]
        return T, 1
    if k == 'F5':
        q = len(cert.qvalues)
        from .values import ADT
        nctrl = len(cert.ctrl_scope)
        # control values: scalar for one component, tuples for several; each
        # component gets a dense encoding over its distinct values
        def comps(v): return list(v) if isinstance(v, tuple) else [v]
        comp_values = []
        for j in range(nctrl):
            seen = []
            for v in cert.qvalues:
                cv = comps(v)[j]
                if cv not in seen: seen.append(cv)
            comp_values.append(seen)
        def enc(j, v): return comp_values[j].index(v)
        dec_table = [[enc(j, comps(v)[j]) for j in range(nctrl)] for v in cert.qvalues]
        lines.append(f"static const i64 ctrl_dec_{prefix}[{q}][{nctrl}] = {{{', '.join('{'+', '.join(str(x) for x in row)+'}' for row in dec_table)}}};")
        # encode(control components...) -> combined index, by table search
        lines.append(f"static int ctrl_enc_{prefix}({', '.join('i64 c'+str(j) for j in range(nctrl))}){{ "
                     + " ".join(f"for(int i=0;i<{q};i++){{ if({ ' && '.join(f'ctrl_dec_{prefix}[i][{j}]==c{j}' for j in range(nctrl)) }) return i; }}" for _ in [0])
                     + " return 0; }")
        ctrl_maps = []
        for j in range(nctrl):
            cm = {}
            for v in comp_values[j]:
                if isinstance(v, ADT): cm[v.ctor] = enc(j, v)
            ctrl_maps.append(cm)
        merged_map = {}
        for cm in ctrl_maps: merged_map.update(cm)
        env_pay = dict(env)
        for j, name in enumerate(cert.ctrl_names):
            env_pay[name] = f"ctrl_dec_{prefix}[aux][{j}]" if nctrl > 1 else "aux"
        # with a single component the generated payload sees aux = component encoding (= combined index when scalar)
        PT, pwords = _gen(cert.payload, prefix + "_pay", env_pay, lines, fns, ctor_map=merged_map)
        pidf, pcof, pevf, prhof = f"elem_{prefix}_pay_ident", f"elem_{prefix}_pay_compose", f"elem_{prefix}_pay_ev", f"elem_{prefix}_pay_rho"
        varmap = {cert.ctrl_scope[j]: (f"ctrl_dec_{prefix}[q][{j}]" if nctrl > 1 else "q") for j in range(nctrl)}
        ctrl_exprs = [_csym(cert.ctrl_syms[j], env, varmap=varmap, ctor_map=ctrl_maps[j]) for j in range(nctrl)]
        if nctrl == 1:
            ctrl_next = f"(int)({ctrl_exprs[0]})"
        else:
            ctrl_next = f"ctrl_enc_{prefix}({', '.join('('+e+')' for e in ctrl_exprs)})"
        lines += [f"typedef struct {{ int nxt[{q}]; {PT} pay[{q}]; }} {T};",
                  f"static {T} {idf}(void){{ {T} e; for(int i=0;i<{q};i++){{ e.nxt[i]=i; e.pay[i]={pidf}(); }} return e; }}",
                  f"static {T} {cof}({T} x, {T} y){{ {T} e; for(int i=0;i<{q};i++){{ int q1 = x.nxt[i]; e.nxt[i] = y.nxt[q1]; e.pay[i] = {pcof}(x.pay[i], y.pay[q1]); }} return e; }}",
                  f"static void {evf}({T} r, const i64* s, i64* out){{ int q = (int)s[0]; out[0] = r.nxt[q]; {pevf}(r.pay[q], s+1, out+1); }}",
                  f"static {T} {rhof}(const i64* in, i64 aux){{ (void)aux; {T} e; for(int q=0;q<{q};q++){{ e.nxt[q] = {ctrl_next}; e.pay[q] = {prhof}(in, q); }} return e; }}",
                  # ev(rho(x), s) needs only the row for the current state s[0]
                  f"static void elem_{prefix}_step(const i64* in, const i64* s, i64* out){{ int q = (int)s[0]; out[0] = {ctrl_next}; {pevf}({prhof}(in, q), s+1, out+1); }}"]
        return T, 1 + pwords
    if k == 'F6':
        subs = []
        for i, sub in enumerate(cert.subs):
            subs.append(_gen(sub, f"{prefix}_{i}", env, lines, fns, ctor_map=ctor_map))
        fields = " ".join(f"{t} e{i};" for i, (t, w) in enumerate(subs))
        idparts = ", ".join(f"elem_{prefix}_{i}_ident()" for i in range(len(subs)))
        cofparts = "".join(f"e.e{i} = elem_{prefix}_{i}_compose(x.e{i}, y.e{i}); " for i in range(len(subs)))
        rhoparts = "".join(f"e.e{i} = elem_{prefix}_{i}_rho(in, aux); " for i in range(len(subs)))
        evparts = ""
        off = 0
        for i, (t, w) in enumerate(subs):
            evparts += f"elem_{prefix}_{i}_ev(r.e{i}, s+{off}, out+{off}); "
            off += w
        lines += [f"typedef struct {{ {fields} }} {T};",
                  f"static {T} {idf}(void){{ {T} e = {{{idparts}}}; return e; }}",
                  f"static {T} {cof}({T} x, {T} y){{ {T} e; {cofparts}return e; }}",
                  f"static void {evf}({T} r, const i64* s, i64* out){{ {evparts} }}",
                  f"static {T} {rhof}(const i64* in, i64 aux){{ {T} e; {rhoparts}return e; }}"]
        return T, off
    raise CodegenError(f"C backend does not support family {k}")

def _gen_step_fn(cert, prefix, env, lines, fns, ctor_map):
    """Generate the original step as C over dense finite-state indices (F4)."""
    step = cert.step_node
    if isinstance(step, N.Var):
        if step.name not in fns: raise CodegenError("F4 C codegen: named step not found")
        d = fns[step.name]
        if len(d.params) != 2: raise CodegenError("F4 C codegen: bad step arity")
        sname, xname = d.params[0][0], d.params[1][0]
        body = d.body
        senv = {sname: "s", xname: "in[0]"}
    elif isinstance(step, N.Lambda):
        if len(step.params) != 2 or not isinstance(step.params[0], N.PVar):
            raise CodegenError("F4 C codegen needs a scalar state pattern")
        in_names = []
        pattern_names(step.params[1], in_names)
        senv = {step.params[0].name: "s"}
        for i, nm in enumerate(in_names): senv[nm] = f"in[{i}]"
        body = step.body
    else:
        raise CodegenError("F4 C codegen: unsupported step form")
    stmts = []
    def emit_block(blk):
        for pat, e in blk.bindings:
            if not isinstance(pat, N.PVar): raise CodegenError("F4 C codegen: tuple let pattern in step")
            stmts.append(f"i64 {pat.name} = {_cexpr(e, senv, ctor_map)};")
            senv[pat.name] = pat.name
        stmts.append(f"return {_cexpr(blk.expr, senv, ctor_map)};")
    if isinstance(body, N.Block): emit_block(body)
    else: stmts.append(f"return {_cexpr(body, senv, ctor_map)};")
    fn = f"step_{prefix}"
    lines.append(f"static i64 {fn}(i64 s, const i64* in){{ " + " ".join(stmts) + " }")
    return fn

def generate_c(prog, analysis, fn_name=None):
    rec = None
    for r in analysis.records:
        if "cert" in r and r["kind"] in ('fold', 'scan') and (fn_name is None or r["fn"] == fn_name):
            rec = r; break
    if rec is None: raise CodegenError("no certified fold/scan found to generate C for")
    cert = rec["cert"]
    fns = {d.name: d for d in prog.decls if isinstance(d, N.FnDecl)}
    in_names = []
    if cert.input_pat is not None: pattern_names(cert.input_pat, in_names)
    env = {name: f"in[{i}]" for i, name in enumerate(in_names)}
    free = []
    syms = []
    _collect_coeff_syms(cert, syms)
    def walk_ast(e):
        if isinstance(e, N.Var) and e.name not in env and e.name not in fns and e.name not in free and e.name and e.name[0].islower(): free.append(e.name)
        for v in vars(e).values():
            if isinstance(v, N.Node): walk_ast(v)
            elif isinstance(v, list):
                for y in v:
                    if isinstance(y, N.Node): walk_ast(y)
    def walk_sym(s):
        if s[0] == 'leaf': walk_ast(s[1]); return
        if s[0] == 'nleaf':
            if s[1] not in env and s[1] not in free: free.append(s[1])
            return
        for x in s[1:]:
            if isinstance(x, tuple): walk_sym(x)
            elif isinstance(x, list):
                for y in x: walk_sym(y)
    for s in syms: walk_sym(s)
    # control names of F5 are bound via aux, not free params
    def ctrl_names(c):
        out = list(getattr(c, 'ctrl_names', []) or [])
        if getattr(c, 'payload', None) is not None: out += ctrl_names(c.payload)
        for sub in getattr(c, 'subs', []) or []: out += ctrl_names(sub)
        return out
    bound = set(ctrl_names(cert))
    free = [f for f in free if f not in bound]
    for i, name in enumerate(free): env[name] = f"param[{i}]"
    nin = max(1, len(in_names))
    lines = ["#include <stdio.h>", "#include <stdlib.h>", "#include <stdint.h>", "#include <pthread.h>", "#include <time.h>", HELPERS]
    if free: lines.append(f"static i64 param[{len(free)}];")
    # family declarations for FU codegen; their compose bodies may call helpers
    FAM_C.clear()
    for d in prog.decls:
        if isinstance(d, N.FamilyDecl): FAM_C[d.name] = d
    # emit scalar user functions referenced by coefficients (e.g. is_alpha)
    FN_C.clear()
    called = []
    def find_calls(e):
        if isinstance(e, N.Call) and isinstance(e.f, N.Var) and e.f.name in fns and e.f.name not in called: called.append(e.f.name)
        for v in vars(e).values():
            if isinstance(v, N.Node): find_calls(v)
            elif isinstance(v, list):
                for y in v:
                    if isinstance(y, N.Node): find_calls(y)
    def find_calls_sym(s):
        if s[0] == 'leaf': find_calls(s[1]); return
        for x in s[1:]:
            if isinstance(x, tuple): find_calls_sym(x)
            elif isinstance(x, list):
                for y in x: find_calls_sym(y)
    for s in syms: find_calls_sym(s)
    for d in prog.decls:
        if isinstance(d, N.FamilyDecl): find_calls(d.compose)
    for name in called:
        d = fns[name]; FN_C[name] = f"wrex_fn_{name}"
    for name in called:
        d = fns[name]
        params = ", ".join(f"i64 {pn}" for pn, _ in d.params) or "void"
        fenv = {pn: pn for pn, _ in d.params}
        body = d.body
        if not isinstance(body, N.Block) or body.bindings: raise CodegenError(f"C backend: function {name} used in a coefficient must be a single expression")
        lines.append(f"static i64 wrex_fn_{name}({params}){{ return {_cexpr(body.expr, fenv)}; }}")
    T, words = _gen(cert, "w", env, lines, fns)
    is_scan = rec["kind"] == 'scan'
    if not any(l.startswith("static void elem_w_step(") for l in lines):
        lines.append("static void elem_w_step(const i64* in, const i64* s, i64* out){ elem_w_ev(elem_w_rho(in, 0), s, out); }")
    main = [f"#define NIN {nin}", f"#define WORDS {words}", f"typedef {T} Elem;",
            "#ifndef LANES", f"#define LANES {_lanes(cert)}", "#endif",
            "#ifndef GRAIN", "#define GRAIN 131072", "#endif",
            f"typedef {_input_storage(rec.get('input_ty'))} In;",
            "static In* inputs; static long n; static i64 s0[WORDS];",
            "static long nsub; static Elem* accs; static i64* starts; static i64* outbuf;",
            "/* widen element i's fields to i64 (free once inlined) */",
            "static inline Elem RHO(long i){ i64 in[NIN]; for(int k = 0; k < NIN; k++) in[k] = inputs[i * NIN + k]; return elem_w_rho(in, 0); }",
            "static inline void STEP(long i, const i64* s, i64* o){ i64 in[NIN]; for(int k = 0; k < NIN; k++) in[k] = inputs[i * NIN + k]; elem_w_step(in, s, o); }",
            "static long sub_lo(long k){ return (long)((__int128)k * n / nsub); }",
            "typedef struct { long k0; int trap; } Job;",
            "/* fold each of this worker's LANES sub-blocks to one element, interleaved */",
            "static void* summarize(void* p){ Job* j = (Job*)p; Elem acc[LANES]; long lo[LANES], len[LANES], m = -1;",
            "  for(int l = 0; l < LANES; l++){ acc[l] = elem_w_ident(); lo[l] = sub_lo(j->k0 + l); len[l] = sub_lo(j->k0 + l + 1) - lo[l]; if(m < 0 || len[l] < m) m = len[l]; }",
            "  for(long t = 0; t < m; t++) for(int l = 0; l < LANES; l++) acc[l] = elem_w_compose(acc[l], RHO(lo[l] + t));",
            "  for(int l = 0; l < LANES; l++){ for(long t = m; t < len[l]; t++) acc[l] = elem_w_compose(acc[l], RHO(lo[l] + t)); accs[j->k0 + l] = acc[l]; }",
            "  j->trap = wrex_trap_flag; wrex_trap_flag = 0; return 0; }",
            "/* run the steps of each sub-block from its start state, interleaved */",
            "static void* replay(void* p){ Job* j = (Job*)p; i64 st[LANES][WORDS]; long lo[LANES], len[LANES], m = -1;",
            "  for(int l = 0; l < LANES; l++){ lo[l] = sub_lo(j->k0 + l); len[l] = sub_lo(j->k0 + l + 1) - lo[l]; if(m < 0 || len[l] < m) m = len[l]; for(int w = 0; w < WORDS; w++) st[l][w] = starts[(j->k0 + l) * WORDS + w]; }",
            "  for(long t = 0; t < m; t++) for(int l = 0; l < LANES; l++){ long i = lo[l] + t; i64 o[WORDS]; STEP(i, st[l], o); for(int w = 0; w < WORDS; w++){ st[l][w] = o[w]; outbuf[i * WORDS + w] = o[w]; } }",
            "  for(int l = 0; l < LANES; l++) for(long t = m; t < len[l]; t++){ long i = lo[l] + t; i64 o[WORDS]; STEP(i, st[l], o); for(int w = 0; w < WORDS; w++){ st[l][w] = o[w]; outbuf[i * WORDS + w] = o[w]; } }",
            "  j->trap = wrex_trap_flag; wrex_trap_flag = 0; return 0; }",
            "/* worker 0 runs on the calling thread; returns nonzero if any worker trapped */",
            "static int run_workers(void* (*f)(void*), Job* jobs, long nw){ pthread_t th[nw > 1 ? nw - 1 : 1];",
            "  for(long b = 1; b < nw; b++) pthread_create(&th[b - 1], 0, f, &jobs[b]);",
            "  f(&jobs[0]); int trap = jobs[0].trap;",
            "  for(long b = 1; b < nw; b++){ pthread_join(th[b - 1], 0); trap |= jobs[b].trap; } return trap; }",
            "static i64 result[WORDS];",
            "/* the steps in order from s0; returns the kind of the first trap, or 0 */",
            "static int sequential(void){ i64 s[WORDS], o[WORDS]; for(int w = 0; w < WORDS; w++) s[w] = s0[w]; wrex_trap_flag = 0;",
            "  for(long i = 0; i < n; i++){ STEP(i, s, o); if(wrex_trap_flag) return wrex_trap_flag; for(int w = 0; w < WORDS; w++){ s[w] = o[w];" + (" outbuf[i * WORDS + w] = o[w];" if is_scan else "") + " } }",
            "  for(int w = 0; w < WORDS; w++) result[w] = s[w]; return wrex_trap_flag; }",
            "/* the whole computation; returns nonzero if any step trapped */",
            "static int compute(long p){",
            "  long nw = n / GRAIN; if(nw > p) nw = p; if(nw < 1) nw = 1;",
            "  nsub = nw * LANES; wrex_trap_flag = 0; wrex_trap_capture = 1;",
            "  i64 s[WORDS], o[WORDS]; for(int w = 0; w < WORDS; w++) s[w] = s0[w];"]
    if is_scan:
        main += ["  if(!outbuf) outbuf = malloc(sizeof(i64) * (n > 0 ? n : 1) * WORDS);"]
    # Vectorizing a narrow input against int64 state spends more on widening
    # than it saves (tokenizer: 18 us vectorized, 10 us unrolled, n = 50,000).
    narrow = _input_storage(rec.get('input_ty')) != "i64"
    main += ["  if(nsub == 1){ /* one worker and one lane: no summaries, just the steps */",
             ("    _Pragma(\"clang loop vectorize(disable) interleave_count(4)\")" if narrow else ""),
             "    for(long i = 0; i < n; i++){ STEP(i, s, o); for(int w = 0; w < WORDS; w++){ s[w] = o[w];" + (" outbuf[i * WORDS + w] = o[w];" if is_scan else "") + " } }",
             "    for(int w = 0; w < WORDS; w++) result[w] = s[w];",
             "    return wrex_trap_flag; }",
             "  static Job* jobs; static long cap; if(nsub > cap){ cap = nsub; accs = realloc(accs, sizeof(Elem) * cap); starts = realloc(starts, sizeof(i64) * cap * WORDS); jobs = realloc(jobs, sizeof(Job) * cap); }",
             "  for(long b = 0; b < nw; b++){ jobs[b].k0 = b * LANES; jobs[b].trap = 0; }",
             "  int trap = run_workers(summarize, jobs, nw);"]
    if not is_scan:
        main += ["  Elem total = elem_w_ident();",
                 "  for(long k = 0; k < nsub; k++) total = elem_w_compose(total, accs[k]);",
                 "  elem_w_ev(total, s, result); return trap | wrex_trap_flag; }"]
    else:
        main += ["  /* the start state of each sub-block is the prefix of summaries before it */",
                 "  Elem prefix = elem_w_ident();",
                 "  for(long k = 0; k < nsub; k++){ elem_w_ev(prefix, s, starts + k * WORDS); prefix = elem_w_compose(prefix, accs[k]); }",
                 "  return trap | wrex_trap_flag | run_workers(replay, jobs, nw); }"]
    main += ["int main(int argc, char** argv){",
             "  long p = argc > 1 ? atol(argv[1]) : 4; if(p < 1) p = 1;"]
    if free:
        main.append(f"  for(int i = 0; i < {len(free)}; i++){{ if(scanf(\"%lld\", &param[i]) != 1) return 1; }}")
    main += ["  if(scanf(\"%ld\", &n) != 1) return 1;",
             "  inputs = malloc(sizeof(In) * (n > 0 ? n : 1) * NIN);",
             "  for(long i = 0; i < n * NIN; i++){ i64 v; if(scanf(\"%lld\", &v) != 1) return 1; inputs[i] = (In)v; }",
             "  for(int i = 0; i < WORDS; i++){ if(scanf(\"%lld\", &s0[i]) != 1) return 1; }",
             "#ifdef WREX_BENCH",
             "  /* benchmark hook: print the fastest of WREX_BENCH runs, in ns, to stderr */",
             "  long long best = -1; int trap = 0;",
             "  for(int r = 0; r < WREX_BENCH && !trap; r++){ struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0); trap = compute(p); clock_gettime(CLOCK_MONOTONIC, &t1);",
             "    long long dt = (t1.tv_sec - t0.tv_sec) * 1000000000LL + (t1.tv_nsec - t0.tv_nsec); if(best < 0 || dt < best) best = dt; }",
             "  fprintf(stderr, \"%lld\\n\", best);",
             "#else",
             "  int trap = compute(p);",
             "#endif",
             "  /* any trap in the fast pass: redo the steps in order, which reports",
             "     the first trap exactly or, if the trap was in a state the program",
             "     never reaches, produces the result */",
             "  if(trap && (trap = sequential())){ fprintf(stderr, \"trap: %s\\n\", wrex_trap_name[trap]); return 3; }"]
    if is_scan:
        main += ["  for(long i = 0; i < n * WORDS; i++) printf(\"%s%lld\", i ? \" \" : \"\", outbuf[i]);"]
    else:
        main += ["  for(int i = 0; i < WORDS; i++) printf(\"%s%lld\", i ? \" \" : \"\", result[i]);"]
    main += ["  printf(\"\\n\"); return 0; }"]
    src = f"/* generated by wrex codegen: {rec['kind']} in fn {rec['fn']} ({cert.describe()}); state words = {words} */\n"
    return src + "\n".join(lines) + "\n" + "\n".join(main) + "\n"
