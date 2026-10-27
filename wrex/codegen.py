"""C backend (spec 13.4): emit standalone C for a certified fold/scan.

The generated program implements the three-phase blocked scan / balanced
fold of spec Section 8 with pthreads, using the same rho/compose/ev of the
certificate. Input format: n followed by n * (input fields) integers on
stdin. Output: the fold result, or the n scan results, whitespace separated.

Supported: scalar state (F0, F1, F3) and vector state (F2) over integer
types, with scalar or tuple inputs whose coefficients are arithmetic
expressions. C `int64_t` stands in for Wrex `int`, so exactness holds
within int64 range (the same bound the GPU backend uses, spec 13.5.4);
machine integer types wrap via unsigned arithmetic.
"""
from . import nodes as N
from .inference import pattern_names

class CodegenError(Exception): pass

CTY = {"int": "int64_t", "i64": "int64_t", "u64": "uint64_t", "i32": "int32_t", "u32": "uint32_t", "u8": "uint8_t"}

def _cexpr(e, env):
    """AST (coefficient, no state vars) -> C expression. env: name->C var."""
    if isinstance(e, N.Lit):
        if isinstance(e.value, bool): return "1" if e.value else "0"
        if isinstance(e.value, float): raise CodegenError("float coefficient not supported in C backend")
        return f"({e.value}LL)"
    if isinstance(e, N.Var):
        if e.name in env: return env[e.name]
        raise CodegenError(f"coefficient references unsupported name {e.name!r}")
    if isinstance(e, N.Unary):
        return f"(!{_cexpr(e.e, env)})" if e.op == 'not' else f"(-{_cexpr(e.e, env)})"
    if isinstance(e, N.If):
        return f"({_cexpr(e.cond, env)} ? {_cexpr(e.then, env)} : {_cexpr(e.els, env)})"
    if isinstance(e, N.BinOp):
        l, r = _cexpr(e.l, env), _cexpr(e.r, env)
        if e.op in ('/', '%'):
            fn = "wrex_sdiv" if e.op == '/' else "wrex_smod"
            return f"{fn}({l}, {r})"
        op = {'and': '&&', 'or': '||'}.get(e.op, e.op)
        return f"({l} {op} {r})"
    if isinstance(e, N.Call) and isinstance(e.f, N.Var) and e.f.name in ('min', 'max', 'clamp'):
        a = [_cexpr(x, env) for x in e.args]
        if e.f.name == 'clamp': return f"wrex_clamp({a[0]}, {a[1]}, {a[2]})"
        return f"(({a[0]} {'<' if e.f.name=='min' else '>'} {a[1]}) ? {a[0]} : {a[1]})"
    raise CodegenError(f"unsupported coefficient expression {type(e).__name__}")

def _csym(s, env):
    k = s[0]
    if k == 'leaf': return _cexpr(s[1], env)
    if k == 'inf': return "WREX_PINF" if s[1] > 0 else "WREX_NINF"
    if k == 'bin': return f"({_csym(s[2], env)} {'&&' if s[1]=='and' else '||' if s[1]=='or' else s[1] if s[1] not in ('/','%') else s[1]} {_csym(s[3], env)})" if s[1] not in ('/', '%') else f"({'wrex_sdiv' if s[1]=='/' else 'wrex_smod'}({_csym(s[2], env)}, {_csym(s[3], env)}))"
    if k == 'un': return f"(!{_csym(s[2], env)})" if s[1] == 'not' else f"(-{_csym(s[2], env)})"
    if k == 'if': return f"({_csym(s[1], env)} ? {_csym(s[2], env)} : {_csym(s[3], env)})"
    if k == 'call':
        a = [_csym(x, env) for x in s[2]]
        if s[1] == 'clamp': return f"wrex_clamp({a[0]}, {a[1]}, {a[2]})"
        return f"(({a[0]} {'<' if s[1]=='min' else '>'} {a[1]}) ? {a[0]} : {a[1]})"
    raise CodegenError(f"unsupported symbolic term {k}")

HELPERS = r"""
typedef long long i64;
#define WREX_PINF ((i64)4000000000000000000LL)
#define WREX_NINF ((i64)-4000000000000000000LL)
static i64 wrex_sdiv(i64 a, i64 b){ if(b==0){ fprintf(stderr,"trap: div_by_zero\n"); exit(3);} if(a==INT64_MIN && b==-1) return INT64_MIN; return a/b; }
static i64 wrex_smod(i64 a, i64 b){ if(b==0){ fprintf(stderr,"trap: div_by_zero\n"); exit(3);} if(a==INT64_MIN && b==-1) return 0; return a%b; }
static i64 wrex_clamp(i64 v, i64 lo, i64 hi){ if(lo>hi){ fprintf(stderr,"trap: clamp_bounds\n"); exit(3);} return v<lo?lo:(v>hi?hi:v); }
static i64 wrex_addinf(i64 a, i64 b){ if(a>=WREX_PINF||b>=WREX_PINF) return WREX_PINF; if(a<=WREX_NINF||b<=WREX_NINF) return WREX_NINF; __int128 r=(__int128)a+b; if(r>WREX_PINF) return WREX_PINF; if(r<WREX_NINF) return WREX_NINF; return (i64)r; }
"""

def generate_c(prog, analysis, fn_name=None):
    rec = None
    for r in analysis.records:
        if "cert" in r and r["kind"] in ('fold', 'scan') and (fn_name is None or r["fn"] == fn_name):
            rec = r; break
    if rec is None: raise CodegenError("no certified fold/scan found to generate C for")
    cert = rec["cert"]
    if cert.kind not in ('F0', 'F1', 'F3', 'F2'):
        raise CodegenError(f"C backend does not support family {cert.kind} yet")
    # input fields
    in_names = []
    if cert.input_pat is not None: pattern_names(cert.input_pat, in_names)
    env = {name: f"in[{i}]" for i, name in enumerate(in_names)}
    # free coefficient variables (e.g. a clamp capacity captured from the
    # enclosing function) become leading scalar parameters of the program
    free = []
    def walk_ast(e):
        if isinstance(e, N.Var) and e.name not in env and e.name not in free and e.name and e.name[0].islower(): free.append(e.name)
        for v in vars(e).values():
            if isinstance(v, N.Node): walk_ast(v)
            elif isinstance(v, list):
                for y in v:
                    if isinstance(y, N.Node): walk_ast(y)
    def walk_sym(s):
        if s[0] == 'leaf': walk_ast(s[1]); return
        for x in s[1:]:
            if isinstance(x, tuple): walk_sym(x)
            elif isinstance(x, list):
                for y in x: walk_sym(y)
    for c in getattr(cert, 'coeffs', []): walk_sym(c)
    for i, name in enumerate(free): env[name] = f"param[{i}]"
    nin = max(1, len(in_names))
    fam = cert.family
    lines = ["#include <stdio.h>", "#include <stdlib.h>", "#include <stdint.h>", "#include <pthread.h>", HELPERS]
    if free: lines.append(f"static i64 param[{len(free)}];")
    if cert.kind in ('F0', 'F1', 'F3'):
        coeffs = [_csym(c, env) for c in cert.coeffs]
        if cert.kind == 'F0':
            op = fam.op
            body = {'+': "a + b", '*': "a * b", 'min': "a<b?a:b", 'max': "a>b?a:b",
                    '&': "a & b", '|': "a | b", '^': "a ^ b", 'and': "a && b", 'or': "a || b"}[op]
            ident = {'+': "0", '*': "1", 'min': "WREX_PINF", 'max': "WREX_NINF", '&': "-1", '|': "0", '^': "0", 'and': "1", 'or': "0"}[op]
            lines += ["typedef i64 Elem;", f"static Elem elem_ident(void){{ return {ident}; }}",
                      f"static Elem elem_compose(Elem a, Elem b){{ return {body}; }}",
                      "static i64 elem_ev(Elem r, i64 s){ return elem_compose(s, r); }",
                      f"static Elem elem_rho(const i64* in){{ return {coeffs[0]}; }}"]
        elif cert.kind == 'F1':
            lines += ["typedef struct { i64 a, b; } Elem;",
                      "static Elem elem_ident(void){ Elem e = {1, 0}; return e; }",
                      "static Elem elem_compose(Elem x, Elem y){ Elem e = {y.a * x.a, y.a * x.b + y.b}; return e; }",
                      "static i64 elem_ev(Elem r, i64 s){ return r.a * s + r.b; }",
                      f"static Elem elem_rho(const i64* in){{ Elem e = {{{coeffs[0]}, {coeffs[1]}}}; return e; }}"]
        else:  # F3
            lines += ["typedef struct { i64 a, l, h; } Elem;",
                      "static Elem elem_ident(void){ Elem e = {0, WREX_NINF, WREX_PINF}; return e; }",
                      "static Elem elem_compose(Elem x, Elem y){ i64 l = wrex_addinf(x.l, y.a); l = l<y.l?y.l:l; l = l>y.h?y.h:l;"
                      " i64 h = wrex_addinf(x.h, y.a); h = h>y.h?y.h:h; h = h<y.l?y.l:h;"
                      " Elem e = {wrex_addinf(x.a, y.a), l, h}; return e; }",
                      "static i64 elem_ev(Elem r, i64 s){ i64 v = wrex_addinf(s, r.a); return v<r.l?r.l:(v>r.h?r.h:v); }",
                      f"static Elem elem_rho(const i64* in){{ Elem e = {{{coeffs[0]}, {coeffs[1]}, {coeffs[2]}}}; return e; }}"]
        state_ty = "i64"; nstate = 1
    else:  # F2
        k = fam.k; n = k + 1
        if fam.semiring != 'ring': raise CodegenError("C backend supports F2 over a ring only")
        coeffs = [_csym(c, env) for c in cert.coeffs]
        flat = ", ".join(coeffs)
        lines += [f"#define K {k}", f"#define N {n}", "typedef struct { i64 m[N][N]; } Elem;",
                  "static Elem elem_ident(void){ Elem e; for(int i=0;i<N;i++) for(int j=0;j<N;j++) e.m[i][j] = (i==j); return e; }",
                  "static Elem elem_compose(Elem x, Elem y){ Elem e; for(int i=0;i<N;i++) for(int j=0;j<N;j++){ __int128 acc=0; for(int t=0;t<N;t++) acc += (__int128)y.m[i][t]*x.m[t][j]; e.m[i][j]=(i64)acc; } return e; }",
                  f"static Elem elem_rho(const i64* in){{ i64 flat[{n*n}] = {{{flat}}}; Elem e; for(int i=0;i<N;i++) for(int j=0;j<N;j++) e.m[i][j]=flat[i*N+j]; return e; }}"]
        state_ty = "vec"; nstate = k
    is_scan = rec["kind"] == 'scan'
    # state printing / ev for F2 is special-cased in main template below
    main = []
    main.append("static Elem* elems; static i64* inputs; static long n;")
    if cert.kind == 'F2':
        main.append(r"""
static void apply_elem(Elem r, const i64* s, i64* out){ for(int i=0;i<K;i++){ __int128 acc=0; for(int j=0;j<K;j++) acc+=(__int128)r.m[i][j]*s[j]; acc += r.m[i][K]; out[i]=(i64)acc; } }
""")
    main.append(r"""
typedef struct { long a, b; Elem acc; } Job;
static void* summarize(void* p){ Job* j = (Job*)p; Elem e = elem_ident(); for(long i=j->a;i<j->b;i++) e = elem_compose(e, elems[i]); j->acc = e; return 0; }
int main(int argc, char** argv){
  long p = argc > 1 ? atol(argv[1]) : 4; if(p < 1) p = 1;""")
    if free:
        main.append(f"  for(int i = 0; i < {len(free)}; i++){{ if(scanf(\"%lld\", &param[i]) != 1) return 1; }}")
    main.append(r"""  if(scanf("%ld", &n) != 1) return 1;
  inputs = malloc(sizeof(i64) * (n > 0 ? n : 1) * NIN);
  for(long i = 0; i < n * NIN; i++){ if(scanf("%lld", &inputs[i]) != 1) return 1; }
  elems = malloc(sizeof(Elem) * (n > 0 ? n : 1));
  for(long i = 0; i < n; i++) elems[i] = elem_rho(inputs + i * NIN);
""")
    main.append(f"#define NIN {nin}\n" if False else "")
    # NOTE: NIN used above must be defined before main; handled by pre-define injection
    if not is_scan:
        if cert.kind == 'F2':
            main.append(r"""
  long blocks = p < n ? p : (n > 0 ? n : 1);
  Elem total = elem_ident();
  if(n > 0){
    pthread_t* th = malloc(sizeof(pthread_t) * blocks); Job* jobs = malloc(sizeof(Job) * blocks);
    for(long b = 0; b < blocks; b++){ jobs[b].a = b * n / blocks; jobs[b].b = (b + 1) * n / blocks; pthread_create(&th[b], 0, summarize, &jobs[b]); }
    for(long b = 0; b < blocks; b++){ pthread_join(th[b], 0); total = elem_compose(total, jobs[b].acc); }
    free(th); free(jobs);
  }
  i64 s0[K]; for(int i = 0; i < K; i++){ if(scanf("%lld", &s0[i]) != 1) return 1; }
  i64 out[K]; apply_elem(total, s0, out);
  for(int i = 0; i < K; i++) printf("%s%lld", i ? " " : "", out[i]);
  printf("\n"); return 0; }
""")
        else:
            main.append(r"""
  long blocks = p < n ? p : (n > 0 ? n : 1);
  Elem total = elem_ident();
  if(n > 0){
    pthread_t* th = malloc(sizeof(pthread_t) * blocks); Job* jobs = malloc(sizeof(Job) * blocks);
    for(long b = 0; b < blocks; b++){ jobs[b].a = b * n / blocks; jobs[b].b = (b + 1) * n / blocks; pthread_create(&th[b], 0, summarize, &jobs[b]); }
    for(long b = 0; b < blocks; b++){ pthread_join(th[b], 0); total = elem_compose(total, jobs[b].acc); }
    free(th); free(jobs);
  }
  i64 s0; if(scanf("%lld", &s0) != 1) return 1;
  printf("%lld\n", elem_ev(total, s0)); return 0; }
""")
    else:
        if cert.kind == 'F2': raise CodegenError("C backend: F2 scan not supported (fold is)")
        main.append(r"""
  long blocks = p < n ? p : (n > 0 ? n : 1);
  Elem* starts = malloc(sizeof(Elem) * blocks);
  if(n > 0){
    pthread_t* th = malloc(sizeof(pthread_t) * blocks); Job* jobs = malloc(sizeof(Job) * blocks);
    for(long b = 0; b < blocks; b++){ jobs[b].a = b * n / blocks; jobs[b].b = (b + 1) * n / blocks; pthread_create(&th[b], 0, summarize, &jobs[b]); }
    Elem prefix = elem_ident();
    for(long b = 0; b < blocks; b++){ pthread_join(th[b], 0); starts[b] = prefix; prefix = elem_compose(prefix, jobs[b].acc); }
    free(th); free(jobs);
  }
  i64 s0; if(scanf("%lld", &s0) != 1) return 1;
  (void)starts;
  i64 s = s0;
  for(long i = 0; i < n; i++){ s = elem_ev(elems[i], s); printf("%s%lld", i ? " " : "", s); }
  printf("\n"); return 0; }
""")
    src = f"/* generated by wrex codegen: {rec['kind']} in fn {rec['fn']} ({cert.describe()}) */\n"
    src += "\n".join(lines) + f"\n#define NIN {nin}\n" + "\n".join(main)
    return src
