import random
from . import families as F
from .values import Trap, fmt

def _eq(a,b): return a==b

def check_family(name,fam,gen_r,gen_s,trials=200,seed=1):
    rnd=random.Random(seed); errs=[]
    ident=fam.identity()
    for _ in range(trials):
        r1,r2,r3=(gen_r(rnd) for _ in range(3)); s=gen_s(rnd)
        # associativity
        if not _eq(fam.compose(fam.compose(r1,r2),r3),fam.compose(r1,fam.compose(r2,r3))): errs.append("associativity"); break
        if not _eq(fam.ev(ident,s),s): errs.append("L1"); break
        if not _eq(fam.ev(fam.compose(r1,r2),s),fam.ev(r2,fam.ev(r1,s))): errs.append("L2"); break
    return errs

def family_law_tests():
    results={}
    ri=lambda rnd: rnd.randint(-20,20)
    results['F0 +']=check_family('F0',F.F0('+','int'),ri,ri)
    results['F0 min']=check_family('F0',F.F0('min','int'),ri,ri)
    results['F0 * i64']=check_family('F0',F.F0('*','i64'),lambda r:r.randint(-1000,1000),lambda r:r.randint(-1000,1000))
    results['F1 int']=check_family('F1',F.F1('int'),lambda r:(r.randint(-5,5),r.randint(-20,20)),ri)
    results['F1 i64']=check_family('F1',F.F1('i64'),lambda r:(r.randint(-5,5),r.randint(-20,20)),lambda r:r.randint(-1000,1000))
    def mat(rnd,k=2,sem='ring'):
        fam=F.F2(k,sem,'int'); return fam.compose(fam.identity(),[[rnd.randint(-3,3) for _ in range(k+1)] for __ in range(k+1)]) if False else [[rnd.randint(-3,3) if i<k else rnd.randint(-3,3) for j in range(k+1)] for i in range(k)]+[[0]*k+[1]]
    # build matrices properly: last row [0,0,1]
    def mat2(rnd):
        return [[rnd.randint(-3,3),rnd.randint(-3,3),rnd.randint(-5,5)],[rnd.randint(-3,3),rnd.randint(-3,3),rnd.randint(-5,5)],[0,0,1]]
    results['F2 ring']=check_family('F2',F.F2(2,'ring','int'),mat2,lambda r:(r.randint(-5,5),r.randint(-5,5)))
    def matT(rnd):
        return [[rnd.randint(-3,5),rnd.randint(-3,5),rnd.randint(-5,5)],[rnd.randint(-3,5),rnd.randint(-3,5),rnd.randint(-5,5)],[F.NINF,F.NINF,0]]
    results['F2 maxplus']=check_family('F2t',F.F2(2,'maxplus','int'),matT,lambda r:(r.randint(-5,5),r.randint(-5,5)))
    results['F3']=check_family('F3',F.F3(),lambda r:(r.randint(-5,5),r.choice([F.NINF,-10,0]),r.choice([0,10,F.INF])),ri)
    results['F4']=check_family('F4',F.F4([0,1,2]),lambda r:tuple(r.choice([0,1,2]) for _ in range(3)),lambda r:r.choice([0,1,2]))
    f5=F.F5([False,True],F.F0('+','int'))
    results['F5']=check_family('F5',f5,lambda r:tuple((r.choice([False,True]),r.randint(-5,5)) for _ in range(2)),lambda r:(r.choice([False,True]),r.randint(-10,10)))
    f6=F.F6([F.F1('int'),F.F0('+','int')],[ [0],[1] ])
    results['F6']=check_family('F6',f6,lambda r:((r.randint(-3,3),r.randint(-5,5)),r.randint(-5,5)),lambda r:(r.randint(-5,5),r.randint(-5,5)))
    return results

def exhaustive_finite_laws():
    """Exhaustive (not sampled) law checks over every element of small
    finite families: F4 over a 3-state set, and F5 with 2 control states
    and payload elements in {-1, 0, 1} under F0(+)."""
    results = {}
    # F4: all 27 functions {0,1,2} -> {0,1,2}
    import itertools
    fam = F.F4([0, 1, 2]); elems = list(itertools.product([0, 1, 2], repeat=3)); errs = []
    for a in elems:
        if fam.ev(fam.compose(fam.identity(), a), 0) != fam.ev(a, 0): errs.append("identity"); break
        for b in elems:
            for c in elems:
                if fam.compose(fam.compose(a, b), c) != fam.compose(a, fam.compose(b, c)): errs.append("assoc"); break
            for s in (0, 1, 2):
                if fam.ev(fam.compose(a, b), s) != fam.ev(b, fam.ev(a, s)): errs.append("L2"); break
    results['F4 exhaustive (27 elements)'] = errs
    # F5: control {0,1}, payload F0(+) with elements in {-1,0,1}: 36 elements
    fam5 = F.F5([0, 1], F.F0('+', 'int'))
    per_q = [(q2, m) for q2 in (0, 1) for m in (-1, 0, 1)]
    elems5 = [(a, b) for a in per_q for b in per_q]; errs = []
    for a in elems5:
        for b in elems5:
            for c in elems5:
                if fam5.compose(fam5.compose(a, b), c) != fam5.compose(a, fam5.compose(b, c)): errs.append("assoc"); break
            for s in [(q, v) for q in (0, 1) for v in (-2, 0, 2)]:
                try:
                    if fam5.ev(fam5.compose(a, b), s) != fam5.ev(b, fam5.ev(a, s)): errs.append("L2")
                except Exception: errs.append("ev error")
                break
    results['F5 exhaustive (36 elements)'] = errs
    return results

def exhaustive_bounded_laws():
    """Exhaustive law checks over bounded domains for F0-F3, including
    infinity boundaries for F3 and tropical-style min/max for F0."""
    import itertools
    results = {}
    D = [-2, -1, 0, 1, 2]
    # F0 over each operator: elements and states from the domain
    for op, dom in [('+', D), ('*', D), ('min', D), ('max', D), ('&', [0, 1, 2, 3]), ('|', [0, 1, 2, 3]), ('^', [0, 1, 2, 3])]:
        fam = F.F0(op, 'int'); errs = []
        for a in dom:
            if fam.ev(fam.identity(), a) != a: errs.append("L1"); break
            for b in dom:
                for c in dom:
                    if fam.compose(fam.compose(a, b), c) != fam.compose(a, fam.compose(b, c)): errs.append("assoc"); break
                for s in dom:
                    if fam.ev(fam.compose(a, b), s) != fam.ev(b, fam.ev(a, s)): errs.append("L2"); break
        results[f'F0 {op} bounded'] = errs
    for op in ('and', 'or'):
        fam = F.F0(op, 'bool'); errs = []
        for a in (False, True):
            for b in (False, True):
                for c in (False, True):
                    if fam.compose(fam.compose(a, b), c) != fam.compose(a, fam.compose(b, c)): errs.append("assoc")
                for s in (False, True):
                    if fam.ev(fam.compose(a, b), s) != fam.ev(b, fam.ev(a, s)): errs.append("L2")
        results[f'F0 {op} bounded'] = errs
    # F1: a in {-1,0,1,2}, b in {-2..2} (20 elements, all triples)
    fam = F.F1('int'); elems = [(a, b) for a in (-1, 0, 1, 2) for b in D]; errs = []
    for a in elems:
        if fam.ev(fam.identity(), 0) != 0: errs.append("L1"); break
        for b in elems:
            for c in elems:
                if fam.compose(fam.compose(a, b), c) != fam.compose(a, fam.compose(b, c)): errs.append("assoc"); break
            for s in (-3, 0, 3):
                if fam.ev(fam.compose(a, b), s) != fam.ev(b, fam.ev(a, s)): errs.append("L2"); break
    results['F1 bounded (20 elements)'] = errs
    # F3: a in {-1,0,1}, intervals over {-inf,-2,0,2,+inf} (45 elements, all triples)
    bounds = [F.NINF, -2, 0, 2, F.INF]
    intervals = [(l, h) for l in bounds for h in bounds if l <= h]
    elems3 = [(a, l, h) for a in (-1, 0, 1) for (l, h) in intervals]; fam3 = F.F3(); errs = []
    for a in elems3:
        for b in elems3:
            for c in elems3:
                if fam3.compose(fam3.compose(a, b), c) != fam3.compose(a, fam3.compose(b, c)): errs.append("assoc"); break
            for s in (-3, 0, 3):
                if fam3.ev(fam3.compose(a, b), s) != fam3.ev(b, fam3.ev(a, s)): errs.append("L2"); break
    results['F3 bounded+inf (45 elements)'] = errs
    return results

def run_with_plan(src,plan,workers=8,argv=None):
    from .compiler import compile_source, make_interpreter
    from .interp import Config, run_main
    prog,checker,analysis=compile_source(src)
    interp=make_interpreter(prog,analysis,Config(plan=plan,workers=workers))
    try:
        v=run_main(interp,argv or [])
        return ("ok",fmt(v))
    except Trap as t:
        return ("trap",t.kind)

def differential(src,argv=None):
    outs={}
    for plan in ('seq','par','auto'):
        outs[plan]=run_with_plan(src,plan,argv=argv)
    ok=len(set(outs.values()))==1
    return ok,outs
