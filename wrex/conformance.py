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
