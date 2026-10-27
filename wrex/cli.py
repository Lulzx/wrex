import sys, os
from .compiler import compile_source, make_interpreter, CompileError
from .parser import ParseError
from .lexer import LexError
from .checker import CheckError, show
from .interp import Config, run_main
from .values import Trap, Out, fmt
from . import cost as costmod

def _load(path):
    with open(path,encoding='utf-8') as f: src=f.read()
    return src,compile_source(src)

def _threshold(rec,workers):
    # smallest n (searched exponentially) where parallel beats sequential, else None
    from . import cost as C
    cert=rec.get("cert")
    if cert is None: return None
    import math
    for n in [10,100,1000,10000,41000,100000,1000000,10000000]:
        costs=(C.scan_costs if rec["kind"]=='scan' else C.fold_costs)(n,workers,cert,5)
        par=[v for k,v in costs.items() if k!='Seq']
        if par and min(par)<costs['Seq']: return n
    return None

def cmd_explain(path,workers=8):
    src,(prog,checker,analysis)=_load(path)
    base=os.path.basename(path)
    if not analysis.records: print(f"{base}: no folds, scans, or tree folds")
    for rec in analysis.records:
        print(f"{base}:{rec['line']}  {rec['kind']}")
        if rec["kind"] in ('fold','scan'):
            print(f"  state       {show(rec['state_ty'])}")
            if "cert" in rec:
                c=rec["cert"]
                print(f"  family      {c.describe()}   (dimension {c.dim}, ⋄ cost {c.cost})")
                print(f"  L3          {c.l3}")
                if getattr(c,'reassoc',False): print("  warning     @reassoc: floating-point plans reassociate and are outside the plan-independence guarantee (spec 11)")
                th=_threshold(rec,workers)
                plan='ParScan' if rec['kind']=='scan' else 'ParFold'
                if th: print(f"  plan        {plan} when n >= {th} on {workers} workers, else Seq")
                else: print(f"  plan        Seq (parallel plan does not beat Seq at calibrated costs on {workers} workers)")
            else:
                print(f"  family      none")
                print(f"  runs sequentially. {rec['obs'].reason}")
        else:
            print(f"  tree        {show(rec['state_ty'])} -> {show(rec['result_ty'])}")
            if "cert" in rec:
                print(f"  contexts    {rec['cert'].describe()}")
                print(f"  L3          {rec['cert'].l3}")
                print(f"  plan        Contract when contraction beats ForkJoin by cost model, else ForkJoin")
            else:
                print(f"  contexts    none ({rec['obs'].reason})")
                print(f"  plan        ForkJoin")
    return 0

def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ('-h','--help'):
        print("usage: wrex run [--plan=seq|par|auto] [--device=auto|cpu|metal|cuda|webgpu] file.wrx [args...]\n"
              "       wrex check file.wrx\n       wrex explain file.wrx\n       wrex test file.wrx\n       wrex calibrate")
        return 0
    cmd=argv[0]
    if cmd=='bench':
        from .bench import main as bench_main
        return bench_main(argv[1:])
    if cmd=='fuzz':
        from .fuzz import main as fuzz_main
        return fuzz_main(argv[1:])
    if cmd=='codegen':
        fn=None; out=None; rest=[]
        i=1
        while i < len(argv):
            a=argv[i]
            if a.startswith('--fn='): fn=a.split('=',1)[1]
            elif a in ('-o','--out'): i+=1; out=argv[i]
            else: rest.append(a)
            i+=1
        if not rest: print("wrex codegen: missing file",file=sys.stderr); return 2
        try:
            src,(prog,checker,analysis)=_load(rest[0])
            from .codegen import generate_c, CodegenError
            csrc=generate_c(prog,analysis,fn)
        except Exception as ex:
            print(f"error: {ex}",file=sys.stderr); return 1
        if out:
            with open(out,'w') as f: f.write(csrc)
            print(out)
        else: print(csrc)
        return 0
    try:
        if cmd=='calibrate':
            cal=costmod.calibrate(); print(f"calibrated: {cal}"); return 0
        if cmd in ('check','explain','test','run'):
            plan='auto'; device='cpu'; workers=8; rest=[]
            for a in argv[1:]:
                if a.startswith('--plan='): plan=a.split('=',1)[1]
                elif a.startswith('--device='):
                    device=a.split('=',1)[1]
                    if device=='auto': device='cpu'
                elif a.startswith('--workers='): workers=int(a.split('=',1)[1])
                else: rest.append(a)
            if not rest: print(f"wrex {cmd}: missing file",file=sys.stderr); return 2
            path=rest[0]
            if cmd=='explain': return cmd_explain(path,workers)
            src,(prog,checker,analysis)=_load(path)
            if cmd=='check':
                for rec in analysis.records:
                    if "cert" in rec: print(f"{path}:{rec['line']} {rec['kind']}: {rec['cert'].describe()}")
                    else: print(f"{path}:{rec['line']} {rec['kind']}: sequential. {rec['obs'].reason}")
                print("ok"); return 0
            if cmd=='test':
                from .conformance import family_law_tests, differential
                bad=0
                laws=family_law_tests()
                for name,errs in laws.items():
                    print(f"family laws {name}: {'ok' if not errs else 'FAIL '+str(errs)}")
                    bad+=bool(errs)
                from .conformance import exhaustive_finite_laws
                for name,errs in exhaustive_finite_laws().items():
                    print(f"exhaustive laws {name}: {'ok' if not errs else 'FAIL '+str(errs)}")
                    bad+=bool(errs)
                if any(d.name=='main' for d in prog.decls):
                    ok,outs=differential(src,rest[1:])
                    print(f"differential (seq/par/auto): {'ok' if ok else 'FAIL '+str(outs)}")
                    bad+=not ok
                else:
                    print("differential: no main; skipped (family laws + check only)")
                return 1 if bad else 0
            # run
            interp=make_interpreter(prog,analysis,Config(plan=plan,workers=workers,device=device))
            if not any(d.name=='main' for d in prog.decls):
                print("error: program has no main function",file=sys.stderr); return 1
            result=run_main(interp,rest[1:])
            if isinstance(result,Out):
                for item in result.items: print(fmt(item))
            elif result is not None: print(fmt(result))
            return 0
        print(f"unknown command {cmd}",file=sys.stderr); return 2
    except Trap as t:
        print(str(t),file=sys.stderr); return 1
    except (ParseError,LexError,CheckError,CompileError) as ex:
        print(f"error: {ex}",file=sys.stderr); return 1

if __name__=='__main__': sys.exit(main())
