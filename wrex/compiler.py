from .parser import parse
from .checker import Checker, resolve, show
from . import nodes as N
from .inference import infer_step, infer_treefold, Cert, Obstruction
from .interp import Interpreter, Config

class CompileError(Exception): pass

class Analysis:
    def __init__(self): self.records=[]  # dicts: kind,line,node,cert,obs,fn,state_ty

def _walk(node,fn,visit):
    if isinstance(node,N.Node): visit(node,fn)
    for v in vars(node).values() if isinstance(node,N.Node) else []:
        if isinstance(v,N.Node): _walk(v,fn,visit)
        elif isinstance(v,list):
            for y in v:
                if isinstance(y,N.Node): _walk(y,fn,visit)
                elif isinstance(y,tuple):
                    for z in y:
                        if isinstance(z,N.Node): _walk(z,fn,visit)

def build_user_families(prog,checker):
    """Evaluate family declarations and machine-check their laws (spec 6.8).

    Returns (families, rejected, ops): certified name -> UserMonoid,
    name -> rejection reason, and every declared name -> callable operator.
    """
    from . import families as F
    from .inference import verify_user_family
    from .interp import Closure
    from .values import Trap
    types={d.name:d.variants for d in prog.decls if isinstance(d,N.TypeDecl)}
    families={}; rejected={}; ops={}
    decls=[d for d in prog.decls if isinstance(d,N.FamilyDecl)]
    if not decls: return families,rejected,ops
    interp=Interpreter(prog)
    for d in decls:
        sty=resolve(d.state_ty,checker.subst)
        try: ident=interp.eval_expr(d.identity,{})
        except Trap as t: rejected[d.name]=f"user family '{d.name}' rejected: identity traps ({t.kind}: {t.msg})"; continue
        clo=Closure(d.compose,{},interp)
        fn=lambda a,b,clo=clo,interp=interp: interp.call_value(clo,(a,b))
        # every declared family stays callable as an ordinary operator, so
        # sequential fallback evaluates the step exactly as written
        ops[d.name]=F.UserMonoid(d.name,show(sty),ident,fn,"unchecked")
        ok,info=verify_user_family(d.name,sty,ident,fn,types)
        if ok: families[d.name]=F.UserMonoid(d.name,show(sty),ident,fn,info)
        else: rejected[d.name]=info
    return families,rejected,ops

def analyze(prog,checker):
    fns={d.name:d for d in prog.decls if isinstance(d,N.FnDecl)}
    types={d.name:d.variants for d in prog.decls if isinstance(d,N.TypeDecl)}
    analysis=Analysis()
    analysis.user_families,analysis.rejected_families,analysis.family_ops=build_user_families(prog,checker)
    subst=checker.subst
    for d in prog.decls:
        if not isinstance(d,N.FnDecl): continue
        finite='@finite' in d.annotations; reassoc='@reassoc' in d.annotations
        def visit(node,fn,d=d,finite=finite,reassoc=reassoc):
            if isinstance(node,N.Call) and isinstance(node.f,N.Var) and node.f.name in ('fold','scan') and len(node.args)==3:
                xt=resolve(node.args[0].ty,subst); st=resolve(node.args[1].ty,subst)
                xt_el=xt[1] if isinstance(xt,tuple) and xt and xt[0]=='list' else None
                r=infer_step(node.args[2],st,xt_el,fns,types,finite,reassoc,analysis.user_families,analysis.rejected_families)
                rec={"kind":node.f.name,"line":node.line,"node":node,"fn":d.name,"state_ty":st,"input_ty":xt_el}
                if isinstance(r,Cert): rec["cert"]=r
                else: rec["obs"]=r
                analysis.records.append(rec)
            elif isinstance(node,N.TreeFold):
                ctx,obs=infer_treefold(node,types,fns,analysis.user_families,analysis.rejected_families)
                rec={"kind":"treefold","line":node.line,"node":node,"fn":d.name,"state_ty":resolve(node.scrut.ty,subst),"result_ty":resolve(node.ty,subst)}
                if ctx is not None: rec["cert"]=ctx
                else: rec["obs"]=obs
                analysis.records.append(rec)
        _walk(d.body,d.name,visit)
    # @par enforcement
    errors=[]
    for rec in analysis.records:
        fn=fns[rec["fn"]]
        if '@par' in fn.annotations and "obs" in rec:
            errors.append(f"{rec['kind']} at line {rec['line']} in @{('par')} fn {rec['fn']}: {rec['obs'].reason}")
    if errors: raise CompileError("\n".join(errors))
    return analysis

def compile_source(src):
    prog=parse(src)
    checker=Checker(prog); checker.check()
    analysis=analyze(prog,checker)
    return prog,checker,analysis

def make_interpreter(prog,analysis,config=None):
    interp=Interpreter(prog,config or Config())
    interp.user_families=dict(getattr(analysis,'family_ops',None) or getattr(analysis,'user_families',{}) or {})
    for d in prog.decls:
        if isinstance(d,N.FnDecl):
            interp.annotations[d.name]=d.annotations
            if '@seq' in d.annotations: interp.seq_fns.add(d.name)
    for rec in analysis.records:
        if "cert" not in rec: continue
        if rec["kind"] in ('fold','scan'): interp.certificates[id(rec["node"])]=rec["cert"]
        else: interp.ctx_certs[id(rec["node"])]=rec["cert"]
    return interp
