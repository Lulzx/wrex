from . import nodes as N
from .values import ADT, Out, Trap, wrap, is_machine, fmt
import os

class Closure:
    def __init__(self,lam,env,interp): self.lam=lam; self.env=env; self.interp=interp
    def __call__(self,*args): return self.interp.call_closure(self,args)
    def __repr__(self): return "<lambda>"

class Config:
    def __init__(self,plan="auto",workers=8,device="cpu"):
        self.plan=plan; self.workers=workers; self.device=device

class Interpreter:
    def __init__(self,prog,config=None,checker=None):
        import sys as _sys
        if _sys.getrecursionlimit() < 100000: _sys.setrecursionlimit(100000)
        self.prog=prog; self.config=config or Config()
        self.fns={}; self.ctors={}; self.types={}
        for d in prog.decls:
            if isinstance(d,N.FnDecl): self.fns[d.name]=d
            if isinstance(d,N.TypeDecl):
                self.types[d.name]=d.variants
                for c,f in d.variants: self.ctors[c]=(d.name,f)
        self.output=[]; self.certificates={}  # id(node)->cert, filled by compiler
        self.ctx_certs={}; self.fn_stack=[]; self.seq_fns=set(); self.annotations={}
        self.user_families={}  # name -> UserMonoid, filled by compiler (spec 6.8)
        self._plans=None
    # ---------- patterns
    def match_pattern(self,pat,val,env):
        if isinstance(pat,N.PWild): return True
        if isinstance(pat,N.PVar): env[pat.name]=val; return True
        if isinstance(pat,N.PLit): return val==pat.value and type(val)==type(pat.value) or val==pat.value
        if isinstance(pat,N.PTuple):
            if not isinstance(val,tuple) or len(val)!=len(pat.items): return False
            for p,v in zip(pat.items,val):
                if not self.match_pattern(p,v,env): return False
            return True
        if isinstance(pat,N.PCtor):
            if not isinstance(val,ADT) or val.ctor!=pat.name or len(val.fields)!=len(pat.items): return False
            for p,v in zip(pat.items,val.fields):
                if not self.match_pattern(p,v,env): return False
            return True
        return False
    def bind_pattern(self,pat,val):
        env={}
        if not self.match_pattern(pat,val,env): raise Trap("match","pattern binding failed",getattr(pat,'line',None))
        return env
    # ---------- calls
    def call_closure(self,clo,args):
        lam=clo.lam
        if len(args)!=len(lam.params): raise Trap("arity","lambda arity mismatch",lam.line)
        env=dict(clo.env)
        for p,a in zip(lam.params,args): env.update(self.bind_pattern(p,a))
        return self.eval_block(lam.body,env)
    def call_value(self,f,args,line=None):
        if isinstance(f,Closure): return f(*args)
        if isinstance(f,str) and f in self.fns: return self.call_fn(f,args)
        if callable(f): return f(*args)
        raise Trap("call","not callable",line)
    def call_fn(self,name,args):
        d=self.fns[name]
        if len(args)!=len(d.params): raise Trap("arity",f"{name} arity",d.line)
        env={}
        for (pn,_),a in zip(d.params,args):
            env[pn]=a
        self.fn_stack.append(name)
        try: return self.eval_block(d.body,env)
        finally: self.fn_stack.pop()
    # ---------- eval
    def eval_block(self,blk,env):
        env=dict(env)
        binds=blk.bindings; i=0
        while i < len(binds):
            pat,e=binds[i]
            # product scan fusion (spec 8.3): consecutive scans over the same input
            if (i+1 < len(binds) and isinstance(e,N.Call) and isinstance(e.f,N.Var) and e.f.name=='scan'
                    and isinstance(binds[i+1][1],N.Call) and isinstance(binds[i+1][1].f,N.Var) and binds[i+1][1].f.name=='scan'):
                from .fusion import structural_key, fused_scans
                e2=binds[i+1][1]
                if structural_key(e.args[0])==structural_key(e2.args[0]):
                    xs=self._fold_input(e.args[0],env)
                    s1=self.eval_expr(e.args[1],env); st1=self.eval_expr(e.args[2],env)
                    s2=self.eval_expr(e2.args[1],env); st2=self.eval_expr(e2.args[2],env)
                    out1,out2=fused_scans(self,e,e2,xs,s1,st1,s2,st2)
                    env.update(self.bind_pattern(pat,out1)); env.update(self.bind_pattern(binds[i+1][0],out2))
                    i+=2; continue
            v=self.eval_expr(e,env); env.update(self.bind_pattern(pat,v)); i+=1
        return self.eval_expr(blk.expr,env)
    def _fold_input(self,e,env):
        """Evaluate a fold/scan input, fusing a feeding map (spec 8.3):
        the mapped array is never materialized; the map function's traps
        are checked in order first, matching reference evaluation order."""
        if isinstance(e,N.Call) and isinstance(e.f,N.Var) and e.f.name=='map' and len(e.args)==2:
            from .fusion import LazyMap
            base=self._fold_input(e.args[0],env); f=self.eval_expr(e.args[1],env)
            lazy=LazyMap(base,f,self)
            for x in lazy: pass  # trap-order pre-pass; values recomputed lazily later (pure)
            return lazy
        return self.eval_expr(e,env)
    def eval_expr(self,e,env):
        if isinstance(e,N.Lit): return wrap(e.value,e.ty) if is_machine(e.ty) else e.value
        if isinstance(e,N.Var):
            if e.name in env: return env[e.name]
            if e.name in self.ctors and not self.ctors[e.name][1]:
                return ADT(e.name,(),self.ctors[e.name][0])
            if e.name in self.fns:
                d=self.fns[e.name]
                # first-class fn ref as closure-like: wrap
                interp=self
                class FnRef:
                    def __call__(self,*a): return interp.call_fn(e.name,a)
                return FnRef()
            if e.name in self.user_families:
                fam=self.user_families[e.name]
                class FamRef:
                    def __call__(self,*a): return fam.compose(a[0],a[1])
                return FamRef()
            raise Trap("name",f"undefined {e.name}",e.line)
        if isinstance(e,N.CtorExpr):
            vals=[self.eval_expr(a,env) for a in e.args]
            tn=self.ctors[e.name][0] if e.name in self.ctors else ""
            return ADT(e.name,tuple(vals),tn)
        if isinstance(e,N.TupleExpr): return tuple(self.eval_expr(x,env) for x in e.items)
        if isinstance(e,N.ListExpr): return [self.eval_expr(x,env) for x in e.items]
        if isinstance(e,N.SeqExpr):
            out=Out(); last=None
            for x in e.items:
                v=self.eval_expr(x,env); last=v
                if isinstance(v,Out): out.items.extend(v.items)
            return out if out.items else last
        if isinstance(e,N.Block): return self.eval_block(e,env)
        if isinstance(e,N.Lambda): return Closure(e,dict(env),self)
        if isinstance(e,N.If):
            return self.eval_expr(e.then,env) if self.eval_expr(e.cond,env) else self.eval_expr(e.els,env)
        if isinstance(e,N.Unary):
            v=self.eval_expr(e.e,env)
            if e.op=='not': return not v
            r=-v; return wrap(r,e.ty) if is_machine(e.ty) else r
        if isinstance(e,N.BinOp): return self.eval_binop(e,env)
        if isinstance(e,N.Index):
            obj=self.eval_expr(e.obj,env); idx=self.eval_expr(e.idx,env)
            if not isinstance(obj,list) or not (0 <= idx < len(obj)):
                raise Trap("index_out_of_bounds",f"index {idx} out of bounds",e.line)
            return obj[idx]
        if isinstance(e,N.Match):
            v=self.eval_expr(e.scrut,env)
            for pat,blk in e.arms:
                env2=dict(env)
                if self.match_pattern(pat,v,env2): return self.eval_block(blk,env2)
            raise Trap("match_failure","no match arm matched",e.line)
        if isinstance(e,N.TreeFold): return self.eval_treefold(e,env)
        if isinstance(e,N.Call): return self.eval_call(e,env)
        raise Trap("eval",f"unknown node {type(e).__name__}",getattr(e,'line',None))
    def eval_binop(self,e,env):
        op=e.op
        l=self.eval_expr(e.l,env); r=self.eval_expr(e.r,env)
        if op=='and': return bool(l and r)
        if op=='or': return bool(l or r)
        if op=='==': return l==r
        if op=='!=': return l!=r
        if op=='<': return l<r
        if op=='<=': return l<=r
        if op=='>': return l>r
        if op=='>=': return l>=r
        if op in ('/','%'):
            if r==0: raise Trap("div_by_zero","division by zero" if op=='/' else "remainder by zero",e.line)
            q=abs(l)//abs(r)
            if (l<0)!=(r<0): q=-q
            res=q if op=='/' else l-q*r
            return wrap(res,e.ty) if is_machine(e.ty) else res
        if op=='+': res=l+r
        elif op=='-': res=l-r
        elif op=='*': res=l*r
        elif op=='&': res=int(l)&int(r)
        elif op=='|': res=int(l)|int(r)
        elif op=='^': res=int(l)^int(r)
        elif op=='<<': res=int(l)<<int(r)
        elif op=='>>': res=int(l)>>int(r)
        else: raise Trap("op",op,e.line)
        return wrap(res,e.ty) if is_machine(e.ty) else res
    def eval_call(self,e,env):
        if isinstance(e.f,N.Var):
            name=e.f.name
            if name in ('map','zip','range','len','fold','scan','min','max','clamp','print','write','read_i64s','read_ints','read_u8s'):
                return self.eval_builtin(name,e,env)
            if name in self.fns:
                args=[self.eval_expr(a,env) for a in e.args]; return self.call_fn(name,args)
            if name in self.user_families:
                args=[self.eval_expr(a,env) for a in e.args]; return self.user_families[name].compose(args[0],args[1])
        f=self.eval_expr(e.f,env); args=[self.eval_expr(a,env) for a in e.args]
        return self.call_value(f,args,e.line)
    def eval_builtin(self,name,e,env):
        args=e.args
        if name=='len': return len(self.eval_expr(args[0],env))
        if name=='range':
            a=self.eval_expr(args[0],env); b=self.eval_expr(args[1],env); return list(range(a,b))
        if name=='zip':
            xs=self.eval_expr(args[0],env); ys=self.eval_expr(args[1],env)
            if len(xs)!=len(ys): raise Trap("zip_length","zip of arrays with unequal lengths",e.line)
            return list(zip(xs,ys))
        if name=='map':
            xs=self.eval_expr(args[0],env); f=self.eval_expr(args[1],env)
            if self.config.plan!='seq' and self.config.device=='cpu':
                return [self.call_value(f,(x,),e.line) for x in xs]  # ParMap: elementwise, order/traps preserved
            return [self.call_value(f,(x,),e.line) for x in xs]
        if name in ('min','max'):
            a=self.eval_expr(args[0],env); b=self.eval_expr(args[1],env); return min(a,b) if name=='min' else max(a,b)
        if name=='clamp':
            v=self.eval_expr(args[0],env); lo=self.eval_expr(args[1],env); hi=self.eval_expr(args[2],env)
            if lo>hi: raise Trap("clamp_bounds","clamp with lo > hi",e.line)
            return min(hi,max(lo,v))
        if name=='print':
            v=self.eval_expr(args[0],env); return Out([v])
        if name=='write':
            p=self.eval_expr(args[0],env); v=self.eval_expr(args[1],env)
            with open(p,'w') as fh: fh.write(fmt(v))
            return Out([])
        if name in ('read_i64s','read_ints','read_u8s'):
            p=self.eval_expr(args[0],env)
            with open(p) as fh: data=fh.read().split()
            ty={'read_i64s':'i64','read_ints':'int','read_u8s':'u8'}[name]
            return [wrap(int(x),ty) for x in data]
        if name=='fold':
            xs=self._fold_input(args[0],env); s0=self.eval_expr(args[1],env); step=self.eval_expr(args[2],env)
            return self.do_fold(e,xs,s0,step)
        if name=='scan':
            xs=self._fold_input(args[0],env); s0=self.eval_expr(args[1],env); step=self.eval_expr(args[2],env)
            return self.do_scan(e,xs,s0,step)
        raise Trap("builtin",name,e.line)
    # fold/scan/treefold dispatch (plans.py overrides via do_* )
    def do_fold(self,e,xs,s0,step):
        from . import plans
        return plans.fold(self,e,xs,s0,step)
    def do_scan(self,e,xs,s0,step):
        from . import plans
        return plans.scan(self,e,xs,s0,step)
    def eval_treefold(self,e,env):
        from . import plans
        return plans.treefold(self,e,env)
    # sequential treefold used by plans
    def treefold_seq(self,tree,arms,env):
        if not isinstance(tree,ADT): raise Trap("treefold","not an algebraic value",None)
        tn=tree.type_name or (self.ctors[tree.ctor][0] if tree.ctor in self.ctors else "")
        fields=self.ctors[tree.ctor][1] if tree.ctor in self.ctors else []
        for pat,blk in arms:
            if not isinstance(pat,N.PCtor) or pat.name!=tree.ctor: continue
            env2=dict(env); ok=True
            # evaluate children left-to-right first
            vals=[]
            for ft,fv in zip(fields,tree.fields):
                vals.append(self.treefold_seq(fv,arms,env) if ft==tn else fv)
            for p,v in zip(pat.items,vals):
                if not self.match_pattern(p,v,env2): ok=False; break
            if ok: return self.eval_block(blk,env2)
        raise Trap("match_failure","treefold: no arm matched",None)

def run_main(interp,argv=None):
    if 'main' not in interp.fns: raise Trap("no_main","program has no main function",None)
    return interp.call_fn('main',[list(argv or [])])
