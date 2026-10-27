from . import nodes as N
from .values import MACHINE

class CheckError(Exception): pass

_counter=[0]
def fresh():
    _counter[0]+=1; return ('var',_counter[0])
def resolve(t,subst):
    while isinstance(t,tuple) and t and t[0]=='var' and t[1] in subst: t=subst[t[1]]
    if isinstance(t,tuple) and t:
        if t[0]=='list': return ('list',resolve(t[1],subst))
        if t[0]=='tuple': return ('tuple',[resolve(x,subst) for x in t[1]])
        if t[0]=='fn': return ('fn',[resolve(x,subst) for x in t[1]],resolve(t[2],subst))
    return t
def unify(a,b,subst,ctx=""):
    a=resolve(a,subst); b=resolve(b,subst)
    if a==b: return a
    if isinstance(a,tuple) and a and a[0]=='var': subst[a[1]]=b; return b
    if isinstance(b,tuple) and b and b[0]=='var': subst[b[1]]=a; return a
    if isinstance(a,tuple) and isinstance(b,tuple) and a[0]==b[0]:
        if a[0]=='list': unify(a[1],b[1],subst,ctx); return resolve(a,subst)
        if a[0]=='tuple' and len(a[1])==len(b[1]):
            for x,y in zip(a[1],b[1]): unify(x,y,subst,ctx)
            return resolve(a,subst)
        if a[0]=='fn' and len(a[1])==len(b[1]):
            for x,y in zip(a[1],b[1]): unify(x,y,subst,ctx)
            unify(a[2],b[2],subst,ctx); return resolve(a,subst)
    # numeric widening: unsuffixed literals handled via int compat: int ~ machine allowed? keep strict except int vs machine unify to machine
    if a=="int" and isinstance(b,str) and b in MACHINE: return b
    if b=="int" and isinstance(a,str) and a in MACHINE: return a
    raise CheckError(f"type mismatch {ctx}: {show(a)} vs {show(b)}")
def show(t):
    if isinstance(t,tuple) and t:
        if t[0]=='var': return "?"
        if t[0]=='list': return f"[{show(t[1])}]"
        if t[0]=='tuple': return "("+", ".join(show(x) for x in t[1])+")"
        if t[0]=='fn': return "("+", ".join(show(x) for x in t[1])+") -> "+show(t[2])
    return str(t)

BUILTIN_FNS={}  # handled specially

class Checker:
    def __init__(self,prog):
        self.prog=prog; self.types={}; self.ctors={}; self.fns={}; self.subst={}
        for d in prog.decls:
            if isinstance(d,N.TypeDecl):
                self.types[d.name]=d.variants
                for c,fields in d.variants: self.ctors[c]=(d.name,fields)
        for d in prog.decls:
            if isinstance(d,N.FnDecl): self.fns[d.name]=d
    def check(self):
        for d in self.prog.decls:
            if isinstance(d,N.FnDecl): self.check_fn(d)
        # default unresolved numeric type variables to `int` (spec 3: unsuffixed literals take type from context)
        for i in range(1,_counter[0]+1):
            if i not in self.subst: self.subst[i]="int"
        def fix(node):
            if isinstance(node,N.Node):
                if node.ty is not None: node.ty=resolve(node.ty,self.subst)
                for v in vars(node).values():
                    if isinstance(v,N.Node): fix(v)
                    elif isinstance(v,list):
                        for y in v:
                            if isinstance(y,N.Node): fix(y)
                            elif isinstance(y,tuple):
                                for z in y:
                                    if isinstance(z,N.Node): fix(z)
        fix(self.prog)
        return self.prog
    def check_fn(self,d):
        self.cur_fn=d; env={n:t for n,t in d.params}
        self.check_block(d.body,env,d.ret)
    def check_block(self,blk,env,expected=None):
        env=dict(env)
        for pat,e in blk.bindings:
            t=self.check_expr(e,env); self.bind_pattern(pat,t,env)
        t=self.check_expr(blk.expr,env)
        if expected is not None: unify(t,expected,self.subst,f"return of fn")
        blk.ty=resolve(t,self.subst); return blk.ty
    def bind_pattern(self,pat,ty,env):
        ty=resolve(ty,self.subst)
        if isinstance(pat,N.PVar): env[pat.name]=ty
        elif isinstance(pat,N.PWild): pass
        elif isinstance(pat,N.PLit): pass
        elif isinstance(pat,N.PTuple):
            if not (isinstance(ty,tuple) and ty and ty[0]=='tuple'):
                tt=('tuple',[fresh() for _ in pat.items]); unify(ty,tt,self.subst,"tuple pattern"); ty=resolve(ty,self.subst)
            for p,t in zip(pat.items,ty[1]): self.bind_pattern(p,t,env)
        elif isinstance(pat,N.PCtor):
            if pat.name not in self.ctors: raise CheckError(f"unknown constructor {pat.name}")
            tname,fields=self.ctors[pat.name]; unify(ty,tname,self.subst,f"ctor {pat.name}")
            if len(pat.items)!=len(fields): raise CheckError(f"ctor {pat.name} arity")
            for p,t in zip(pat.items,fields): self.bind_pattern(p,t,env)
        pat.ty=ty
    def pattern_env(self,pat,ty,env):
        e=dict(env); self.bind_pattern(pat,ty,e); return e
    def check_lambda(self,lam,param_tys,ret_ty,env):
        e=dict(env)
        if len(lam.params)!=len(param_tys): raise CheckError("lambda arity mismatch")
        for p,t in zip(lam.params,param_tys): self.bind_pattern(p,t,e)
        # body is Block
        t=self.check_block(lam.body,e,ret_ty)
        lam.ty=('fn',list(param_tys),resolve(ret_ty,self.subst)); return lam.ty
    def check_expr(self,e,env):
        t=self._check(e,env); e.ty=resolve(t,self.subst); return e.ty
    def _check(self,e,env):
        S=self.subst
        if isinstance(e,N.Lit):
            if e.suffix: return e.suffix
            if isinstance(e.value,bool): return "bool"
            if isinstance(e.value,float): return "f64"
            if isinstance(e.value,str): return "Str"
            return fresh()
        if isinstance(e,N.Var):
            if e.name in env: return env[e.name]
            if e.name in self.ctors:
                tn,fields=self.ctors[e.name]
                if fields: raise CheckError(f"constructor {e.name} needs arguments")
                return tn
            if e.name in self.fns:
                d=self.fns[e.name]; return ('fn',[t for _,t in d.params],d.ret)
            raise CheckError(f"line {e.line}: undefined name {e.name!r}")
        if isinstance(e,N.CtorExpr):
            if e.name not in self.ctors: raise CheckError(f"line {e.line}: unknown constructor {e.name}")
            tn,fields=self.ctors[e.name]
            if len(e.args)!=len(fields): raise CheckError(f"ctor {e.name} arity {len(e.args)} != {len(fields)}")
            for a,ft in zip(e.args,fields): unify(self.check_expr(a,env),ft,S,f"ctor {e.name}")
            return tn
        if isinstance(e,N.TupleExpr):
            return ('tuple',[self.check_expr(x,env) for x in e.items])
        if isinstance(e,N.ListExpr):
            el=fresh()
            for x in e.items: unify(self.check_expr(x,env),el,S,"list literal")
            return ('list',resolve(el,S))
        if isinstance(e,N.SeqExpr):
            t=None
            for x in e.items: t=self.check_expr(x,env)
            return t
        if isinstance(e,N.Block): return self.check_block(e,env)
        if isinstance(e,N.If):
            unify(self.check_expr(e.cond,env),"bool",S,"if condition")
            a=self.check_expr(e.then,env); b=self.check_expr(e.els,env); return unify(a,b,S,"if branches")
        if isinstance(e,N.BinOp):
            lt=self.check_expr(e.l,env); rt=self.check_expr(e.r,env)
            if e.op in ('and','or'): unify(lt,"bool",S); unify(rt,"bool",S); return "bool"
            if e.op in ('==','!='):
                unify(lt,rt,S,"equality"); return "bool"
            if e.op in ('<','<=','>','>='):
                unify(lt,rt,S,"comparison"); return "bool"
            # arithmetic/bitwise
            r=unify(lt,rt,S,f"operator {e.op}")
            r=resolve(r,S)
            if r=='bool': raise CheckError(f"operator {e.op} on bool")
            return r
        if isinstance(e,N.Unary):
            t=self.check_expr(e.e,env)
            return "bool" if e.op=='not' else t
        if isinstance(e,N.Index):
            ot=self.check_expr(e.obj,env); unify(self.check_expr(e.idx,env),"int",S,"index") if False else self.check_expr(e.idx,env)
            ot=resolve(ot,S)
            if isinstance(ot,tuple) and ot[0]=='list': return ot[1]
            raise CheckError(f"indexing non-list {show(ot)}")
        if isinstance(e,N.Lambda):
            pts=[fresh() for _ in e.params]; rt=fresh(); return self.check_lambda(e,pts,rt,env)
        if isinstance(e,N.Match):
            st=self.check_expr(e.scrut,env); rt=fresh()
            for pat,blk in e.arms:
                env2=self.pattern_env(pat,st,env); t=self.check_block(blk,env2); unify(t,rt,S,"match arm")
            return resolve(rt,S)
        if isinstance(e,N.TreeFold):
            st=resolve(self.check_expr(e.scrut,env),S)
            if not (isinstance(st,str) and st in self.types): raise CheckError(f"treefold over non-algebraic type {show(st)}")
            rt=fresh()
            for pat,blk in e.arms:
                if not isinstance(pat,N.PCtor): raise CheckError("treefold arm must be constructor pattern")
                tn,fields=self.ctors[pat.name]
                env2=dict(env)
                if len(pat.items)!=len(fields): raise CheckError("treefold arm arity")
                for p,ft in zip(pat.items,fields):
                    self.bind_pattern(p, rt if ft==st else ft, env2)
                t=self.check_block(blk,env2); unify(t,rt,S,"treefold arm")
            return resolve(rt,S)
        if isinstance(e,N.Call): return self.check_call(e,env)
        raise CheckError(f"cannot check {type(e).__name__}")
    def check_call(self,e,env):
        S=self.subst
        # builtin / named function via Var
        if isinstance(e.f,N.Var):
            name=e.f.name
            if name in ('map','zip','range','len','fold','scan','min','max','clamp','print','write','read_i64s','read_ints','read_u8s'):
                return self.check_builtin(name,e,env)
            if name in self.fns:
                d=self.fns[name]
                if len(e.args)!=len(d.params): raise CheckError(f"{name} arity")
                for a,(_,pt) in zip(e.args,d.params):
                    if isinstance(a,N.Lambda): self.check_lambda(a,[resolve(t,S) if False else t for t in self._fn_arg_types(a,pt)],self._lam_ret(pt),env) if False else self.check_expr(a,env)
                    else: unify(self.check_expr(a,env),pt,S,f"arg of {name}")
                return d.ret
            if name in env:
                ft=resolve(env[name],S)
                argts=[self.check_expr(a,env) for a in e.args]
                rt=fresh()
                if isinstance(ft,tuple) and ft[0]=='fn': unify(ft,('fn',argts,rt),S,"call")
                return resolve(rt,S)
        if isinstance(e.f,N.CtorExpr): return self.check_expr(e.f,env)  # unreachable
        ft=self.check_expr(e.f,env); argts=[self.check_expr(a,env) for a in e.args]; rt=fresh()
        if isinstance(resolve(ft,S),tuple) and resolve(ft,S)[0]=='fn': unify(ft,('fn',argts,rt),S,"call")
        return resolve(rt,S)
    def _fn_arg_types(self,a,pt): return []
    def _lam_ret(self,pt): return fresh()
    def check_builtin(self,name,e,env):
        S=self.subst; args=e.args
        def need(n):
            if len(args)!=n: raise CheckError(f"{name} expects {n} args, got {len(args)}")
        if name=='len':
            need(1); t=self.check_expr(args[0],env); return "int"
        if name=='range':
            need(2); self.check_expr(args[0],env); self.check_expr(args[1],env); return ('list',"int")
        if name=='zip':
            need(2); a=self.check_expr(args[0],env); b=self.check_expr(args[1],env)
            a=resolve(a,S); b=resolve(b,S)
            ea=a[1] if isinstance(a,tuple) and a[0]=='list' else fresh()
            eb=b[1] if isinstance(b,tuple) and b[0]=='list' else fresh()
            return ('list',('tuple',[ea,eb]))
        if name=='map':
            need(2); xt=self.check_expr(args[0],env); xt=resolve(xt,S)
            el=xt[1] if isinstance(xt,tuple) and xt[0]=='list' else fresh(); rt=fresh()
            if isinstance(args[1],N.Lambda): self.check_lambda(args[1],[el],rt,env)
            else: self.check_expr(args[1],env)
            return ('list',resolve(rt,S))
        if name in ('fold','scan'):
            need(3); xt=self.check_expr(args[0],env); xt=resolve(xt,S)
            el=xt[1] if isinstance(xt,tuple) and xt[0]=='list' else fresh()
            st=self.check_expr(args[1],env)
            if isinstance(args[2],N.Lambda): self.check_lambda(args[2],[st,el],st,env)
            else: self.check_expr(args[2],env)
            return resolve(st,S) if name=='fold' else ('list',resolve(st,S))
        if name in ('min','max'):
            need(2); a=self.check_expr(args[0],env); b=self.check_expr(args[1],env); return unify(a,b,S,name)
        if name=='clamp':
            need(3); a=self.check_expr(args[0],env); self.check_expr(args[1],env); self.check_expr(args[2],env); return a
        if name=='print':
            need(1); self.check_expr(args[0],env); return "Out"
        if name=='write':
            need(2); self.check_expr(args[0],env); self.check_expr(args[1],env); return "Out"
        if name in ('read_i64s',): need(1); self.check_expr(args[0],env); return ('list','i64')
        if name in ('read_ints',): need(1); self.check_expr(args[0],env); return ('list','int')
        if name in ('read_u8s',): need(1); self.check_expr(args[0],env); return ('list','u8')
        raise CheckError(name)

def check_program(prog):
    Checker(prog).check(); return prog

def is_finite_type(ty,types,finite_ctx=False,seen=None):
    seen=seen or set()
    if ty=="bool": return True
    if isinstance(ty,str) and ty in MACHINE: return finite_ctx
    if isinstance(ty,tuple) and ty:
        if ty[0]=='tuple': return all(is_finite_type(x,types,finite_ctx,seen) for x in ty[1])
        if ty[0]=='list': return False
    if isinstance(ty,str) and ty in types:
        if ty in seen: return True
        seen=seen|{ty}
        return all(is_finite_type(ft,types,finite_ctx,seen) for _,fields in types[ty] for ft in fields)
    return False
