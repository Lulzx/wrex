from . import nodes as N
from .values import Trap, ADT, wrap
from . import families as F
from .checker import is_finite_type, resolve

INF=F.INF; NINF=F.NINF

# Sym: tuples ('var',idx) ('nleaf',name) ('leaf',ast) ('bin',op,l,r,ty) ('un',op,e,ty)
#      ('if',c,t,e) ('call',name,[args],ty) ('tuple',[..]) ('inf',sign)
def sym_vars(s):
    if s[0]=='var': return {s[1]}
    out=set()
    for x in s[1:]:
        if isinstance(x,tuple): out|=sym_vars(x)
        elif isinstance(x,list):
            for y in x: out|=sym_vars(y)
    return out

def eval_sym(interp,s,env,comp):
    k=s[0]
    if k=='proj': return eval_sym(interp,s[1],env,comp)[s[2]]
    if k=='var': return comp[s[1]]
    if k=='nleaf': return env[s[1]]
    if k=='leaf': return interp.eval_expr(s[1],env)
    if k=='inf': return INF if s[1]>0 else NINF
    if k=='tuple': return tuple(eval_sym(interp,x,env,comp) for x in s[1])
    if k=='if':
        return eval_sym(interp,s[2],env,comp) if eval_sym(interp,s[1],env,comp) else eval_sym(interp,s[3],env,comp)
    if k=='un':
        v=eval_sym(interp,s[2],env,comp)
        if s[1]=='not': return not v
        r=-v; return wrap(r,s[3]) if isinstance(s[3],str) else r
    if k=='call':
        vals=[eval_sym(interp,x,env,comp) for x in s[2]]
        if s[1]=='min': return min(vals)
        if s[1]=='max': return max(vals)
        if s[1]=='clamp':
            v,lo,hi=vals
            if lo>hi: raise Trap("clamp_bounds","clamp with lo > hi")
            return min(hi,max(lo,v))
        raise Trap("infer","unsupported call in coefficient")
    if k=='bin':
        # build a fake BinOp to reuse interpreter semantics (incl. traps/wrap)
        node=N.BinOp(op=s[1],line=1); node.ty=s[4]
        lv=eval_sym(interp,s[2],env,comp); rv=eval_sym(interp,s[3],env,comp)
        # replicate eval_binop on values
        op=s[1]
        if op=='and': return bool(lv and rv)
        if op=='or': return bool(lv or rv)
        if op=='==': return lv==rv
        if op=='!=': return lv!=rv
        if op=='<': return lv<rv
        if op=='<=': return lv<=rv
        if op=='>': return lv>rv
        if op=='>=': return lv>=rv
        if op in ('/','%'):
            if rv==0: raise Trap("div_by_zero","division by zero")
            q=abs(lv)//abs(rv)
            if (lv<0)!=(rv<0): q=-q
            res=q if op=='/' else lv-q*rv
            from .values import is_machine
            return wrap(res,s[4]) if is_machine(s[4]) else res
        if op=='+': res=lv+rv
        elif op=='-': res=lv-rv
        elif op=='*': res=lv*rv
        elif op=='&': res=int(lv)&int(rv)
        elif op=='|': res=int(lv)|int(rv)
        elif op=='^': res=int(lv)^int(rv)
        elif op=='<<': res=int(lv)<<int(rv)
        elif op=='>>': res=int(lv)>>int(rv)
        else: raise Trap("infer",f"bad op {op}")
        from .values import is_machine
        return wrap(res,s[4]) if is_machine(s[4]) else res
    raise Trap("infer",f"bad sym {k}")

ZERO=('leaf',N.Lit(value=0,line=1)); ONE=('leaf',N.Lit(value=1,line=1))
def coeff_bin(op,a,b): return ('bin',op,a,b,None)

class Obstruction:
    def __init__(self,reason,detail=""): self.reason=reason; self.detail=detail
    def __repr__(self): return f"Obstruction({self.reason})"

def flatten_pattern(pat,ty,flat,skeleton):
    """flat: list of (name,ty); skeleton: ('leaf',idx)|('tuple',[skel])|('ignore',ty)"""
    if isinstance(pat,N.PVar):
        idx=len(flat); flat.append((pat.name,ty)); return ('leaf',idx)
    if isinstance(pat,N.PWild):
        idx=len(flat); flat.append((None,ty)); return ('leaf',idx)
    if isinstance(pat,N.PTuple):
        tys=ty[1] if isinstance(ty,tuple) and ty and ty[0]=='tuple' else [None]*len(pat.items)
        return ('tuple',[flatten_pattern(p,t,flat,skeleton) for p,t in zip(pat.items,tys)])
    # ctor / lit pattern for state: treat as single named-less component
    idx=len(flat); flat.append((None,ty)); return ('leaf',idx)

def assemble(skel,vals):
    if skel[0]=='leaf': return vals[skel[1]]
    return tuple(assemble(s,vals) for s in skel[1])
def decompose(skel,val,out):
    if skel[0]=='leaf': out[skel[1]]=val; return
    for s,v in zip(skel[1],val): decompose(s,v,out)

def count_values(ty,types,finite_ctx,cap=257):
    if ty=="bool": return 2
    if isinstance(ty,str) and ty in F.__dict__.get('MACHINE',{}): pass
    from .values import MACHINE
    if isinstance(ty,str) and ty in MACHINE: return cap if not finite_ctx else (1<<MACHINE[ty][0]) if (1<<MACHINE[ty][0])<=cap else cap
    if isinstance(ty,tuple) and ty and ty[0]=='tuple':
        n=1
        for t in ty[1]:
            n*=count_values(t,types,finite_ctx,cap)
            if n>=cap: return cap
        return n
    if isinstance(ty,str) and ty in types:
        total=0
        for _,fields in types[ty]:
            n=1
            for ft in fields: n*=count_values(ft,types,finite_ctx,cap)
            total+=n
            if total>=cap: return cap
        return total
    return cap

def enum_values(ty,types,interp_types=None):
    if ty=="bool": return [False,True]
    from .values import MACHINE
    if isinstance(ty,str) and ty in MACHINE: return list(range(1<<MACHINE[ty][0])) if (1<<MACHINE[ty][0])<=256 else []
    if isinstance(ty,tuple) and ty and ty[0]=='tuple':
        pools=[enum_values(t,types) for t in ty[1]]; out=[()]
        for p in pools: out=[o+(v,) for o in out for v in p]
        return out
    if isinstance(ty,str) and ty in types:
        out=[]
        for cname,fields in types[ty]:
            pools=[enum_values(ft,types) for ft in fields]; combos=[()]
            for p in pools: combos=[o+(v,) for o in combos for v in p]
            for combo in combos: out.append(ADT(cname,combo,ty))
        return out
    return []

# ---------- AST -> Sym
class SymBuilder:
    def __init__(self,state_vars,coeff_names):
        self.state_vars=state_vars  # name->global comp idx
        self.coeff_names=set(coeff_names)
    def build_block(self,blk,local):
        local=dict(local)
        for pat,e in blk.bindings:
            s=self.build(e,local)
            self.bind_sym(pat,s,local)
        return self.build(blk.expr,local)
    def bind_sym(self,pat,s,local):
        if isinstance(pat,N.PVar): local[pat.name]=s
        elif isinstance(pat,N.PTuple):
            if s[0]=='tuple':
                for p,x in zip(pat.items,s[1]): self.bind_sym(p,x,local)
            else:
                for i,p in enumerate(pat.items): self.bind_sym(p,('proj',s,i),local)
    def build(self,e,local):
        if isinstance(e,N.Block): return self.build_block(e,local)
        if isinstance(e,N.Var):
            if e.name in local: return local[e.name]
            if e.name in self.state_vars: return ('var',self.state_vars[e.name])
            return ('leaf',e)
        if isinstance(e,N.Lit): return ('leaf',e)
        if isinstance(e,N.TupleExpr): return ('tuple',[self.build(x,local) for x in e.items])
        if isinstance(e,N.BinOp): return ('bin',e.op,self.build(e.l,local),self.build(e.r,local),e.ty)
        if isinstance(e,N.Unary): return ('un',e.op,self.build(e.e,local),e.ty)
        if isinstance(e,N.If): return ('if',self.build(e.cond,local),self.build(e.then,local),self.build(e.els,local))
        if isinstance(e,N.Match):
            scrut=self.build(e.scrut,local)
            if sym_vars(scrut): return ('unsupported',e)
            result=None
            for pat,blk in reversed(e.arms):
                local2=dict(local)
                if isinstance(pat,N.PWild):
                    cond=None
                elif isinstance(pat,N.PVar):
                    local2[pat.name]=scrut; cond=None
                elif isinstance(pat,N.PLit):
                    cond=('bin','==',scrut,('leaf',N.Lit(value=pat.value,line=e.line)),None)
                elif isinstance(pat,N.PCtor) and not pat.items:
                    cond=('bin','==',scrut,('leaf',N.CtorExpr(name=pat.name,args=[],line=e.line)),None)
                else:
                    return ('unsupported',e)
                arm=self.build_block(blk,local2)
                result=arm if cond is None else ('if',cond,arm,result if result is not None else arm)
            return result if result is not None else ('unsupported',e)
        if isinstance(e,N.Call) and isinstance(e.f,N.Var) and e.f.name in ('min','max','clamp'):
            return ('call',e.f.name,[self.build(a,local) for a in e.args],e.ty)
        if isinstance(e,N.TupleExpr): return ('tuple',[self.build(x,local) for x in e.items])
        # anything else: coeff if no state vars inside, else unsupported marker as leaf (poly will fail via vars check)
        s=('leaf',e)
        # check state dependence by scanning var names
        names=set()
        def walk(x):
            if isinstance(x,N.Var) and x.name in self.state_vars and x.name not in local: names.add(x.name)
            for v in vars(x).values():
                if isinstance(v,N.Node): walk(v)
                elif isinstance(v,list):
                    for y in v:
                        if isinstance(y,N.Node): walk(y)
        walk(e)
        if names: return ('unsupported',e)
        return s

def trap_state_dependent(e,state_names,local_names=()):
    """Return description if a trapping op depends on numeric state (spec 7.3)."""
    state_names=set(state_names)-set(local_names)
    def dep(x):
        if isinstance(x,N.Var): return x.name in state_names
        return any(dep(v) for v in vars(x).values() if isinstance(v,N.Node)) or \
               any(dep(y) for v in vars(x).values() if isinstance(v,list) for y in v if isinstance(y,N.Node))
    def walk(x):
        if isinstance(x,N.BinOp) and x.op in ('/','%') and dep(x.r): return "divisor of `/`/`%` depends on the numeric state"
        if isinstance(x,N.Index) and dep(x.idx): return "array index depends on the numeric state"
        if isinstance(x,N.Call) and isinstance(x.f,N.Var) and x.f.name=='clamp' and (dep(x.args[1]) or dep(x.args[2])): return "`clamp` bounds depend on the numeric state"
        if isinstance(x,N.Match) and dep(x.scrut): return "`match` scrutinee depends on the numeric state"
        for v in vars(x).values():
            if isinstance(v,N.Node):
                r=walk(v)
                if r: return r
            elif isinstance(v,list):
                for y in v:
                    if isinstance(y,N.Node):
                        r=walk(y)
                        if r: return r
        return None
    body=e.body if isinstance(e,N.Lambda) else e
    return walk(body) if isinstance(body,N.Node) else None

# ---------- polynomials
class NotNormal(Exception): pass
def sym_to_poly(s):
    k=s[0]
    if k=='var': return {(s[1],):ONE}
    if k in ('leaf','nleaf','inf'): return {():s}
    if k=='unsupported': raise NotNormal("unsupported subterm")
    if k=='bin':
        if s[1] in ('+','-'):
            a=sym_to_poly(s[2]); b=sym_to_poly(s[3])
            out=dict(a)
            for m,c in b.items():
                nc=coeff_bin('-',ZERO,c) if s[1]=='-' else c
                # for '-': poly b negated
                if s[1]=='-': nc=('un','-',c,None)
                out[m]=coeff_bin('+',out[m],nc) if m in out else nc
            return out
        if s[1]=='*':
            a=sym_to_poly(s[2]); b=sym_to_poly(s[3]); out={}
            for m1,c1 in a.items():
                for m2,c2 in b.items():
                    m=tuple(sorted(m1+m2)); c=coeff_bin('*',c1,c2)
                    out[m]=coeff_bin('+',out[m],c) if m in out else c
            return out
        raise NotNormal(f"operator {s[1]}")
    if k=='un' and s[1]=='-':
        return {m:('un','-',c,None) for m,c in sym_to_poly(s[2]).items()}
    if k=='if':
        if sym_vars(s[1]): raise NotNormal("branch condition depends on the numeric state")
        a=sym_to_poly(s[2]); b=sym_to_poly(s[3]); out={}
        for m in set(a)|set(b):
            ca=a.get(m,ZERO); cb=b.get(m,ZERO); out[m]=('if',s[1],ca,cb)
        return out
    raise NotNormal(k)

def render_monomial(m,names):
    if not m: return "1"
    parts=[]
    for i in m: parts.append(names[i] if i < len(names) else f"s{i}")
    return " * ".join(parts)

# ---------- clamp form
def to_clamp(s):
    k=s[0]
    if k=='var': return (ZERO,('inf',-1),('inf',1))
    if k=='bin' and s[1] in ('+','-'):
        # e +/- coeff
        for a,b,sgn in ((s[2],s[3],1),(s[3],s[2],1)):
            if not sym_vars(b) and s[1]=='+' or (a is s[2] and not sym_vars(b)):
                pass
        if not sym_vars(s[3]):
            a,l,h=to_clamp(s[2]); c=s[3] if s[1]=='+' else ('un','-',s[3],None)
            return (coeff_bin('+',a,c), coeff_bin('+',l,c) if l[0]!='inf' else l, coeff_bin('+',h,c) if h[0]!='inf' else h)
        if not sym_vars(s[2]) and s[1]=='+':
            a,l,h=to_clamp(s[3]); return (coeff_bin('+',a,s[2]), coeff_bin('+',l,s[2]) if l[0]!='inf' else l, coeff_bin('+',h,s[2]) if h[0]!='inf' else h)
        raise NotNormal("clamp add")
    if k=='call':
        if s[1]=='max':
            a,b=s[2]
            if not sym_vars(b): a1,l,h=to_clamp(a); return (a1, ('call','max',[l,b],None), h)
            if not sym_vars(a): a1,l,h=to_clamp(b); return (a1, ('call','max',[l,a],None), h)
        if s[1]=='min':
            a,b=s[2]
            if not sym_vars(b): a1,l,h=to_clamp(a); return (a1,l,('call','min',[h,b],None))
            if not sym_vars(a): a1,l,h=to_clamp(b); return (a1,l,('call','min',[h,a],None))
        if s[1]=='clamp':
            x,lo,hi=s[2]
            if sym_vars(lo) or sym_vars(hi): raise NotNormal("clamp bounds state-dependent")
            a,l,h=to_clamp(x)
            nl=('call','min',[('call','max',[l,lo],None),hi],None); nh=('call','max',[('call','min',[h,hi],None),lo],None)
            return (a,nl,nh)
        raise NotNormal("clamp call")
    if k=='if':
        if sym_vars(s[1]): raise NotNormal("branch condition depends on the numeric state (not a clamp form)")
        a1,l1,h1=to_clamp(s[2]); a2,l2,h2=to_clamp(s[3])
        return (('if',s[1],a1,a2),('if',s[1],l1,l2),('if',s[1],h1,h2))
    raise NotNormal(f"clamp form {k}")

# ---------- tropical form
def to_tropical(s,mode):
    # returns (dict var->coeff, const coeff); operation is max (or min) of var+coeff
    k=s[0]; combine='max' if mode=='maxplus' else 'min'
    if k=='var': return ({s[1]:ZERO},('inf',-1) if mode=='maxplus' else ('inf',1))
    if not sym_vars(s): return ({},s)
    if k=='bin' and s[1]=='+':
        # one side coeff shift
        if not sym_vars(s[3]):
            d,c=to_tropical(s[2],mode); return ({i:coeff_bin('+',v,s[3]) for i,v in d.items()}, coeff_bin('+',c,s[3]) if c[0]!='inf' else c)
        if not sym_vars(s[2]):
            d,c=to_tropical(s[3],mode); return ({i:coeff_bin('+',v,s[2]) for i,v in d.items()}, coeff_bin('+',c,s[2]) if c[0]!='inf' else c)
        raise NotNormal("tropical add")
    if k=='call' and s[1]==combine:
        d1,c1=to_tropical(s[2][0],mode); d2,c2=to_tropical(s[2][1],mode); out={}
        for i in set(d1)|set(d2):
            if i in d1 and i in d2: out[i]=('call',combine,[d1[i],d2[i]],None)
            else: out[i]=d1.get(i,d2.get(i))
        cc=('call',combine,[c1,c2],None)
        return (out,cc)
    if k=='if':
        if sym_vars(s[1]): raise NotNormal("branch condition depends on state")
        d1,c1=to_tropical(s[2],mode); d2,c2=to_tropical(s[3],mode); out={}
        empty=('inf',-1) if mode=='maxplus' else ('inf',1)
        for i in set(d1)|set(d2): out[i]=('if',s[1],d1.get(i,empty),d2.get(i,empty))
        return (out,('if',s[1],c1,c2))
    raise NotNormal(f"tropical {k}")

# ---------- Certificate
class Cert:
    def __init__(self,kind,family,scope,**kw):
        self.kind=kind; self.family=family; self.scope=list(scope)
        self.__dict__.update(kw)
        self.cost=family.cost; self.dim=family.dim
    def describe(self): return self.family.describe()
    # scope-local state values list <-> family state
    def rho_env(self,interp,env,idx,step):
        kind=self.kind
        if kind in ('F0','F1','F2','F3'):
            vals=[eval_sym(interp,c,env,{}) for c in self.coeffs]
            if kind=='F0': return vals[0]
            if kind=='F1': return (vals[0],vals[1])
            if kind=='F3': return tuple(vals)
            if kind=='F2':
                n=self.family.k+1; return [vals[i*n:(i+1)*n] for i in range(n)]
        if kind=='F4':
            out=[]
            for q in self.qvalues:
                qs=q if len(self.scope)>1 else q
                try: out.append(step(qs))
                except Trap as t: out.append(F.TrapMark(idx,t.kind,t.msg))
            return tuple(out)
        if kind=='F5':
            out=[]
            for q in self.qvalues:
                comp={}; 
                qvals=list(q) if isinstance(q,tuple) else [q]
                for idx_c,v in zip(self.ctrl_scope,qvals): comp[idx_c]=v
                env2=dict(env)
                for name_c,v in zip(self.ctrl_names,qvals): env2[name_c]=v
                try: qnew=tuple(eval_sym(interp,s,env2,comp) for s in self.ctrl_syms)
                except Trap as t: out.append(F.TrapMark(idx,t.kind,t.msg)); continue
                try: payload=self.payload.rho_env(interp,env2,idx,step)
                except Trap as t: out.append(F.TrapMark(idx,t.kind,t.msg)); continue
                out.append((qnew[0] if len(qnew)==1 else qnew,payload))
            return tuple(out)
        if kind=='F6':
            return tuple(sub.rho_env(interp,env,idx,step) for sub in self.subs)
        raise Trap("infer","bad cert")
    def ev_state(self,elem,comp_list):
        """comp_list: values aligned with self.scope -> new values aligned with scope"""
        kind=self.kind
        if kind in ('F0','F1','F3'): return [self.family.ev(elem,comp_list[0])]
        if kind=='F2': return list(self.family.ev(elem,tuple(comp_list)))
        if kind=='F4':
            q=comp_list[0] if len(comp_list)==1 else tuple(comp_list)
            r=self.family.ev(elem,q)
            return [r] if len(comp_list)==1 else list(r)
        if kind=='F5':
            ctrl=[comp_list[self.scope.index(i)] for i in self.ctrl_scope]
            pay=[comp_list[self.scope.index(i)] for i in self.payload.scope]
            q=ctrl[0] if len(ctrl)==1 else tuple(ctrl)
            entry=elem[self.family.index[q]]
            if isinstance(entry,F.TrapMark): raise Trap(entry.kind,entry.msg)
            q2,m=entry
            newpay=self.payload.ev_state(m,pay)
            out=list(comp_list)
            for i,v in zip(self.ctrl_scope, [q2] if not isinstance(q2,tuple) else list(q2)): out[self.scope.index(i)]=v
            for i,v in zip(self.payload.scope,newpay): out[self.scope.index(i)]=v
            return out
        if kind=='F6':
            out=list(comp_list)
            for sub,e in zip(self.subs, elem if not isinstance(elem,F.TrapMark) else []):
                subvals=[comp_list[self.scope.index(i)] for i in sub.scope]
                new=sub.ev_state(e,subvals)
                for i,v in zip(sub.scope,new): out[self.scope.index(i)]=v
            return out
        raise Trap("infer","bad cert ev")

def remap_sym(s,mapping):
    # mapping: global idx -> sym replacement for that var
    if s[0]=='var': return mapping.get(s[1],s)
    out=[s[0]]
    for x in s[1:]:
        if isinstance(x,tuple): out.append(remap_sym(x,mapping))
        elif isinstance(x,list): out.append([remap_sym(y,mapping) for y in x])
        else: out.append(x)
    return tuple(out)

def _l3_for(kind):
    return {'F0':"discharged by ring normalization",'F1':"discharged by ring normalization",
            'F2':"discharged by ring normalization",'F3':"discharged by lattice normalization",
            'F4':"holds by construction (tabulation)",'F5':"construction for control; payload checked per control state",
            'F6':"discharged componentwise"}[kind]

def infer_numeric_group(syms,scope,tys,names,trap_reason,reassoc=False):
    if trap_reason: return Obstruction(trap_reason+" (numeric families rejected before simplification, spec 7.3)")
    if any(t=="f64" for t in tys) and not reassoc: return Obstruction("`f64` state: floating-point arithmetic is not associative, so no family applies (Section 11). Add @reassoc to opt into reassociated floating-point plans, which leave the plan-independence guarantee.")
    if len(scope)==1:
        i=scope[0]; s=syms[0]; ty=tys[0]
        # F0
        fam=None; g=None
        if s[0]=='bin' and s[1] in ('+','*','&','|','^','and','or'):
            if s[2]==('var',i) and not sym_vars(s[3]): fam,g=s[1],s[3]
            elif s[3]==('var',i) and not sym_vars(s[2]): fam,g=s[1],s[2]
        if s[0]=='call' and s[1] in ('min','max'):
            a,b=s[2]
            if a==('var',i) and not sym_vars(b): fam,g=s[1],b
            elif b==('var',i) and not sym_vars(a): fam,g=s[1],a
        if fam is not None:
            c=Cert('F0',F.F0(fam,ty if isinstance(ty,str) else "int"),scope,coeffs=[g]); c.l3=_l3_for('F0'); return c
        # F1 via poly
        degree_obs=None
        try:
            poly=sym_to_poly(s)
            bad=[m for m in poly if len(m)>1]
            if bad:
                degree_obs=Obstruction(f"The new value contains `{render_monomial(bad[0],names)}`, which has degree {len(bad[0])} in the state. Polynomials of degree 2 or more are not closed under composition.")
            elif set(poly)<= {(),(i,)}:
                c=Cert('F1',F.F1(ty if isinstance(ty,str) else "int"),scope,coeffs=[poly.get((i,),ZERO),poly.get((),ZERO)]); c.l3=_l3_for('F1'); return c
        except NotNormal as ex: degree_obs=Obstruction(f"step is not affine in the state ({ex})")
        # F3
        try:
            a,l,h=to_clamp(s)
            c=Cert('F3',F.F3(),scope,coeffs=[a,l,h]); c.l3=_l3_for('F3'); return c
        except NotNormal: pass
        return degree_obs or Obstruction("step is neither affine nor a shifted clamp in the state")
    # coupled: F2 ring
    if len(scope)>8: return Obstruction(f"coupled state dimension {len(scope)} exceeds the F2 limit k <= 8")
    polys=[]; degree_obs=None
    try:
        for s in syms:
            p=sym_to_poly(s); polys.append(p)
            bad=[m for m in p if len(m)>1]
            if bad: degree_obs=Obstruction(f"The new value contains `{render_monomial(bad[0],names)}`, which has degree {len(bad[0])} in the state. Polynomials of degree 2 or more are not closed under composition.")
        if degree_obs is None:
            k=len(scope); coeffs=[]
            for r in range(k):
                for j in scope: coeffs.append(polys[r].get((j,),ZERO))
                coeffs.append(polys[r].get((),ZERO))
            for j in range(k): coeffs.append(ZERO)
            coeffs.append(ONE)
            ty=tys[0] if isinstance(tys[0],str) else "int"
            c=Cert('F2',F.F2(k,"ring",ty),scope,coeffs=coeffs); c.l3=_l3_for('F2'); return c
    except NotNormal as ex: degree_obs=degree_obs or Obstruction(f"step is not linear in the state ({ex})")
    # tropical F2 (int only)
    if all(t=="int" for t in tys):
        for mode in ("maxplus","minplus"):
            try:
                k=len(scope); coeffs=[]
                for s in syms:
                    d,const=to_tropical(s,mode)
                    for j in scope: coeffs.append(d.get(j,('inf',-1) if mode=='maxplus' else ('inf',1)))
                    coeffs.append(const)
                empty=('inf',-1) if mode=='maxplus' else ('inf',1)
                for j in range(k): coeffs.append(empty)
                coeffs.append(ZERO)
                c=Cert('F2',F.F2(k,mode,"int"),scope,coeffs=coeffs); c.l3="discharged by lattice normalization"; return c
            except NotNormal: continue
    return degree_obs or Obstruction("coupled state is neither ring-linear nor tropical-linear")

def make_f4(scope,tys,types,finite_ctx):
    total=1
    for t in tys: total*=count_values(t,types,finite_ctx)
    if total>256: return Obstruction(f"finite state has {total} values, exceeding the F4 limit |Q| <= 256")
    if len(scope)==1: qvalues=enum_values(tys[0],types)
    else: qvalues=enum_values(('tuple',tys),types)
    c=Cert('F4',F.F4(qvalues),scope,qvalues=qvalues); c.l3=_l3_for('F4'); return c

def infer_group(syms,scope,tys,names,types,finite_ctx,trap_reason,reassoc=False):
    finite=[is_finite_type(t,types,finite_ctx) for t in tys]
    # numeric attempt when no user/bool control types present (bool handled by F0 too)
    if not any(finite) or all(isinstance(t,str) and (t in ("int","f64") or t in __import__('wrex.values',fromlist=['MACHINE']).MACHINE) for t in tys):
        r=infer_numeric_group(syms,scope,tys,names,trap_reason,reassoc)
        if isinstance(r,Cert): return r
        if all(finite):
            f4=make_f4(scope,tys,types,finite_ctx)
            return f4 if isinstance(f4,Cert) else r
        return r
    if all(finite):
        # try cheap numeric for single bool/int control (e.g. bool with and/or)
        r=infer_numeric_group(syms,scope,tys,names,None,reassoc) if len(scope)==1 and tys[0]=="bool" else None
        if isinstance(r,Cert): return r
        return make_f4(scope,tys,types,finite_ctx)
    # mixed control + payload (F5)
    ctrl=[k for k,f in enumerate(finite) if f]; pay=[k for k,f in enumerate(finite) if not f]
    ctrl_scope=[scope[k] for k in ctrl]; pay_scope=[scope[k] for k in pay]
    for k in ctrl:
        deps=sym_vars(syms[k]) & set(pay_scope)
        if deps:
            return Obstruction(f"The control state `{names[scope[k]] or 'state'}` depends on the numeric state `{names[list(deps)[0]] or 'state'}`. Finite-control families require the control state to depend only on itself and the input.")
    mapping={scope[k]:('nleaf',names[scope[k]]) for k in ctrl}
    pay_syms=[remap_sym(syms[k],mapping) for k in pay]
    payload=infer_group(pay_syms,pay_scope,[tys[k] for k in pay],names,types,finite_ctx,trap_reason,reassoc)
    if not isinstance(payload,Cert): return payload
    if payload.kind not in ('F0','F1','F2','F3','F6'):
        return Obstruction(f"payload family {payload.kind} is not admitted in F5 (spec 6.6 allows F0-F3 and F6)")
    ctrl_tys=[tys[k] for k in ctrl]
    if len(ctrl_scope)==1: qvalues=enum_values(ctrl_tys[0],types)
    else: qvalues=enum_values(('tuple',ctrl_tys),types)
    if len(qvalues)>256: return Obstruction("control state exceeds |Q| <= 256 for F5")
    fam=F.F5(qvalues,payload.family)
    c=Cert('F5',fam,scope,ctrl_scope=ctrl_scope,payload=payload,ctrl_syms=[syms[k] for k in ctrl],
           ctrl_names=[names[scope[k]] for k in ctrl],qvalues=qvalues)
    c.payload_scope=pay_scope; c.l3=_l3_for('F5'); return c

def infer_groups(syms,scope,tys,names,types,finite_ctx,trap_reason,reassoc=False):
    # connected components of dependency graph
    parent=list(range(len(scope)))
    def find(a):
        while parent[a]!=a: parent[a]=parent[parent[a]]; a=parent[a]
        return a
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra!=rb: parent[ra]=rb
    pos={g:k for k,g in enumerate(scope)}
    for k,s in enumerate(syms):
        for v in sym_vars(s):
            if v in pos: union(k,pos[v])
    groups={}
    for k in range(len(scope)): groups.setdefault(find(k),[]).append(k)
    if len(groups)==1:
        return infer_group(syms,scope,tys,names,types,finite_ctx,trap_reason,reassoc)
    subs=[]
    for ks in groups.values():
        r=infer_group([syms[k] for k in ks],[scope[k] for k in ks],[tys[k] for k in tys and ks],names,types,finite_ctx,trap_reason,reassoc)
        if not isinstance(r,Cert):
            if all(is_finite_type(t,types,finite_ctx) for t in tys):
                return make_f4(scope,tys,types,finite_ctx)
            return r
        subs.append(r)
    spans=[[scope.index(i) for i in sub.scope] for sub in subs]
    fam=F.F6([s.family for s in subs],spans)
    c=Cert('F6',fam,scope,subs=subs); c.l3=_l3_for('F6'); return c

def pattern_names(pat,out):
    if isinstance(pat,N.PVar): out.append(pat.name)
    elif isinstance(pat,(N.PTuple,N.PCtor)):
        for p in pat.items: pattern_names(p,out)

def infer_step(step_node,state_ty,input_ty,fns,types,finite_ctx,reassoc=False):
    """Returns Cert or Obstruction."""
    if isinstance(step_node,N.Var) and step_node.name in fns:
        d=fns[step_node.name]; params=d.params
        # params are (name,type): build var patterns
        pats=[N.PVar(name=n) for n,_ in params]; body=d.body
    elif isinstance(step_node,N.Lambda):
        pats=step_node.params; body=step_node.body
    else:
        return Obstruction("step is not a lambda or named function; representation inference requires an inline step")
    if len(pats)!=2: return Obstruction("step must take (state, input)")
    flat=[]; skel=flatten_pattern(pats[0],state_ty,flat,None)
    state_vars={name:i for i,(name,_) in enumerate(flat) if name}
    names=[n for n,_ in flat]; tys=[t for _,t in flat]
    builder=SymBuilder(state_vars,())
    bodysym=builder.build_block(body,{}) if isinstance(body,N.Block) else builder.build(body,{})
    def flatten_sym(skel,s):
        if skel[0]=='leaf': return [s]
        if s[0]=='if':
            a=flatten_sym(skel,s[2]); b=flatten_sym(skel,s[3])
            if a is None or b is None: return None
            return [('if',s[1],x,y) for x,y in zip(a,b)]
        if s[0]!='tuple': return None
        out=[]
        for sub,ss in zip(skel[1],s[1]):
            r=flatten_sym(sub,ss)
            if r is None: return None
            out+=r
        return out
    syms=flatten_sym(skel,bodysym)
    if syms is None or len(syms)!=len(flat): return Obstruction("step does not return a state of the expected shape")
    trap_reason=trap_state_dependent(step_node,state_vars.keys())
    cert=infer_groups(syms,list(range(len(flat))),tys,names,types,finite_ctx,trap_reason,reassoc)
    if isinstance(cert,Cert):
        cert.state_skel=skel; cert.flat_tys=tys; cert.state_names=names
        cert.step_node=step_node; cert.input_pat=pats[1]; cert.reassoc=bool(reassoc and any(t=='f64' for t in tys))
    return cert

# ---------- tree contexts
def infer_treefold(node,types,fns):
    """Returns (ctx Cert-like or None, obstruction or None). Cert per (ctor,pos)."""
    T=node.scrut.ty; V=node.ty
    if not isinstance(T,str) or T not in types: return None,Obstruction("treefold scrutinee is not an algebraic type")
    # result may be a tuple: the hole is then a tuple of state components
    if isinstance(V,tuple) and V and V[0]=='tuple':
        vtys=list(V[1]); m=len(vtys)
    else:
        vtys=[V]; m=1
    per={}
    # try common F1 first, then F3: infer each context with numeric group and check family match
    candidates=[]
    for pat,blk in node.arms:
        if not isinstance(pat,N.PCtor): continue
        fields=dict((c,f) for c,f in types[T])[pat.name] if False else None
        ctor_fields=None
        for cname,fs in types[T]:
            if cname==pat.name: ctor_fields=fs
        for j,(p,ft) in enumerate(zip(pat.items,ctor_fields)):
            if ft!=T: continue
            if not isinstance(p,N.PVar): return None,Obstruction(f"context for {pat.name} field {j}: recursive field is not a simple variable")
            hole=p.name
            builder=SymBuilder({hole:0} if m==1 else {},())
            # other pattern vars are coeff leaves via Var fallback (not state vars) - good
            seed={} if m==1 else {hole:('tuple',[('var',i) for i in range(m)])}
            s=builder.build_block(blk,seed)
            if m>1:
                if s[0]!='tuple' or len(s[1])!=m: return None,Obstruction(f"context for {pat.name} field {j}: arm does not return a {m}-tuple")
                candidates.append(((pat.name,j),list(s[1])))
            else:
                candidates.append(((pat.name,j),s))
    if not candidates: return None,Obstruction("no recursive fields; contraction is unnecessary")
    if m>1:
        # componentwise inference per context, then a forced common F2 ring form
        certs={}; compatible=True; famkey=None
        for key,syms in candidates:
            r=infer_groups(syms,list(range(m)),vtys,[f"v{i}" for i in range(m)],types,False,None)
            if not isinstance(r,Cert): compatible=False; break
            fk=(r.kind, type(r.family).__name__, getattr(r.family,'semiring',None), getattr(r.family,'k',None), getattr(r.family,'op',None))
            if famkey is None: famkey=fk
            elif fk!=famkey: compatible=False; break
            certs[key]=r
        if compatible and certs:
            fam=next(iter(certs.values())).family
            tc=Cert('CTX',fam,list(range(m)),per=certs); tc.l3="context L3 discharged by normalization (spec 9.2)"; return tc,None
        # forced F2 over a ring for every context
        certs={}; ok=True
        for key,syms in candidates:
            try:
                polys=[sym_to_poly(s) for s in syms]
                if any(len(mono)>1 for p in polys for mono in p): ok=False; break
            except NotNormal: ok=False; break
            coeffs=[]
            for r in range(m):
                for j in range(m): coeffs.append(polys[r].get((j,),ZERO))
                coeffs.append(polys[r].get((),ZERO))
            for j in range(m): coeffs.append(ZERO)
            coeffs.append(ONE)
            certs[key]=Cert('F2',F.F2(m,"ring",vtys[0] if isinstance(vtys[0],str) else "int"),list(range(m)),coeffs=coeffs)
        if ok and certs:
            fam=next(iter(certs.values())).family
            tc=Cert('CTX',fam,list(range(m)),per=certs); tc.l3="context L3 discharged by ring normalization (spec 9.2)"; return tc,None
        return None,Obstruction("a tuple-valued arm is not affine in a single child result, so no context representation exists")
    # F1 for all?
    for mode in ('F1','F3','F0'):
        certs={}; ok=True
        for key,s in candidates:
            if mode=='F1':
                try:
                    poly=sym_to_poly(s)
                    if set(poly)-{(),(0,)}: ok=False; break
                    certs[key]=Cert('F1',F.F1(V if isinstance(V,str) else "int"),[0],coeffs=[poly.get((0,),ZERO),poly.get((),ZERO)])
                except NotNormal: ok=False; break
            elif mode=='F3':
                try:
                    a,l,h=to_clamp(s); certs[key]=Cert('F3',F.F3(),[0],coeffs=[a,l,h])
                except NotNormal: ok=False; break
            else:
                r=infer_numeric_group([s],[0],[V],["v"],None)
                if not isinstance(r,Cert): ok=False; break
                certs[key]=r
        if ok:
            fam=next(iter(certs.values())).family
            tc=Cert('CTX',fam,[0],per=certs); tc.l3="context L3 discharged by normalization (spec 9.2)"; return tc,None
    return None,Obstruction("an arm is not affine/clamp in a single child result, so no context representation exists")

