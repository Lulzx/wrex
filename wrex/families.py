from .values import wrap, Trap

INF=float('inf'); NINF=float('-inf')

class TrapMark:
    __slots__=("idx","kind","msg")
    def __init__(self,idx,kind="trap",msg=""): self.idx=idx; self.kind=kind; self.msg=msg
    def __repr__(self): return f"⊥{self.idx}"

def is_inf(v): return isinstance(v,float) and v in (INF,NINF)
def add_inf(v,a):
    if is_inf(v): return v
    if is_inf(a): return a
    return v+a
def clean(v):
    if isinstance(v,float) and v not in (INF,NINF) and v==int(v): return int(v)
    return v

def trap_compose(a,b):
    """Return winning TrapMark if either is one, else None."""
    ta=a if isinstance(a,TrapMark) else None; tb=b if isinstance(b,TrapMark) else None
    if ta and tb: return ta if ta.idx<=tb.idx else tb
    return ta or tb

class Family:
    name="F?"; cost=1; dim=1
    def identity(self): raise NotImplementedError
    def compose(self,r1,r2): raise NotImplementedError
    def ev(self,r,s): raise NotImplementedError
    def describe(self): return self.name

class F0(Family):
    name="F0"
    def __init__(self,op,ty="int"):
        self.op=op; self.ty=ty; self.cost=1; self.dim=1
    def identity(self):
        return {'+':0,'*':1,'min':INF,'max':NINF,'and':True,'or':False,'&':-1 if self.ty.startswith('i') else (1<<64)-1,'|':0,'^':0}[self.op]
    def _op(self,a,b):
        op=self.op
        if op=='+': r=a+b
        elif op=='*': r=a*b
        elif op=='min': r=min(a,b)
        elif op=='max': r=max(a,b)
        elif op=='and': r=bool(a and b)
        elif op=='or': r=bool(a or b)
        elif op=='&': r=int(a)&int(b)
        elif op=='|': r=int(a)|int(b)
        elif op=='^': r=int(a)^int(b)
        else: raise ValueError(op)
        return wrap(clean(r),self.ty) if not isinstance(r,bool) else r
    def compose(self,r1,r2):
        t=trap_compose(r1,r2)
        return t if t else self._op(r1,r2)
    def ev(self,r,s):
        if isinstance(r,TrapMark): raise Trap(r.kind,r.msg)
        return self._op(s,r)
    def describe(self): return f"F0 direct monoid ({self.op} over {self.ty})"

class F1(Family):
    name="F1"
    def __init__(self,ty="int"): self.ty=ty; self.cost=3; self.dim=1
    def identity(self): return (1,0)
    def compose(self,r1,r2):
        t=trap_compose(r1,r2)
        if t: return t
        (a1,b1),(a2,b2)=r1,r2
        return (wrap(a2*a1,self.ty), wrap(a2*b1+b2,self.ty))
    def ev(self,r,s):
        if isinstance(r,TrapMark): raise Trap(r.kind,r.msg)
        a,b=r; return wrap(a*s+b,self.ty)
    def describe(self): return f"F1 affine over {self.ty}"

class F2(Family):
    name="F2"
    def __init__(self,k,semiring="ring",ty="int"):
        self.k=k; self.semiring=semiring; self.ty=ty; self.cost=(k+1)**3; self.dim=k
    def _zero(self): return {'ring':0,'maxplus':NINF,'minplus':INF}[self.semiring]
    def _one(self): return 1 if self.semiring=='ring' else 0
    def _add(self,a,b):
        if self.semiring=='ring': return wrap(a+b,self.ty)
        return max(a,b) if self.semiring=='maxplus' else min(a,b)
    def _mul(self,a,b):
        if self.semiring=='ring': return wrap(a*b,self.ty)
        return add_inf(a,b)
    def identity(self):
        n=self.k+1; z,o=self._zero(),self._one()
        return [[o if i==j else z for j in range(n)] for i in range(n)]
    def compose(self,r1,r2):
        t=trap_compose(r1,r2)
        if t: return t
        n=self.k+1
        # M2 ⊗ M1
        out=[[self._zero()]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                acc=self._zero()
                for m in range(n): acc=self._add(acc,self._mul(r2[i][m],r1[m][j]))
                out[i][j]=clean(acc)
        return out
    def ev(self,r,s):
        if isinstance(r,TrapMark): raise Trap(r.kind,r.msg)
        vec=list(s)+[self._one()]
        out=[]
        for i in range(self.k):
            acc=self._zero()
            for j in range(self.k+1): acc=self._add(acc,self._mul(r[i][j],vec[j]))
            out.append(wrap(clean(acc),self.ty) if self.semiring=='ring' else clean(acc))
        return tuple(out)
    def describe(self): return f"F2 {'tropical '+self.semiring if self.semiring!='ring' else 'linear'} over {self.ty} (k={self.k})"

class F3(Family):
    name="F3"; cost=6; dim=1
    def identity(self): return (0,NINF,INF)
    def compose(self,r1,r2):
        t=trap_compose(r1,r2)
        if t: return t
        a1,l1,h1=r1; a2,l2,h2=r2
        l=min(max(add_inf(l1,a2),l2),h2); h=max(min(add_inf(h1,a2),h2),l2)
        return (clean(add_inf(a1,a2)),clean(l),clean(h))
    def ev(self,r,s):
        if isinstance(r,TrapMark): raise Trap(r.kind,r.msg)
        a,l,h=r; return clean(min(h,max(l,add_inf(s,a))))
    def describe(self): return "F3 shifted clamps over int"

class F4(Family):
    name="F4"
    def __init__(self,qvalues): self.q=list(qvalues); self.cost=len(self.q); self.dim=len(self.q); self.index={v:i for i,v in enumerate(self.q)}
    def identity(self): return tuple(self.q)
    def compose(self,r1,r2):
        if isinstance(r1,TrapMark) or isinstance(r2,TrapMark): return trap_compose(r1,r2)
        out=[]
        for entry in r1:
            if isinstance(entry,TrapMark): out.append(entry)
            else:
                e2=r2[self.index[entry]]
                out.append(e2)
        return tuple(out)
    def ev(self,r,s):
        if isinstance(r,TrapMark): raise Trap(r.kind,r.msg)
        entry=r[self.index[s]]
        if isinstance(entry,TrapMark): raise Trap(entry.kind,entry.msg)
        return entry
    def describe(self): return f"F4 finite transformations (|Q|={len(self.q)})"

class F5(Family):
    name="F5"
    def __init__(self,qvalues,payload):
        self.q=list(qvalues); self.payload=payload; self.index={v:i for i,v in enumerate(self.q)}
        self.cost=len(self.q)*payload.cost; self.dim=len(self.q)
    def identity(self): return tuple((v,self.payload.identity()) for v in self.q)
    def compose(self,r1,r2):
        if isinstance(r1,TrapMark) or isinstance(r2,TrapMark): return trap_compose(r1,r2)
        out=[]
        for entry in r1:
            if isinstance(entry,TrapMark): out.append(entry); continue
            q1,m1=entry
            e2=r2[self.index[q1]]
            if isinstance(e2,TrapMark): out.append(e2); continue
            q2,m2=e2
            out.append((q2,self.payload.compose(m1,m2)))
        return tuple(out)
    def ev(self,r,s):
        if isinstance(r,TrapMark): raise Trap(r.kind,r.msg)
        q,v=s; entry=r[self.index[q]]
        if isinstance(entry,TrapMark): raise Trap(entry.kind,entry.msg)
        q2,m=entry
        return (q2,self.payload.ev(m,v))
    def describe(self): return f"F5 finite control (|Q|={len(self.q)}) with payload {self.payload.describe()}"

class F6(Family):
    name="F6"
    def __init__(self,subs,spans):
        # subs: list[Family], spans: list[list[int]] component indices into state tuple
        self.subs=subs; self.spans=spans; self.cost=sum(s.cost for s in subs); self.dim=sum(s.dim for s in subs)
    def identity(self): return tuple(s.identity() for s in self.subs)
    def compose(self,r1,r2):
        t=trap_compose(r1,r2)
        if t: return t
        out=[]
        for s,a,b in zip(self.subs,r1,r2):
            c=s.compose(a,b); out.append(c)
            if isinstance(c,TrapMark): pass
        # propagate earliest trap at top level for simplicity when any component trapped
        traps=[c for c in out if isinstance(c,TrapMark)]
        if traps: return min(traps,key=lambda x:x.idx)
        return tuple(out)
    def ev(self,r,s):
        if isinstance(r,TrapMark): raise Trap(r.kind,r.msg)
        out=[None]*len(s)
        for fam,span,elem in zip(self.subs,self.spans,r):
            sub=tuple(s[i] for i in span) if len(span)>1 else s[span[0]]
            v=fam.ev(elem,sub)
            if len(span)>1:
                for i,val in zip(span,v): out[i]=val
            else: out[span[0]]=v
        return tuple(out)
    def describe(self): return "F6 product ("+", ".join(s.describe() for s in self.subs)+")"
