"""tinygrad backend (spec 13.5). Used only when tinygrad is installed and a
non-CPU device is requested; otherwise plans fall back to exact CPU plans.
⋄ is not commutative: every lowering preserves operand order.

Element fields are held as tinygrad tensors with a leading sequence axis
(spec 13.5.2 structure-of-arrays layout). Folds use pairwise tree reduction;
scans use a Hillis-Steele inclusive scan (spec 13.5.3, step-efficient form).
ρ is evaluated per element (it is the certificate's own coefficient code),
⋄ and prefix composition run as tensor operations, and `ev` is applied on
the host — so a failure anywhere falls back to the exact CPU plan.
"""
from .values import Trap
from . import families as F
from . import nodes as N

UNAVAILABLE=object()
_available=None

def available():
    global _available
    if _available is None:
        try:
            import tinygrad  # noqa
            _available=True
        except Exception:
            _available=False
    return _available

def _elems(interp,cert,xs,step):
    from .plans import _rho
    elems=[_rho(interp,cert,x,i,step) for i,x in enumerate(xs)]
    if any(isinstance(e,F.TrapMark) for e in elems): return None
    return elems

# ---- tensor field structures, mirroring Cert kinds ----
def _fields(cert,elems,Tensor):
    k=cert.kind
    if k=='F0': return Tensor([float(e) for e in elems])
    if k=='F1': return (Tensor([float(e[0]) for e in elems]), Tensor([float(e[1]) for e in elems]))
    if k=='F3': return tuple(Tensor([float(e[i]) for e in elems]) for i in range(3))
    if k=='F2': return Tensor([[ [float(v) for v in row] for row in e] for e in elems])
    if k=='F4': return Tensor([[float(cert.family.index[v]) for v in e] for e in elems])
    if k=='F5':
        paycert=cert.payload
        ctrl=Tensor([[float(e2[0] if not isinstance(e2[0],tuple) else 0) for e2 in e] for e in elems]) if False else None
        # control entries may be scalars or tuples; encode as index into qvalues
        idx={q:i for i,q in enumerate(cert.qvalues)}
        ctrl=Tensor([[float(idx[e2[0]]) for e2 in e] for e in elems])
        pay=_fields(paycert,[tuple(e2[1] for e2 in e) if False else [e2[1] for e2 in e][0] if False else None for e in elems],Tensor) if False else _fields_stacked(paycert,elems,Tensor)
        return (ctrl,pay)
    if k=='F6': return tuple(_fields(sub,[e[i] for e in elems],Tensor) for i,sub in enumerate(cert.subs))
    raise ValueError(k)

def _fields_stacked(paycert,elems,Tensor):
    # payload elements stacked per control state: shape leading (n, |Q|, ...)
    per_q=[]
    for qi in range(len(elems[0])):
        per_q.append(_fields(paycert,[e[qi][1] for e in elems],Tensor))
    return _stack(per_q,Tensor)

def _stack(structs,Tensor):
    s0=structs[0]
    if isinstance(s0,tuple): return tuple(_stack([s[i] for s in structs],Tensor) for i in range(len(s0)))
    return Tensor.stack(*structs,dim=1)

def _unstack(fields,nq):
    if isinstance(fields,tuple): return [_unstack_i(fields,i,nq) for i in range(nq)]
    return [fields[:,i] for i in range(nq)]

def _unstack_i(fields,i,nq):
    if isinstance(fields,tuple): return tuple(_unstack_i(f,i,nq) for f in fields)
    return fields[:,i]

def _slice(fields,a,b):
    if isinstance(fields,tuple): return tuple(_slice(f,a,b) for f in fields)
    return fields[a:b]
def _cat(fields_list,Tensor):
    f0=fields_list[0]
    if isinstance(f0,tuple): return tuple(_cat([f[i] for f in fields_list],Tensor) for i in range(len(f0)))
    return fields_list[0].cat(*fields_list[1:],dim=0) if len(fields_list)>1 else fields_list[0]
def _pair(fields):
    # returns (left_fields, right_fields, odd_fields_or_None)
    def n_of(f):
        return int(f.shape[0])
    def get(f):
        n=n_of(f); m=n//2
        return f[:2*m].reshape(m,2,*f.shape[2:]) if False else (f[:2*m:2] if False else None)
    # generic pairing via reshape on leading axis pairs: use stack of even/odd slices
    def split(f):
        n=int(f.shape[0]); m=n//2
        even=f[0:2*m].reshape(m,2,*f.shape[2:])
        return even[:,0], even[:,1], (f[2*m:] if n%2 else None)
    if isinstance(fields,tuple):
        parts=[split(f) if not isinstance(f,tuple) else None for f in fields]
        # tuple structures contain tensors only at leaves; handle recursively instead
        left=tuple(_pair(f)[0] for f in fields); right=tuple(_pair(f)[1] for f in fields); odd=tuple(_pair(f)[2] for f in fields)
        return left,right,odd
    return split(fields)

def _compose(cert,A,B,Tensor):
    """Compose structures A (first) then B, leading axes already aligned."""
    k=cert.kind
    if k=='F0':
        op=cert.family.op
        if op=='+': return A+B
        if op=='*': return A*B
        if op=='min': return Tensor.minimum(A,B)
        if op=='max': return Tensor.maximum(A,B)
        raise ValueError(op)
    if k=='F1':
        a1,b1=A; a2,b2=B; return (a2*a1, a2*b1+b2)
    if k=='F3':
        a1,l1,h1=A; a2,l2,h2=B
        l=Tensor.minimum(Tensor.maximum(l1+a2,l2),h2); h=Tensor.maximum(Tensor.minimum(h1+a2,h2),l2)
        return (a1+a2,l,h)
    if k=='F2':
        if cert.family.semiring=='ring': return B.matmul(A)
        if cert.family.semiring=='maxplus':
            return (B.unsqueeze(2)+A.unsqueeze(1)).max(axis=2,keepdim=False) if False else (B[:,:,:,None]+A[:,:,None,:]).max(axis=2)
        return (B[:,:,:,None]+A[:,:,None,:]).min(axis=2)
    if k=='F4':
        from tinygrad import dtypes
        idx=A.cast(dtypes.int32)
        return B.gather(1,idx)
    if k=='F5':
        from tinygrad import dtypes
        ctrlA,payA=A; ctrlB,payB=B
        idx=ctrlA.cast(dtypes.int32)
        ctrlC=ctrlB.gather(1,idx)
        payB_g=_gather_payload(cert.payload,payB,idx)
        payC=_compose(cert.payload,payA,payB_g,Tensor)
        return (ctrlC,payC)
    if k=='F6':
        return tuple(_compose(sub,a,b,Tensor) for sub,a,b in zip(cert.subs,A,B))
    raise ValueError(k)

def _gather_payload(paycert,fields,idx):
    # fields leading (n, |Q|, ...); gather along axis 1 with idx (n, |Q|)
    if isinstance(fields,tuple): return tuple(_gather_payload(paycert,f,idx) for f in fields)
    n,q=int(idx.shape[0]),int(idx.shape[1])
    rest=fields.shape[2:]
    idxe=idx.reshape(n,q,*([1]*len(rest))).expand(n,q,*rest)
    return fields.gather(1,idxe)

def _to_elems(cert,fields):
    k=cert.kind
    if k=='F0': return fields.tolist()
    if k=='F1':
        a,b=fields; return list(zip(a.tolist(),b.tolist()))
    if k=='F3':
        a,l,h=fields; return list(zip(a.tolist(),l.tolist(),h.tolist()))
    if k=='F2': return fields.tolist()
    if k=='F4': return [tuple(int(v) for v in row) for row in fields.tolist()]
    if k=='F5':
        ctrl,pay=fields; pay_elems=_to_elems_stacked(cert.payload,pay)
        out=[]
        for i,row in enumerate(ctrl.tolist()):
            entry=[]
            for qi,ci in enumerate(row):
                q=cert.qvalues[int(round(ci))]
                entry.append((q,pay_elems[qi][i]))
            out.append(tuple(entry))
        return out
    if k=='F6': return list(zip(*[_to_elems(sub,f) for sub,f in zip(cert.subs,fields)]))
    raise ValueError(k)

def _to_elems_stacked(paycert,fields):
    # inverse of _fields_stacked: returns list per q of element lists
    if isinstance(fields,tuple):
        # need nq from first leaf
        def first(f):
            return f[0] if isinstance(f,tuple) else f
        nq=int(first(fields).shape[1])
        return [_to_elems(paycert,_unstack_i(fields,qi,nq)) for qi in range(nq)]
    nq=int(fields.shape[1])
    return [_to_elems(paycert,fields[:,qi]) for qi in range(nq)]

def _clean(cert,elems):
    # floats back to ints where the family is integral
    out=[]
    for e in elems:
        out.append(_clean1(cert,e))
    return out
def _clean1(cert,e):
    k=cert.kind
    def ci(v): return int(v) if isinstance(v,float) and v not in (float('inf'),float('-inf')) and float(v)==int(v) else v
    if k=='F0': return ci(e)
    if k=='F1': return (ci(e[0]),ci(e[1]))
    if k=='F3': return tuple(ci(v) for v in e)
    if k=='F2': return [[ci(v) for v in row] for row in e]
    if k=='F4': return tuple(cert.family.q[int(v)] if not isinstance(v,int) else v for v in e) if False else tuple(e)
    if k=='F5': return tuple((q,_clean1(cert.payload,p)) for q,p in e)
    if k=='F6': return tuple(_clean1(s,x) for s,x in zip(cert.subs,e))
    return e

def _fold_fields(cert,fields,Tensor):
    def n_of(f):
        return int(f.shape[0]) if not isinstance(f,tuple) else n_of(f[0])
    while n_of(fields)>1:
        left,right,odd=_pair(fields)
        nxt=_compose(cert,left,right,Tensor)
        if odd is not None and odd[0] is not None if isinstance(odd,tuple) else odd is not None:
            nxt=_cat([nxt,odd],Tensor)
        fields=nxt
    return fields

def gpu_fold(interp,node,xs,s0,step,cert):
    if cert is None or cert.kind not in ('F0','F1','F2','F3','F4','F5','F6'): return UNAVAILABLE
    if cert.kind=='F0' and cert.family.op not in ('+','*','min','max'): return UNAVAILABLE
    elems=_elems(interp,cert,xs,step)
    if elems is None or not elems: return UNAVAILABLE
    try:
        from tinygrad import Tensor
        fields=_fold_fields(cert,_fields(cert,elems,Tensor),Tensor)
        elem=_clean(cert,_to_elems(cert,fields))[0]
        # F4 elements are tables of values already; _to_elems returns index rows for comfort: convert
        if cert.kind=='F4': elem=tuple(cert.family.q[int(round(v))] if isinstance(v,float) else v for v in elem)
        from .plans import _ev
        return _ev(interp,cert,elem,s0)
    except Trap:
        raise
    except Exception:
        return UNAVAILABLE

def gpu_contract(interp,node,tree,ctx,env):
    """GPU tree contraction (spec 9.3/13.5): rake and unary conversion run
    through the same exact arm evaluation as the CPU plan, while each
    round's independent ctx-ctx compressions (child ⋄ parent) are composed
    as one batched tinygrad operation over the context family's fields
    (F0, F1, F3, and F2 including tropical and tuple-valued contexts)."""
    if ctx is None or len({c.kind for c in ctx.per.values()})!=1: return UNAVAILABLE
    if next(iter(ctx.per.values())).kind not in ('F0','F1','F2','F3'): return UNAVAILABLE
    try:
        from tinygrad import Tensor
        from . import plans
        from .values import ADT
        T=tree.type_name; ctor_fields={c:f for c,f in interp.types[T]}
        def build(t):
            if not isinstance(t,ADT): return plans._W('val',val=t)
            fields=[build(fv) if ft==T else fv for ft,fv in zip(ctor_fields[t.ctor],t.fields)]
            return plans._W('orig',ctor=t.ctor,fields=fields,raw=t)
        def arm_vals(w,vals):
            return [vals[j] if isinstance(f,plans._W) else f for j,f in enumerate(w.fields)]
        def do_pass(w):
            if w.kind=='val': return w,False
            if w.kind=='ctx':
                child,ch=do_pass(w.child)
                if child.kind=='val': return plans._W('val',val=ctx.family.ev(w.elem,child.val)),True
                return plans._W('ctx',elem=w.elem,child=child),ch
            newfields=[]; changed=False
            for f in w.fields:
                if isinstance(f,plans._W):
                    nf,c=do_pass(f); newfields.append(nf); changed=changed or c
                else: newfields.append(f)
            w2=plans._W('orig',ctor=w.ctor,fields=newfields,raw=w.raw)
            unknown=[j for j,f in enumerate(newfields) if isinstance(f,plans._W) and f.kind!='val']
            if not unknown:
                vals={j:(f.val if isinstance(f,plans._W) else f) for j,f in enumerate(newfields)}
                return plans._W('val',val=plans._eval_arm(interp,node.arms,w.ctor,arm_vals(w2,vals),env)),True
            if len(unknown)==1:
                j=unknown[0]; key=(w.ctor,j)
                if key in ctx.per:
                    vals={jj:(f.val if isinstance(f,plans._W) else f) for jj,f in enumerate(newfields) if jj!=j}
                    env2=dict(env); pat=None
                    for p_,b_ in node.arms:
                        if isinstance(p_,N.PCtor) and p_.name==w.ctor: pat=p_
                    fvals=[vals.get(jj,0) for jj in range(len(newfields))]
                    for jj,(p_,v) in enumerate(zip(pat.items,fvals)):
                        if jj!=j: interp.match_pattern(p_,v,env2)
                    cert=ctx.per[key]
                    def arm_of_hole(hole,j=j,w2=w2,vals=vals):
                        vv=dict(vals); vv[j]=hole
                        return plans._eval_arm(interp,node.arms,w2.ctor,arm_vals(w2,vv),env)
                    elem=cert.rho_env(interp,env2,0,arm_of_hole)
                    return plans._W('ctx',elem=elem,child=newfields[j]),True
            return w2,changed
        root=build(tree); guard=0
        while root.kind!='val':
            guard+=1
            if guard>10000: return UNAVAILABLE
            # collect independent ctx-ctx pairs (no pair shares a node)
            pairs=[]
            def collect(w):
                if w.kind=='ctx' and w.child.kind=='ctx':
                    pairs.append(w); collect(w.child.child); return
                if w.kind=='ctx': collect(w.child); return
                if w.kind=='orig':
                    for f in w.fields:
                        if isinstance(f,plans._W): collect(f)
            collect(root)
            if pairs:
                def flat(e):
                    if isinstance(e,(list,tuple)):
                        out=[]
                        for x in e: out+=flat(x)
                        return out
                    return [e]
                if any(not isinstance(v,(int,float)) or isinstance(v,bool) or abs(v)>=2**53 for w in pairs for e in (w.elem,w.child.elem) for v in flat(e)):
                    return UNAVAILABLE
                rep=next(iter(ctx.per.values()))
                cfields=_fields(rep,[w.child.elem for w in pairs],Tensor)
                pfields=_fields(rep,[w.elem for w in pairs],Tensor)
                composed=_clean(rep,_to_elems(rep,_compose(rep,cfields,pfields,Tensor)))
                for w,elem in zip(pairs,composed):
                    child=w.child; w.elem=elem; w.child=child.child
                continue
            root,changed=do_pass(root)
            if not changed and root.kind!='val': return UNAVAILABLE
        return root.val
    except Trap:
        raise
    except Exception:
        return UNAVAILABLE

def gpu_scan(interp,node,xs,s0,step,cert):
    if cert is None or cert.kind not in ('F0','F1','F2','F3','F4','F5','F6'): return UNAVAILABLE
    if cert.kind=='F0' and cert.family.op not in ('+','*','min','max'): return UNAVAILABLE
    elems=_elems(interp,cert,xs,step)
    if elems is None or not elems: return UNAVAILABLE
    try:
        from tinygrad import Tensor
        fields=_fields(cert,elems,Tensor)
        n=len(elems); step_sz=1
        while step_sz<n:
            left=_slice(fields,0,n-step_sz); right=_slice(fields,step_sz,n)
            comp=_compose(cert,left,right,Tensor)
            fields=_cat([_slice(fields,0,step_sz),comp],Tensor)
            step_sz*=2
        prefixes=_clean(cert,_to_elems(cert,fields))
        if cert.kind=='F4': prefixes=[tuple(cert.family.q[int(round(v))] if isinstance(v,float) else v for v in e) for e in prefixes]
        from .plans import _ev
        return [_ev(interp,cert,e,s0) for e in prefixes]
    except Trap:
        raise
    except Exception:
        return UNAVAILABLE
