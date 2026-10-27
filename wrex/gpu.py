"""tinygrad backend (spec 13.5). Used only when tinygrad is installed and a
non-CPU device is requested; otherwise plans fall back to exact CPU plans.
⋄ is not commutative: every lowering preserves operand order."""
from .values import Trap
from . import families as F

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

def gpu_fold(interp,node,xs,s0,step,cert):
    if cert is None or cert.kind not in ('F0','F1'): return UNAVAILABLE
    if cert.kind=='F0' and cert.family.op not in ('+','*','min','max'): return UNAVAILABLE
    elems=_elems(interp,cert,xs,step)
    if elems is None or not elems: return UNAVAILABLE
    try:
        from tinygrad import Tensor
        if cert.kind=='F0':
            cur=Tensor([float(e) for e in elems])
            while cur.shape[0]>1:
                n=int(cur.shape[0]); m=n//2
                last=cur[n-1:n]
                a=cur[:2*m].reshape(m,2)
                op=cert.family.op
                if op=='+': nxt=a[:,0]+a[:,1]
                elif op=='*': nxt=a[:,0]*a[:,1]
                elif op=='min': nxt=Tensor.minimum(a[:,0],a[:,1])
                else: nxt=Tensor.maximum(a[:,0],a[:,1])
                cur=nxt.cat(last,dim=0) if n%2 else nxt
            elem=cur.tolist()
            if isinstance(elem,list): elem=elem[0]
            if cert.family.ty!='f64' and float(elem)==int(elem): elem=int(elem)
        else:  # F1: (a,b) fields, compose (a2*a1, a2*b1+b2)
            a=Tensor([float(e[0]) for e in elems]); b=Tensor([float(e[1]) for e in elems])
            while a.shape[0]>1:
                n=int(a.shape[0]); m=n//2
                a1=a[:2*m].reshape(m,2); b1=b[:2*m].reshape(m,2)
                na=a1[:,1]*a1[:,0]; nb=a1[:,1]*b1[:,0]+b1[:,1]
                if n%2:
                    na=na.cat(a[-1:],dim=0); nb=nb.cat(b[-1:],dim=0)
                a,b=na,nb
            elem=(int(a.tolist()) if float(a.tolist())==int(a.tolist()) else a.tolist(),
                  int(b.tolist()) if float(b.tolist())==int(b.tolist()) else b.tolist())
        from .plans import _ev
        return _ev(interp,cert,elem,s0)
    except Trap:
        raise
    except Exception:
        return UNAVAILABLE

def gpu_scan(interp,node,xs,s0,step,cert):
    # No tensor scan lowering yet: returning UNAVAILABLE keeps the exact CPU
    # three-phase plan, preserving plan-independence (spec 5.5).
    return UNAVAILABLE
