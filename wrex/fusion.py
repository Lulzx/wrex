"""Fusion (spec 8.3).

Map fusion: a map feeding a fold/scan is fused into the step/rho — the
mapped array is never materialized. Trap order is preserved exactly by
first checking the map function over the input in order (the reference
semantics evaluates the map argument fully before the fold); mapped
values are then recomputed lazily per element (programs are pure).

Product scan fusion: two consecutive scans over the same input become a
single scan over the product state. If either step traps, the reference
order (first scan completes or traps before the second starts) is
restored by falling back to the two sequential scans.
"""
from . import nodes as N
from .values import Trap
import copy

class LazyMap:
    __slots__ = ("base", "f", "interp")
    def __init__(self, base, f, interp): self.base = base; self.f = f; self.interp = interp
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        if isinstance(i, slice): return [self[j] for j in range(*i.indices(len(self)))]
        return self.interp.call_value(self.f, (self.base[i],))
    def __iter__(self):
        for x in self.base: yield self.interp.call_value(self.f, (x,))

def structural_key(node):
    if isinstance(node, N.Node):
        return (type(node).__name__, tuple((k, structural_key(v)) for k, v in vars(node).items() if k not in ('line', 'ty')))
    if isinstance(node, list): return tuple(structural_key(x) for x in node)
    if isinstance(node, tuple): return tuple(structural_key(x) for x in node)
    return node

def _renumber_skel(skel, off):
    if skel[0] == 'leaf': return ('leaf', skel[1] + off)
    return ('tuple', [_renumber_skel(s, off) for s in skel[1]])

def combined_cert(c1, c2):
    """Product certificate (F6) of two scan certificates, or None."""
    from .inference import Cert
    from . import families as F
    if c1 is None or c2 is None: return None
    if structural_key(c1.input_pat) != structural_key(c2.input_pat): return None
    a = len(c1.scope)
    s1 = copy.copy(c1)
    s2 = copy.copy(c2); s2.scope = [i + a for i in c2.scope]; s2.state_skel = _renumber_skel(c2.state_skel, a)
    scope = list(range(a + len(c2.scope)))
    fam = F.F6([c1.family, c2.family], [list(range(a)), list(range(a, a + len(c2.scope)))])
    cert = Cert('F6', fam, scope, subs=[s1, s2])
    cert.state_skel = ('tuple', [c1.state_skel, s2.state_skel])
    cert.input_pat = c1.input_pat
    cert.l3 = "discharged componentwise (product fusion, spec 8.3)"
    return cert

def fused_scans(interp, node1, node2, xs, s1, step1, s2, step2):
    """Run two scans over xs as one product scan; returns (out1, out2)."""
    from . import plans
    c1 = interp.certificates.get(id(node1)); c2 = interp.certificates.get(id(node2))
    cert = combined_cert(c1, c2)
    def combined_step(state, x):
        a, b = state
        return (interp.call_value(step1, (a, x)), interp.call_value(step2, (b, x)))
    try:
        if cert is not None:
            dummy = object()
            interp.certificates[id(dummy)] = cert
            combined = plans.scan(interp, dummy, xs, (s1, s2), combined_step)
        else:
            combined = []
            s = (s1, s2)
            with plans._nested_seq(interp) if False else _null():
                for x in xs:
                    s = combined_step(s, x); combined.append(s)
    except Trap:
        # restore reference trap order: first scan, then second
        out1 = plans._seq_scan(interp, xs, s1, step1)
        out2 = plans._seq_scan(interp, xs, s2, step2)
        return out1, out2
    return [a for a, _ in combined], [b for _, b in combined]

class _null:
    def __enter__(self): return None
    def __exit__(self, *a): return False
