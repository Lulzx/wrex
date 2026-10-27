import math
from . import nodes as N
from .values import Trap, ADT
from . import families as F
from .inference import Cert, decompose, assemble
from . import cost as costmod

def _cert_for(interp,node):
    return interp.certificates.get(id(node))

def _rho(interp,cert,x,idx,step):
    env=interp.bind_pattern(cert.input_pat,x) if getattr(cert,'input_pat',None) is not None else {}
    stepfn=lambda s: interp.call_value(step,(s,x))
    try:
        return cert.rho_env(interp,env,idx,stepfn)
    except Trap as t:
        return F.TrapMark(idx,t.kind,t.msg)

def _ev(interp,cert,elem,s0):
    comp={}
    decompose(cert.state_skel,s0,comp)
    vals=[comp[i] for i in range(len(comp))]
    out=cert.ev_state(elem,vals)
    return assemble(cert.state_skel,out)

class _nested_seq:
    """Scheduling rule for nested parallelism (spec section 16, question 3):
    while a parallel plan is evaluating step functions (rho or replay), any
    fold/scan/treefold inside those steps runs sequentially. The outer plan
    owns the parallelism; nested plans would oversubscribe and, in this
    runtime, would interleave plan decisions unpredictably."""
    def __init__(self,interp): self.interp=interp
    def __enter__(self): self.interp.parallel_depth=getattr(self.interp,'parallel_depth',0)+1
    def __exit__(self,*a): self.interp.parallel_depth-=1

def _seq_fold(interp,xs,s0,step):
    s=s0
    for x in xs: s=interp.call_value(step,(s,x))
    return s
def _seq_scan(interp,xs,s0,step):
    out=[]; s=s0
    for x in xs: s=interp.call_value(step,(s,x)); out.append(s)
    return out

def _forced_seq(interp):
    return any(fn in interp.seq_fns for fn in interp.fn_stack) or getattr(interp,'parallel_depth',0)>0

def fold(interp,node,xs,s0,step):
    cert=_cert_for(interp,node)
    cfg=interp.config
    if _forced_seq(interp): return _seq_fold(interp,xs,s0,step)
    if cfg.device not in (None,'cpu'):
        from . import gpu
        if gpu.available():
            use=True
            if cfg.plan=='auto' and cert is not None:
                # GPU cost guard (spec 10.2/13.5.5): only offload when the device wins
                cal=costmod.load_cal()
                use=costmod.gpu_fold_cost(len(xs),cert,cal) < len(xs)*costmod.op_count(getattr(cert,'step_node',None))
            if use:
                with _nested_seq(interp):
                    r=gpu.gpu_fold(interp,node,xs,s0,step,cert)
                if r is not gpu.UNAVAILABLE: return r
    if cfg.plan=='seq' or cert is None: return _seq_fold(interp,xs,s0,step)
    if cfg.plan=='auto':
        if cert.cost>costmod.REP_BENEFIT_MAX_COST: return _seq_fold(interp,xs,s0,step)
        costs=costmod.fold_costs(len(xs),cfg.workers,cert,costmod.op_count(getattr(cert,'step_node',None)))
        if min(costs,key=costs.get)!='ParFold': return _seq_fold(interp,xs,s0,step)
    if len(xs)==0: return s0
    with _nested_seq(interp):
        elems=[_rho(interp,cert,x,i,step) for i,x in enumerate(xs)]
        # balanced tree reduction (spec 8.2)
        while len(elems)>1:
            nxt=[]
            for i in range(0,len(elems)-1,2): nxt.append(cert.family.compose(elems[i],elems[i+1]))
            if len(elems)%2: nxt.append(elems[-1])
            elems=nxt
        elem=elems[0]
        if isinstance(elem,F.TrapMark): return _seq_fold(interp,xs,s0,step)  # replays to exact trap
        try: return _ev(interp,cert,elem,s0)
        except Trap: return _seq_fold(interp,xs,s0,step)

def scan(interp,node,xs,s0,step):
    cert=_cert_for(interp,node)
    cfg=interp.config
    if _forced_seq(interp): return _seq_scan(interp,xs,s0,step)
    if cfg.device not in (None,'cpu'):
        from . import gpu
        if gpu.available():
            use=True
            if cfg.plan=='auto' and cert is not None:
                cal=costmod.load_cal()
                use=costmod.gpu_scan_cost(len(xs),cert,cal) < len(xs)*costmod.op_count(getattr(cert,'step_node',None))
            if use:
                with _nested_seq(interp):
                    r=gpu.gpu_scan(interp,node,xs,s0,step,cert)
                if r is not gpu.UNAVAILABLE: return r
    if cfg.plan=='seq' or cert is None: return _seq_scan(interp,xs,s0,step)
    if cfg.plan=='auto':
        if cert.cost>costmod.REP_BENEFIT_MAX_COST: return _seq_scan(interp,xs,s0,step)
        costs=costmod.scan_costs(len(xs),cfg.workers,cert,costmod.op_count(getattr(cert,'step_node',None)))
        if min(costs,key=costs.get)!='ParScan': return _seq_scan(interp,xs,s0,step)
    n=len(xs)
    if n==0: return []
    p=max(1,min(cfg.workers,n))
    bounds=[(b*n//p,(b+1)*n//p) for b in range(p)]
    with _nested_seq(interp):
        # Phase 1: summarize blocks
        summaries=[]
        for a,b in bounds:
            elem=cert.family.identity()
            for i in range(a,b): elem=cert.family.compose(elem,_rho(interp,cert,xs[i],i,step))
            summaries.append(elem)
        # Phase 2: exclusive prefixes -> block start states
        starts=[]; prefix=cert.family.identity()
        try:
            for b in range(p):
                starts.append(_ev(interp,cert,prefix,s0))
                prefix=cert.family.compose(prefix,summaries[b])
        except Trap:
            return _seq_scan(interp,xs,s0,step)
        if isinstance(prefix,F.TrapMark): return _seq_scan(interp,xs,s0,step)
        # Phase 3: replay step through each block
        out=[None]*n
        for (a,b),start in zip(bounds,starts):
            s=start
            for i in range(a,b):
                s=interp.call_value(step,(s,xs[i])); out[i]=s
        return out

# ---------------- tree folds
class _W:
    __slots__=("kind","val","ctor","fields","arm","ctxkey","elem","child","raw")
    def __init__(self,kind,**kw):
        self.kind=kind; self.val=None; self.ctor=None; self.fields=[]; self.arm=None; self.ctxkey=None; self.elem=None; self.child=None; self.raw=None
        for k,v in kw.items(): setattr(self,k,v)

def _eval_arm(interp,arms,wanted_ctor,field_vals,env):
    for pat,blk in arms:
        if not isinstance(pat,N.PCtor) or pat.name!=wanted_ctor: continue
        env2=dict(env); ok=True
        for p,v in zip(pat.items,field_vals):
            if not interp.match_pattern(p,v,env2): ok=False; break
        if ok: return interp.eval_block(blk,env2)
    raise Trap("match_failure","treefold: no arm matched")

def treefold(interp,node,env):
    tree=interp.eval_expr(node.scrut,env)
    cfg=interp.config
    if _forced_seq(interp): return interp.treefold_seq(tree,node.arms,env)
    ctx=interp.ctx_certs.get(id(node))
    if cfg.plan=='seq' or ctx is None:
        # ForkJoin and Seq share the exact post-order semantics; ForkJoin is the auto default without ctx need
        return interp.treefold_seq(tree,node.arms,env)
    if cfg.plan=='auto':
        n=getattr(tree,'size',1); h=getattr(tree,'height',1)
        costs=costmod.tree_costs(n,h,cfg.workers,ctx,costmod.op_count(node))
        if min(costs,key=costs.get)!='Contract':
            return interp.treefold_seq(tree,node.arms,env)
    try:
        with _nested_seq(interp):
            if cfg.device not in (None,'cpu'):
                from . import gpu
                if gpu.available():
                    r=gpu.gpu_contract(interp,node,tree,ctx,env)
                    if r is not gpu.UNAVAILABLE: return r
            return _contract(interp,node,tree,ctx,env)
    except Trap:
        return interp.treefold_seq(tree,node.arms,env)  # re-runs to exact first trap (spec 9.4)

def _contract(interp,node,tree,ctx,env):
    T=tree.type_name
    ctor_fields={c:f for c,f in interp.types[T]}
    def build(t):
        if not isinstance(t,ADT): return _W('val',val=t)
        fields=[]
        for ft,fv in zip(ctor_fields[t.ctor],t.fields):
            fields.append(build(fv) if ft==T else fv)
        return _W('orig',ctor=t.ctor,fields=fields,raw=t)
    def arm_vals(w,vals_by_pos):
        out=[]
        for j,f in enumerate(w.fields):
            if isinstance(f,_W): out.append(vals_by_pos[j])
            else: out.append(f)
        return out
    def step_pass(w):
        # returns (new_w, changed)
        if w.kind=='val': return w,False
        if w.kind=='ctx':
            child,changed=step_pass(w.child)
            if child.kind=='val': return _W('val',val=ctx.family.ev(w.elem,child.val)),True
            if child.kind=='ctx':
                # child applied first, then parent: child ⋄ parent (spec 9.3)
                return _W('ctx',elem=ctx.family.compose(child.elem,w.elem),child=child.child),True
            return _W('ctx',elem=w.elem,child=child),changed
        # orig
        newfields=[]; changed=False
        for f in w.fields:
            if isinstance(f,_W):
                nf,c=step_pass(f); newfields.append(nf); changed=changed or c
            else: newfields.append(f)
        w2=_W('orig',ctor=w.ctor,fields=newfields,raw=w.raw)
        unknown=[j for j,f in enumerate(newfields) if isinstance(f,_W) and f.kind!='val']
        if not unknown:
            vals={j:(f.val if isinstance(f,_W) else f) for j,f in enumerate(newfields)}
            return _W('val',val=_eval_arm(interp,node.arms,w.ctor,arm_vals(w2,vals),env)),True
        if len(unknown)==1:
            j=unknown[0]; key=(w.ctor,j)
            if key in ctx.per:
                vals={jj:(f.val if isinstance(f,_W) else f) for jj,f in enumerate(newfields) if jj!=j}
                # env for rho_ctx: bind arm pattern with known/raw, hole skipped
                env2=dict(env)
                pat=None
                for p_,b_ in node.arms:
                    if isinstance(p_,N.PCtor) and p_.name==w.ctor: pat=p_
                fvals=[]
                for jj,f in enumerate(newfields):
                    fvals.append(vals.get(jj,0))
                for jj,(p_,v) in enumerate(zip(pat.items,fvals)):
                    if jj!=j: interp.match_pattern(p_,v,env2)
                cert=ctx.per[key]
                def arm_of_hole(hole,j=j,w2=w2,vals=vals):
                    vv=dict(vals); vv[j]=hole
                    return _eval_arm(interp,node.arms,w2.ctor,arm_vals(w2,vv),env)
                elem=cert.rho_env(interp,env2,0,arm_of_hole)
                return _W('ctx',elem=elem,child=newfields[j]),True
        return w2,changed
    root=build(tree)
    guard=0
    while root.kind!='val':
        root,changed=step_pass(root); guard+=1
        if guard>10000: raise Trap("contract","contraction did not converge")
    return root.val
