import math, json, os, time

CAL_PATH=os.path.join(os.path.expanduser("~"),".wrex-calibration.json")
CAL={"c_sync":10.0,"c_op":1.0,"kappa":2.0,"kappa_prime":1.0,"beta":1e9,"c_launch":5000.0}
def load_cal():
    cal=dict(CAL)
    try:
        with open(CAL_PATH) as f: cal.update(json.load(f))
    except Exception: pass
    return cal

def op_count(node):
    if node is None: return 1
    n=1
    for v in vars(node).values():
        if hasattr(v,'__dict__'): n+=op_count(v)
        elif isinstance(v,list):
            for y in v:
                if hasattr(y,'__dict__'): n+=op_count(y)
                elif isinstance(y,tuple):
                    for z in y:
                        if hasattr(z,'__dict__'): n+=op_count(z)
    return max(1,n)

def fold_costs(n,p,cert,c_step,cal=None):
    cal=cal or load_cal()
    c_rho=c_step
    if cert is not None and cert.kind in ('F4','F5'): c_rho=c_step*max(1,cert.family.dim)
    c_comp=comp_cost(cert,cal); c_ev=c_comp
    t_seq=n*c_step
    if cert is None or p<=1: return {"Seq":t_seq}
    t_par=(n/p)*(c_rho+c_comp)+math.log2(max(p,2))*(c_comp+cal["c_sync"])+c_ev
    return {"Seq":t_seq,"ParFold":t_par}

def scan_costs(n,p,cert,c_step,cal=None):
    cal=cal or load_cal()
    c_rho=c_step
    if cert is not None and cert.kind in ('F4','F5'): c_rho=c_step*max(1,cert.family.dim)
    c_comp=comp_cost(cert,cal)
    t_seq=n*c_step
    if cert is None or p<=1: return {"Seq":t_seq}
    t_par=(n/p)*(c_rho+c_comp+c_step)+p*c_comp+2*cal["c_sync"]
    return {"Seq":t_seq,"ParScan":t_par}

def tree_costs(n,h,p,ctx,c_arm,cal=None):
    cal=cal or load_cal()
    t_fj=(n/max(p,1))*c_arm+h*cal["c_sync"]
    out={"ForkJoin":t_fj}
    if ctx is not None:
        out["Contract"]=cal["kappa"]*(n/max(p,1))*(c_arm+ctx.cost)+cal["kappa_prime"]*math.log2(max(n,2))*cal["c_sync"]
    return out

def gpu_fold_cost(n,cert,cal=None,P_dev=1024,B=0):
    cal=cal or load_cal()
    return B/cal["beta"]+math.ceil(math.log2(max(n,2)))*cal["c_launch"]+(n/P_dev)*(1+cert.cost)
def gpu_scan_cost(n,cert,cal=None,P_dev=1024,B=0):
    cal=cal or load_cal()
    return B/cal["beta"]+2*math.ceil(math.log2(max(n,2)))*cal["c_launch"]+(2*n/P_dev)*(1+cert.cost+cert.cost)

def comp_cost(cert,cal=None):
    """Cost of one compose: calibrated per-family ratio if available, else the spec cost."""
    cal=cal or load_cal()
    table=cal.get("compose_cost") or {}
    if cert is not None and cert.kind in table: return float(table[cert.kind])
    return cert.cost if cert else 0

def calibrate():
    cal=load_cal()
    t0=time.perf_counter(); s=0
    for i in range(200000): s+=i*i
    dt=time.perf_counter()-t0
    cal["c_op"]=max(0.1,dt/200000*1e6)  # relative units
    # per-family compose costs, measured relative to F0 (spec 10.2 calibration)
    try:
        from . import families as F
        import random as _r
        rnd=_r.Random(0)
        specimens={
            "F0": (F.F0('+','int'), lambda: rnd.randint(-99,99)),
            "F1": (F.F1('int'), lambda: (rnd.randint(-3,3), rnd.randint(-99,99))),
            "F3": (F.F3(), lambda: (rnd.randint(-5,5), rnd.randint(-50,0), rnd.randint(0,50))),
            "F2": (F.F2(2,'ring','int'), lambda: [[rnd.randint(-3,3)]*3 for _ in range(3)]),
            "F4": (F.F4(list(range(8))), lambda: tuple(rnd.randrange(8) for _ in range(8))),
        }
        times={}
        for name,(fam,gen) in specimens.items():
            elems=[gen() for _ in range(64)]
            t0=time.perf_counter()
            for i in range(2000): fam.compose(elems[i%64], elems[(i+1)%64])
            times[name]=max(time.perf_counter()-t0, 1e-9)
        base=times["F0"]
        cal["compose_cost"]={k: round(max(1.0, v/base), 2) for k,v in times.items()}
    except Exception:
        pass
    with open(CAL_PATH,'w') as f: json.dump(cal,f,indent=1)
    return cal
