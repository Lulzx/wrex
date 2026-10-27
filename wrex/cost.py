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
    c_comp=cert.cost if cert else 0; c_ev=c_comp
    t_seq=n*c_step
    if cert is None or p<=1: return {"Seq":t_seq}
    t_par=(n/p)*(c_rho+c_comp)+math.log2(max(p,2))*(c_comp+cal["c_sync"])+c_ev
    return {"Seq":t_seq,"ParFold":t_par}

def scan_costs(n,p,cert,c_step,cal=None):
    cal=cal or load_cal()
    c_rho=c_step
    if cert is not None and cert.kind in ('F4','F5'): c_rho=c_step*max(1,cert.family.dim)
    c_comp=cert.cost if cert else 0
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

def calibrate():
    cal=load_cal()
    # measure a simple op loop to scale c_op / c_sync relative constants
    t0=time.perf_counter(); s=0
    for i in range(200000): s+=i*i
    dt=time.perf_counter()-t0
    cal["c_op"]=max(0.1,dt/200000*1e6)  # relative units
    with open(CAL_PATH,'w') as f: json.dump(cal,f,indent=1)
    return cal
