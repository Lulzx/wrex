class Trap(Exception):
    def __init__(self,kind,msg,line=None):
        super().__init__(msg); self.kind=kind; self.msg=msg; self.line=line
    def __str__(self): return f"trap: {self.kind}: {self.msg}"

class ADT:
    __slots__=("ctor","fields","type_name","size","height")
    def __init__(self,ctor,fields,type_name=""):
        self.ctor=ctor; self.fields=tuple(fields); self.type_name=type_name
        self.size=1+sum(f.size if isinstance(f,ADT) else 1 for f in self.fields)
        # height: leaf=1
        hs=[f.height if isinstance(f,ADT) else 0 for f in self.fields]
        self.height=1+(max(hs) if hs else 0)
    def __eq__(self,o): return isinstance(o,ADT) and self.ctor==o.ctor and self.fields==o.fields
    def __hash__(self): return hash((self.ctor,self.fields))
    def __repr__(self):
        if not self.fields: return self.ctor
        return f"{self.ctor}({', '.join(repr(f) for f in self.fields)})"

class Out:
    def __init__(self,items=None): self.items=list(items or [])
    def __repr__(self): return "; ".join(repr(i) for i in self.items)

MACHINE={"i64":(64,True),"u64":(64,False),"i32":(32,True),"u32":(32,False),"u8":(8,False)}
def wrap(v,ty):
    if ty in MACHINE:
        bits,signed=MACHINE[ty]; mod=1<<bits; v%=mod
        if signed and v>= (1<<(bits-1)): v-=mod
        return v
    return v
def is_machine(ty): return isinstance(ty,str) and ty in MACHINE
def is_int_ty(ty): return ty=="int" or is_machine(ty)
def fmt(v):
    if isinstance(v,bool): return "true" if v else "false"
    if isinstance(v,ADT): return repr(v)
    if isinstance(v,tuple): return "("+", ".join(fmt(x) for x in v)+")"
    if isinstance(v,list): return "["+", ".join(fmt(x) for x in v)+"]"
    if isinstance(v,Out): return "\n".join(fmt(i) for i in v.items)
    return str(v)
