from dataclasses import dataclass, field
from typing import Any

@dataclass
class Node:
    line: int = 1
    ty: Any = None  # filled by checker

@dataclass
class Program(Node):
    decls: list = field(default_factory=list)

@dataclass
class TypeDecl(Node):
    name: str = ""
    variants: list = field(default_factory=list)  # [(ctor,[type,...])]

@dataclass
class FamilyDecl(Node):
    name: str = ""
    state_ty: Any = None
    identity: Any = None  # expr
    compose: Any = None  # Lambda with 2 params

@dataclass
class FnDecl(Node):
    name: str = ""
    params: list = field(default_factory=list)  # [(name,type)]
    ret: Any = None
    body: Any = None
    annotations: list = field(default_factory=list)

# types are tuples/strings: ('list',t) ('tuple',[..]) or str name
@dataclass
class Block(Node):
    bindings: list = field(default_factory=list)  # [(pattern,expr)]
    expr: Any = None

@dataclass
class Lambda(Node):
    params: list = field(default_factory=list)  # patterns
    body: Any = None

@dataclass
class If(Node):
    cond: Any = None; then: Any = None; els: Any = None

@dataclass
class Match(Node):
    scrut: Any = None; arms: list = field(default_factory=list)  # [(pattern,block)]

@dataclass
class TreeFold(Node):
    scrut: Any = None; arms: list = field(default_factory=list)

@dataclass
class BinOp(Node):
    op: str = ""; l: Any = None; r: Any = None

@dataclass
class Unary(Node):
    op: str = ""; e: Any = None

@dataclass
class Call(Node):
    f: Any = None; args: list = field(default_factory=list)

@dataclass
class Index(Node):
    obj: Any = None; idx: Any = None

@dataclass
class Var(Node):
    name: str = ""

@dataclass
class CtorExpr(Node):
    name: str = ""; args: list = field(default_factory=list)

@dataclass
class Lit(Node):
    value: Any = None; raw: str = ""; suffix: str = ""

@dataclass
class TupleExpr(Node):
    items: list = field(default_factory=list)

@dataclass
class ListExpr(Node):
    items: list = field(default_factory=list)

@dataclass
class SeqExpr(Node):
    items: list = field(default_factory=list)

# patterns
@dataclass
class PVar(Node):
    name: str = ""
@dataclass
class PWild(Node):
    pass
@dataclass
class PLit(Node):
    value: Any = None; raw: str = ""
@dataclass
class PTuple(Node):
    items: list = field(default_factory=list)
@dataclass
class PCtor(Node):
    name: str = ""; items: list = field(default_factory=list)
