from .lexer import tokenize
from . import nodes as N
import re

class ParseError(Exception): pass

def parse_int_raw(v):
    suf=""
    for s in ("i64","u64","i32","u32","u8"):
        if v.endswith(s) and not v.startswith(("0x","0b")):
            # ensure suffix not part of hex (decimal only)
            suf=s; v=v[:-len(s)]; break
        if v.endswith(s) and v.startswith("0x"):
            suf=s; v=v[:-len(s)]; break
    clean=v.replace("_","")
    if clean.startswith("0x"): val=int(clean,16)
    elif clean.startswith("0b"): val=int(clean,2)
    else: val=int(clean)
    return val,suf

class Parser:
    def __init__(self,toks): self.toks=toks; self.i=0
    def peek(self,k=0): return self.toks[min(self.i+k,len(self.toks)-1)]
    def next(self):
        t=self.toks[self.i]; self.i+=1; return t
    def accept(self,kind):
        if self.peek().kind==kind: return self.next()
        return None
    def expect(self,kind):
        t=self.peek()
        if t.kind!=kind: raise ParseError(f"line {t.line}: expected {kind!r}, got {t.kind!r} ({t.val!r})")
        return self.next()
    def skip_nl(self):
        while self.peek().kind=='NEWLINE': self.next()

    def parse_program(self):
        decls=[]
        self.skip_nl()
        while self.peek().kind!='EOF':
            decls.append(self.parse_decl()); self.skip_nl()
        return N.Program(decls=decls,line=1)

    def parse_decl(self):
        if self.peek().kind=='type': return self.parse_typedecl()
        anns=[]
        while self.peek().kind=='@':
            self.next(); anns.append('@'+self.expect('NAME').val)
        return self.parse_fndecl(anns)

    def parse_typedecl(self):
        t=self.expect('type'); name=self.expect('NAME').val
        self.expect('='); variants=[self.parse_variant()]
        while self.accept('|'): variants.append(self.parse_variant())
        return N.TypeDecl(name=name,variants=variants,line=t.line)

    def parse_variant(self):
        name=self.expect('NAME').val; tys=[]
        if self.accept('('):
            tys.append(self.parse_type())
            while self.accept(','): tys.append(self.parse_type())
            self.expect(')')
        return (name,tys)

    def parse_fndecl(self,anns):
        self.skip_nl()
        t=self.expect('fn'); name=self.expect('NAME').val
        self.expect('('); params=[]
        if self.peek().kind!=')':
            params.append(self.parse_param())
            while self.accept(','): params.append(self.parse_param())
        self.expect(')'); self.expect('->'); ret=self.parse_type(); self.expect(':')
        body=self.parse_block()
        return N.FnDecl(name=name,params=params,ret=ret,body=body,annotations=anns,line=t.line)

    def parse_param(self):
        n=self.expect('NAME'); self.expect(':'); return (n.val,self.parse_type())

    def parse_type(self):
        t=self.peek()
        if t.kind=='[':
            self.next(); inner=self.parse_type(); self.expect(']'); return ('list',inner)
        if t.kind=='(':
            self.next(); items=[self.parse_type()]
            while self.accept(','): items.append(self.parse_type())
            self.expect(')')
            return items[0] if len(items)==1 else ('tuple',items)
        name=self.expect('NAME').val
        return name

    def parse_block(self):
        self.skip_nl()
        if self.peek().kind=='let':
            bindings=[]
            while self.peek().kind=='let':
                lt=self.next(); pat=self.parse_pattern(); self.expect('='); e=self.parse_expr()
                bindings.append((pat,e)); self.skip_nl()
            e=self.parse_expr()
            return N.Block(bindings=bindings,expr=e,line=e.line if e else 1)
        if self.peek().kind=='INDENT':
            self.next(); bindings=[]
            self.skip_nl()
            while self.peek().kind=='let':
                lt=self.next(); pat=self.parse_pattern(); self.expect('='); e=self.parse_expr()
                bindings.append((pat,e)); self.skip_nl()
            e=self.parse_expr(); self.skip_nl(); self.expect('DEDENT')
            return N.Block(bindings=bindings,expr=e,line=e.line if e else 1)
        e=self.parse_expr()
        return N.Block(bindings=[],expr=e,line=e.line if e else 1)

    # ---- patterns
    def parse_pattern(self):
        t=self.peek()
        if t.kind=='INT':
            self.next(); v,s=parse_int_raw(t.val); return N.PLit(value=v,raw=t.val,line=t.line)
        if t.kind in ('true','false'):
            self.next(); return N.PLit(value=(t.kind=='true'),line=t.line)
        if t.kind=='(':
            self.next(); items=[self.parse_pattern()]
            while self.accept(','): items.append(self.parse_pattern())
            self.expect(')'); return N.PTuple(items=items,line=t.line)
        if t.kind=='NAME':
            self.next()
            if t.val=='_': return N.PWild(line=t.line)
            if t.val[0].isupper():
                items=[]
                if self.accept('('):
                    items.append(self.parse_pattern())
                    while self.accept(','): items.append(self.parse_pattern())
                    self.expect(')')
                return N.PCtor(name=t.val,items=items,line=t.line)
            return N.PVar(name=t.val,line=t.line)
        raise ParseError(f"line {t.line}: bad pattern {t.kind} {t.val!r}")

    # ---- expressions
    def parse_expr(self):
        # lambda lookahead
        if self.peek().kind=='(' and self._lambda_ahead():
            return self.parse_lambda()
        if self.peek().kind=='if': return self.parse_if()
        if self.peek().kind=='match': return self.parse_match()
        if self.peek().kind=='treefold': return self.parse_treefold()
        e=self.parse_or()
        if self.peek().kind==';':
            items=[e]
            while self.accept(';'): items.append(self.parse_expr())
            return N.SeqExpr(items=items,line=e.line)
        return e

    def _lambda_ahead(self):
        depth=0; j=self.i
        while j < len(self.toks):
            k=self.toks[j].kind
            if k=='(': depth+=1
            elif k==')':
                depth-=1
                if depth==0: return j+1 < len(self.toks) and self.toks[j+1].kind=='=>'
            j+=1
        return False

    def parse_lambda(self):
        t=self.expect('('); params=[]
        if self.peek().kind!=')':
            params.append(self.parse_pattern())
            while self.accept(','): params.append(self.parse_pattern())
        self.expect(')'); self.expect('=>'); body=self.parse_block()
        return N.Lambda(params=params,body=body,line=t.line)

    def parse_if(self):
        t=self.expect('if'); cond=self.parse_expr(); self.expect('then')
        self.skip_nl()
        then=self.parse_expr()
        self.skip_nl()
        self.expect('else'); self.skip_nl(); els=self.parse_expr()
        return N.If(cond=cond,then=then,els=els,line=t.line)

    def _parse_arms(self):
        self.skip_nl()
        arms=[]
        if self.accept('INDENT'):
            self.skip_nl()
            while self.peek().kind!='DEDENT':
                pat=self.parse_pattern(); self.expect('=>'); blk=self.parse_block()
                arms.append((pat,blk)); self.skip_nl()
            self.expect('DEDENT')
            return arms
        # arms inside brackets: NEWLINE-separated, terminated by ')' / EOF
        while self.peek().kind not in (')','EOF','DEDENT'):
            pat=self.parse_pattern(); self.expect('=>'); blk=self.parse_block()
            arms.append((pat,blk)); self.skip_nl()
        return arms

    def parse_match(self):
        t=self.expect('match'); scrut=self.parse_expr(); self.expect(':')
        return N.Match(scrut=scrut,arms=self._parse_arms(),line=t.line)

    def parse_treefold(self):
        t=self.expect('treefold'); scrut=self.parse_expr(); self.expect(':')
        return N.TreeFold(scrut=scrut,arms=self._parse_arms(),line=t.line)

    def _bin(self,sub,ops):
        e=sub()
        while self.peek().kind in ops:
            op=self.next(); r=sub(); e=N.BinOp(op=op.kind,l=e,r=r,line=op.line)
        return e
    def parse_or(self): return self._bin(self.parse_and,('or',))
    def parse_and(self): return self._bin(self.parse_cmp,('and',))
    def parse_cmp(self): return self._bin(self.parse_bitor,('==','!=','<','<=','>','>='))
    def parse_bitor(self): return self._bin(self.parse_bitxor,('|',))
    def parse_bitxor(self): return self._bin(self.parse_bitand,('^',))
    def parse_bitand(self): return self._bin(self.parse_shift,('&',))
    def parse_shift(self): return self._bin(self.parse_add,('<<','>>'))
    def parse_add(self): return self._bin(self.parse_mul,('+','-'))
    def parse_mul(self): return self._bin(self.parse_unary,('*','/','%'))
    def parse_unary(self):
        t=self.peek()
        if t.kind in ('-','not'):
            self.next(); return N.Unary(op=t.kind,e=self.parse_unary(),line=t.line)
        return self.parse_postfix()
    def parse_postfix(self):
        e=self.parse_atom()
        while True:
            if self.peek().kind=='(':
                self.next(); args=[]; self.skip_nl()
                if self.peek().kind!=')':
                    args.append(self.parse_expr()); self.skip_nl()
                    while self.accept(','): self.skip_nl(); args.append(self.parse_expr()); self.skip_nl()
                self.expect(')')
                if isinstance(e,N.CtorExpr) and not e.args:
                    e=N.CtorExpr(name=e.name,args=args,line=e.line)
                else:
                    e=N.Call(f=e,args=args,line=e.line)
            elif self.peek().kind=='[':
                self.next(); idx=self.parse_expr(); self.expect(']')
                e=N.Index(obj=e,idx=idx,line=e.line)
            else: break
        return e
    def parse_atom(self):
        t=self.peek()
        if t.kind=='INT':
            self.next(); v,s=parse_int_raw(t.val); return N.Lit(value=v,raw=t.val,suffix=s,line=t.line)
        if t.kind=='FLOAT':
            self.next(); v=float(t.val.replace("_","").removesuffix("f64")); return N.Lit(value=v,raw=t.val,suffix='f64',line=t.line)
        if t.kind=='STR':
            self.next(); v=t.val[1:-1].encode().decode('unicode_escape'); return N.Lit(value=v,raw=t.val,line=t.line)
        if t.kind in ('true','false'):
            self.next(); return N.Lit(value=(t.kind=='true'),line=t.line)
        if t.kind=='NAME':
            self.next()
            if t.val[0].isupper(): return N.CtorExpr(name=t.val,args=[],line=t.line)
            return N.Var(name=t.val,line=t.line)
        if t.kind=='(':
            self.next(); self.skip_nl()
            if self.accept(')'): return N.TupleExpr(items=[],line=t.line)
            first=self.parse_expr(); items=[first]; self.skip_nl()
            while self.accept(','): self.skip_nl(); items.append(self.parse_expr()); self.skip_nl()
            self.expect(')')
            return first if len(items)==1 else N.TupleExpr(items=items,line=t.line)
        if t.kind=='[':
            self.next(); items=[]; self.skip_nl()
            if self.peek().kind!=']':
                items.append(self.parse_expr()); self.skip_nl()
                while self.accept(','): self.skip_nl(); items.append(self.parse_expr()); self.skip_nl()
            self.expect(']'); return N.ListExpr(items=items,line=t.line)
        raise ParseError(f"line {t.line}: unexpected {t.kind} {t.val!r}")

def parse(src):
    return Parser(tokenize(src)).parse_program()
