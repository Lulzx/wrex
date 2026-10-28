import re
from dataclasses import dataclass

KEYWORDS = {"fn","type","let","if","then","else","match","treefold","true","false","and","or","not","family"}
class LexError(Exception): pass

@dataclass
class Tok:
    kind: str; val: str; line: int

def _strip_comment(line):
    in_str=False; i=0
    while i < len(line):
        c=line[i]
        if in_str:
            if c=='\\': i+=2; continue
            if c=='"': in_str=False
        else:
            if c=='"': in_str=True
            elif c=='#': return line[:i]
        i+=1
    return line

def logical_lines(src):
    """Join lines inside unclosed brackets; reject tabs. Returns [(indent,text,lineno)]."""
    out=[]; buf=None; start=1; indent=0; depth=0
    for ln, raw in enumerate(src.splitlines(),1):
        if '\t' in raw: raise LexError(f"line {ln}: tabs are rejected")
        line=_strip_comment(raw)
        if buf is None:
            if line.strip()=="": continue
            indent=len(line)-len(line.lstrip(' '))
            buf=line.strip(); start=ln; depth=0
        else:
            buf+="\n"+line.strip()
        # count depth ignoring strings
        in_str=False; i=0
        while i < len(line):
            c=line[i]
            if in_str:
                if c=='\\': i+=2; continue
                if c=='"': in_str=False
            else:
                if c=='"': in_str=True
                elif c in '([': depth+=1
                elif c in ')]': depth-=1
            i+=1
        if depth<=0:
            out.append((indent,buf,start)); buf=None
    if buf is not None: out.append((indent,buf,start))
    return out

TOKEN_RE = re.compile(r'''
    (?P<ws>\s+)
  | (?P<float>(?:\d[\d_]*\.\d[\d_]*(?:[eE][+-]?\d+)?|\d[\d_]*[eE][+-]?\d+)(?:f64)?)
  | (?P<int>(?:0x[0-9a-fA-F_]+|0b[01_]+|\d[\d_]*)(?:i64|u64|i32|u32|u8)?)
  | (?P<str>"(?:\\.|[^"\\])*")
  | (?P<name>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<op>=>|==|!=|<=|>=|<<|>>|->|[+\-*/%<>&|^,():=\[\];@])
''', re.VERBOSE)

SUFFIXES = ["i64","u64","i32","u32","u8","f64"]

def _tokenize_segment(text,toks,ln):
    pos=0
    for m in TOKEN_RE.finditer(text):
        if m.start()!=pos: raise LexError(f"line {ln}: unexpected character {text[pos]!r}")
        pos=m.end()
        kind=m.lastgroup; v=m.group()
        if kind=='ws': continue
        if kind=='name' and v in KEYWORDS: toks.append(Tok(v,v,ln))
        elif kind=='name': toks.append(Tok('NAME',v,ln))
        elif kind=='int': toks.append(Tok('INT',v,ln))
        elif kind=='float': toks.append(Tok('FLOAT',v,ln))
        elif kind=='str': toks.append(Tok('STR',v,ln))
        else: toks.append(Tok(v,v,ln))
    if pos!=len(text): raise LexError(f"line {ln}: unexpected character {text[pos]!r}")

def tokenize(src):
    lines=logical_lines(src)
    toks=[]; indents=[0]
    for indent,text,ln in lines:
        if indent>indents[-1]:
            indents.append(indent); toks.append(Tok('INDENT','',ln))
        elif indent<indents[-1]:
            while indents[-1]>indent:
                indents.pop(); toks.append(Tok('DEDENT','',ln))
            if indents[-1]!=indent: raise LexError(f"line {ln}: inconsistent indentation")
        # physical lines joined inside brackets keep NEWLINE separators
        segments=text.split('\n')
        for si,seg in enumerate(segments):
            if si>0: toks.append(Tok('NEWLINE','',ln))
            _tokenize_segment(seg,toks,ln)
        toks.append(Tok('NEWLINE','',ln))
    last=lines[-1][2] if lines else 1
    while len(indents)>1:
        indents.pop(); toks.append(Tok('DEDENT','',last))
    toks.append(Tok('EOF','',last))
    return toks
