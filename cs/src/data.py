"""Turn a folder of Python code into search records.

Each record is one function that has a docstring:
  query = first sentence/paragraph of the docstring (what a developer would type)
  code  = the function source WITHOUT its docstring (so the answer is not leaked)
"""
import ast
import random
import re
from collections import Counter
from pathlib import Path


def _first_paragraph(doc):
    for block in re.split(r"\n\s*\n", doc.strip()):
        text = " ".join(block.split())
        if text:
            return text
    return ""


def _query_from_docstring(doc, min_query_words=3, max_query_words=40):
    text = _first_paragraph(doc).replace("`", "")
    if not text:
        return ""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    query = sentences[0] if sentences else text
    if len(query.split()) < min_query_words:
        query = text
    words = query.split()
    if len(words) > max_query_words:
        query = " ".join(words[:max_query_words])
    return query.strip()


def _dedup_key(query):
    return " ".join(re.findall(r"[A-Za-z0-9_]+", query.lower()))


def extract_functions(root, skip_dirs=("test", "tests"), min_query_words=3):
    records = []
    root_path = Path(root)
    for path in sorted(root_path.rglob("*.py")):
        try:
            rel_path = path.relative_to(root_path).as_posix()
        except ValueError:
            rel_path = path.name
        if any(part in skip_dirs for part in Path(rel_path).parts[:-1]):
            continue
        try:
            src = path.read_text(encoding="utf-8")
            tree = ast.parse(src)
        except (SyntaxError, UnicodeDecodeError, ValueError):
            continue
        lines = src.splitlines()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            doc = ast.get_docstring(node)
            if not doc:
                continue
            query = _query_from_docstring(doc, min_query_words=min_query_words)
            if len(query.split()) < min_query_words:
                continue
            d = node.body[0]  # the docstring statement
            code = "\n".join(lines[node.lineno - 1:d.lineno - 1] + lines[d.end_lineno:node.end_lineno])
            records.append({"id": f"{rel_path}:{node.name}:{node.lineno}", "name": node.name,
                            "query": query, "code": code})
    # identical normalized queries would have more than one right answer, so drop them
    counts = Counter(_dedup_key(r["query"]) for r in records)
    return [r for r in records if counts[_dedup_key(r["query"])] == 1]


def split(records, dev_frac=0.5, seed=42):
    """Fixed, repeatable split. Tune on dev. Touch test only for the final numbers."""
    recs = list(records)
    random.Random(seed).shuffle(recs)
    n = int(len(recs) * dev_frac)
    return recs[:n], recs[n:]
