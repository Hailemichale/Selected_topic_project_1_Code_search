"""Turn a folder of Python code into search records.

Each record is one function that has a docstring:
  query = first line of the docstring (what a developer would type)
  code  = the function source WITHOUT its docstring (so the answer is not leaked)
"""
import ast
import random
from collections import Counter
from pathlib import Path


def extract_functions(root, skip_dirs=("test", "tests"), min_query_words=3):
    records = []
    for path in sorted(Path(root).rglob("*.py")):
        if any(part in skip_dirs for part in path.parts):
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
            lines_doc = doc.strip().splitlines()
            if not lines_doc:
                continue
            query = lines_doc[0].strip()
            if len(query.split()) < min_query_words:
                continue
            d = node.body[0]  # the docstring statement
            code = "\n".join(lines[node.lineno - 1:d.lineno - 1] + lines[d.end_lineno:node.end_lineno])
            records.append({"id": f"{path.name}:{node.name}:{node.lineno}", "name": node.name,
                            "query": query, "code": code})
    # identical queries would have more than one right answer, so drop them
    counts = Counter(r["query"] for r in records)
    return [r for r in records if counts[r["query"]] == 1]


def split(records, dev_frac=0.5, seed=42):
    """Fixed, repeatable split. Tune on dev. Touch test only for the final numbers."""
    recs = list(records)
    random.Random(seed).shuffle(recs)
    n = int(len(recs) * dev_frac)
    return recs[:n], recs[n:]
