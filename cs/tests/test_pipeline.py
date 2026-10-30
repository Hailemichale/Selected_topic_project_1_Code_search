import pytest

from src.baseline import BM25Search, tokenize
from src.data import extract_functions, split
from src.metrics import recall_at_k, reciprocal_rank

SAMPLE = '''
def to_datetime_obj(s):
    """Parse a date string into a datetime object."""
    return s

def add_numbers(a, b):
    """Add two numbers and return the sum."""
    return a + b

def shout(text):
    """Return the text in upper case."""
    return text.upper()
'''


@pytest.fixture
def records(tmp_path):
    (tmp_path / "m.py").write_text(SAMPLE)
    return extract_functions(tmp_path)


def test_tokenize_splits_names():
    assert tokenize("parseDate_now") == ["parse", "date", "now"]


def test_extract_removes_docstring_from_code(records):
    assert len(records) == 3
    assert all('"""' not in r["code"] for r in records)


def test_metrics():
    assert recall_at_k(["a", "b", "c"], "b", 1) == 0.0
    assert recall_at_k(["a", "b", "c"], "b", 2) == 1.0
    assert reciprocal_rank(["a", "b", "c"], "b") == 0.5
    assert reciprocal_rank(["a"], "z") == 0.0


def test_bm25_finds_obvious_match(records):
    search = BM25Search(records)
    best = search.search("add two numbers and return the sum", k=1)[0]
    assert "add_numbers" in best


def test_split_is_repeatable_and_disjoint(records):
    d1, t1 = split(records, seed=1)
    d2, t2 = split(records, seed=1)
    assert [r["id"] for r in d1] == [r["id"] for r in d2]
    assert not {r["id"] for r in d1} & {r["id"] for r in t1}


def test_embedding_search_runs(records):
    pytest.importorskip("sentence_transformers")
    from src.embed import EmbeddingSearch
    try:
        search = EmbeddingSearch(records)
    except Exception as e:  # model download can fail offline
        pytest.skip(f"model unavailable: {e}")
    assert len(search.search("add two numbers", k=2)) == 2
