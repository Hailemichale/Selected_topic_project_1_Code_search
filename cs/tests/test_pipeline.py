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


def test_query_joins_wrapped_docstring_lines(tmp_path):
    (tmp_path / "m.py").write_text(
        "def parse_date(s):\n"
        '    """Parse a date string\n'
        '    into a datetime object."""\n'
        "    return s\n"
    )
    records = extract_functions(tmp_path)
    assert len(records) == 1
    assert records[0]["query"] == "Parse a date string into a datetime object."


def test_query_prefers_first_sentence(tmp_path):
    (tmp_path / "m.py").write_text(
        "def add(a, b):\n"
        '    """Return the sum.\n'
        '    This function adds two numbers."""\n'
        "    return a + b\n"
    )
    records = extract_functions(tmp_path)
    assert len(records) == 1
    assert records[0]["query"] == "Return the sum."


def test_normalized_duplicate_queries_are_dropped(tmp_path):
    (tmp_path / "m.py").write_text(
        "def a():\n"
        '    """Return the sum."""\n'
        "    return 1\n"
        "def b():\n"
        '    """return the sum"""\n'
        "    return 2\n"
        "def c():\n"
        '    """Multiply two numbers."""\n'
        "    return 3\n"
    )
    records = extract_functions(tmp_path)
    assert [r["name"] for r in records] == ["c"]


def test_record_id_uses_relative_path(tmp_path):
    sub = tmp_path / "pkg"
    sub.mkdir()
    (sub / "mod.py").write_text(
        "def f():\n"
        '    """Do a useful thing here."""\n'
        "    return 1\n"
    )
    records = extract_functions(tmp_path)
    assert records[0]["id"].startswith("pkg/mod.py:f:")


def test_skip_dirs_are_relative_to_repo_root(tmp_path):
    tests_root = tmp_path / "tests"
    tests_root.mkdir()
    (tests_root / "m.py").write_text(
        "def f():\n"
        '    """Do a useful thing here."""\n'
        "    return 1\n"
    )
    assert len(extract_functions(tests_root)) == 1

    pkg = tmp_path / "pkg"
    (pkg / "tests").mkdir(parents=True)
    (pkg / "tests" / "bad.py").write_text(
        "def g():\n"
        '    """Do another useful thing."""\n'
        "    return 2\n"
    )
    (pkg / "good.py").write_text(
        "def h():\n"
        '    """Do a third useful thing."""\n'
        "    return 3\n"
    )
    records = extract_functions(pkg)
    assert [r["name"] for r in records] == ["h"]
