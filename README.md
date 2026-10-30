# Code Search

Project 1 for **Selected Topics in Software Engineering** — Addis Ababa University, Department of Software Engineering.

## Overview

Developers joining a new codebase often know *what* a function should do but not *what it is called*. Typing `"parse a date"` into a keyword search returns nothing when the function is actually named `to_datetime_obj`. This project tackles that gap by building and comparing two code-search approaches:

1. **BM25 keyword baseline** — a classical information-retrieval method that scores functions by term overlap.
2. **Transformer-based semantic search** — a pretrained sentence encoder (`all-MiniLM-L6-v2`) that matches the *meaning* of a query to the *meaning* of source code using cosine similarity.

The system parses Python source files, extracts every function that has a docstring, and uses the first line of the docstring as a natural-language query. It then measures how well each search method can retrieve the correct function given that query.

## Project Architecture

```
cs/
├── src/
│   ├── data.py          # AST-based function extraction & dataset splitting
│   ├── baseline.py      # BM25 keyword search (rank-bm25)
│   ├── embed.py         # Transformer search (sentence-transformers + cosine sim)
│   ├── search.py        # CLI entry point for interactive search
│   └── metrics.py       # Recall@K and MRR@10 metric functions
├── eval/
│   └── evaluate.py      # End-to-end evaluation on dev/test splits
├── tests/
│   └── test_pipeline.py # Unit tests for the pipeline
├── results/             # Auto-generated evaluation outputs (JSON + CSV)
├── demo/                # Demo materials
├── requirements.txt     # Python dependencies
└── pytest.ini           # Test configuration
```

### How It Works

1. **Data Extraction** (`src/data.py`): Walks a Python codebase, parses each `.py` file with the `ast` module, and extracts every function/async-function that has a docstring. The first line of the docstring becomes the *query*; the function body (with docstring stripped) becomes the *code*. Duplicate queries are removed to ensure each query has a unique answer.

2. **BM25 Baseline** (`src/baseline.py`): Tokenizes function names and code using camelCase/snake_case splitting, then builds a BM25Okapi index. Queries are tokenized the same way and scored against the index.

3. **Transformer Search** (`src/embed.py`): Encodes every function's name + code into a dense vector using `all-MiniLM-L6-v2`. At query time, the query is embedded with the same model and ranked by cosine similarity against all function vectors.

4. **Evaluation** (`eval/evaluate.py`): Splits the dataset into dev/test with a fixed seed. Runs both search methods on the chosen split and reports **Recall@1**, **Recall@5**, and **MRR@10**. Failed queries are logged to CSV for error analysis.

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.x |
| Keyword Search | [rank-bm25](https://github.com/dorianbrown/rank_bm25) (BM25Okapi) |
| Semantic Search | [sentence-transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`) |
| Code Parsing | Python `ast` module |
| Numerical Ops | NumPy |
| Testing | pytest |

## How to Run

### Setup
```bash
# Create and activate a virtual environment
python -m venv .venv
# Linux/Mac:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Tests
```bash
pytest
```

### Interactive Search
```bash
# Search a Python codebase using the Transformer method (default)
python -m src.search "parse a date" --repo path/to/code --method embed

# Search using the BM25 baseline
python -m src.search "parse a date" --repo path/to/code --method bm25
```

### Evaluation
```bash
# Evaluate on the dev split (for tuning)
python -m eval.evaluate --repo path/to/code

# Final evaluation on the held-out test split (run once)
python -m eval.evaluate --repo path/to/code --on test
```

Results and failed-query logs are written to the `results/` directory.  
Everything runs on a laptop CPU or a free Google Colab session.

## Results (Held-Out Test Split)

| Method | Recall@1 | Recall@5 | MRR@10 |
|---|---|---|---|
| BM25 baseline | — | — | — |
| Transformer | — | — | — |

> *Results will be populated after final evaluation.*

## Failure Analysis

*Three real examples from `results/failures_test_*.csv` — the query, what was returned, and why the method failed — will be documented here after evaluation.*

## Limitations

- **Token limit**: MiniLM reads only the first 256 tokens, so long functions are truncated.
- **Python only**: The current pipeline only processes `.py` files.
- **Docstring dependency**: Functions without docstrings are excluded from the dataset.
- **Single-language queries**: Queries are assumed to be in English.

## Team Members

| Name | ID |
|---|---|
| Peter Kinfe | ATE/7749/15 |
| Abdisa Alemu | ATE/2603/14 |
| Hailemichale Lijalem | ATE/1051/14 |
| Beamlak Tibebu | ATE/3624/14 |
| Loza Wondwossen | ATE/5401/14 |
| Gerawork Aklilu | ATE/3243/14 |

## References

- Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS.
- Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP.
- Robertson, S. & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*. Foundations and Trends in Information Retrieval.
