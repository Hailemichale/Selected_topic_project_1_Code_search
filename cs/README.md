# Code Search

Find the right Python function by describing it in plain English.
Project 1 for Selected Topics in Software Engineering, Addis Ababa University.

**Demo video:** [add unlisted link here]

## Problem
A developer who is new to a codebase knows what a function should do but not what it is called.
Searching for "parse a date" finds nothing if the function is named `to_datetime_obj`.
We compare keyword search against a pretrained Transformer that matches meaning.

## Methods
- **Baseline:** BM25 over each function's name and code (`src/baseline.py`). No Transformer.
- **Transformer:** `all-MiniLM-L6-v2` sentence embeddings with cosine similarity (`src/embed.py`).

## Data
Each record is one Python function that has a docstring. The query is the first line of the
docstring. The docstring is removed from the code before indexing, so the answer is not leaked.
Functions with identical queries are dropped, because they would have two right answers.
The data is split once with a fixed seed into **dev** (for tuning) and **test** (held out).

Dataset used: [fill in, e.g. standard library / project name / number of functions]

## How to run
```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest                                               # tests for the pipeline

# search a folder of Python code
python -m src.search "parse a date" --repo path/to/code --method embed

# evaluate on dev while tuning
python -m eval.evaluate --repo path/to/code
# final numbers, run once on the held-out test split
python -m eval.evaluate --repo path/to/code --on test
```
Results and the list of failed queries are written to `results/`.
Everything runs on a laptop CPU or a free Colab session.

## Results (held-out test split)
| Method | Recall@1 | Recall@5 | MRR@10 |
|---|---|---|---|
| BM25 baseline | [ ] | [ ] | [ ] |
| Transformer | [ ] | [ ] | [ ] |

## Failure analysis
[Three real examples from `results/failures_test_*.csv`: the query, what was returned, and why.]

## Limits
MiniLM reads only the first 256 tokens, so long functions are cut off.

## Team
| Member | Part |
|---|---|
| [name] | Data and test set |
| [name] | BM25 baseline |
| [name] | Transformer search |
| [name] | Evaluation and plots |
| [name] | Demo |
| [name] | Tests, README, video |

## AI tools
[State where you used AI tools, and make sure each of you can explain every line.]

## References
- Vaswani et al. (2017). Attention Is All You Need. NeurIPS.
