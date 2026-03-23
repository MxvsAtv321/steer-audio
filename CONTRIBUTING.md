# Contributing to steer-audio

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

Core tests (no optional dependencies required):

```bash
pytest -m "not optional"
```

Full suite including pandas/gradio tests:

```bash
pip install pandas gradio
pytest
```

## Linting

```bash
ruff check .
ruff format --check .
```

Fix automatically:

```bash
ruff check --fix .
ruff format .
```

## CI

The GitHub Actions workflow runs on every push and pull request:

- **Python matrix:** 3.10 and 3.11
- **Lint:** `ruff check` + `ruff format --check`
- **Tests:** `pytest -m "not optional"` with coverage
- **Coverage:** reported via `pytest-cov`; threshold is not enforced (informational)

Tests marked `@pytest.mark.optional` require `pandas` or `gradio` and are excluded from CI to keep the matrix fast. Install them locally with `pip install -e ".[dev]"` to run the full suite.

## Code Style

- Type hints and docstrings on all new public functions.
- No hardcoded paths — use `pathlib.Path` and `TADA_WORKDIR` env var.
- Keep diffs small and focused; one PR per logical change.
