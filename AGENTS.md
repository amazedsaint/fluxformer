# Repository Guidelines

## Project Structure & Module Organization
- `fluxformer/`: PyTorch modules (e.g., `rev_coupling.py`, `model_v2.py`).
- `baselines/`: NumPy reference implementations and demos.
- `tools/`: Small utilities (e.g., `schedules.py`).
- `tests/`: Script-style tests runnable directly with Python.
- `README.md`, `WHITEPAPER.md`: Usage notes and design details.

## Build, Test, and Development Commands
- Test (NumPy only): `python tests/test_numpy_qwalk_1d.py` (see `README.md` for the full list).
- Run all tests: `for f in tests/test_*.py; do python "$f"; done` (bash).
- No build step is required. Install deps you need (e.g., `numpy`, optional `torch`) in your environment.

## Coding Style & Naming Conventions
- Python 3.10+; 4-space indentation; PEP 8 naming.
- Modules/files: `lower_snake_case.py`; functions/variables: `lower_snake_case`;
  classes: `CamelCase`.
- Prefer small, pure functions and clear tensor/array shapes in docstrings.
- If using formatters/linters, keep diffs minimal; suggested: `black`, `ruff` (optional).

## Testing Guidelines
- Tests are simple scripts with assertions; no framework required.
- Place new tests in `tests/` as `test_<feature>.py` and ensure they print a short OK message.
- Keep tests fast and deterministic (fix RNG seeds where applicable).
- Example: `python tests/test_numpy_rev_coupling.py`.

## Commit & Pull Request Guidelines
- Use clear, scoped commits. Recommended convention: Conventional Commits, e.g., `feat: add RevCoupling inverse`, `fix: stabilize q-walk test`.
- PRs should include: purpose, key changes, how to run tests, and any performance or numerical notes.
- Link related issues or paper sections (`WHITEPAPER.md`) when relevant. Add before/after snippets or metrics if behavior changes.

## Security & Configuration Tips
- Avoid adding heavy dependencies to preserve lightweight setup.
- Guard optional PyTorch code paths; NumPy tests must run without `torch` installed.
- Seed RNGs in examples/tests for reproducibility.
