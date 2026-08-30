# Contributing

Thanks for helping improve the PEFT examples. Small, focused changes with a test and a clear
explanation are easiest to review.

## Set up the project

You need Python 3.11–3.13, [uv](https://docs.astral.sh/uv/), and Node.js 22 or newer.

```bash
uv sync --extra dev
npm --prefix site ci
uv run pre-commit install --hook-type pre-commit --hook-type pre-push
```

The pre-commit hook formats and lints changed Python files, checks the uv lockfile, and validates
the Astro site when its files change. The pre-push hook additionally runs the complete test suite
and production site build.

## Make a change

- Add PEFT configurations to `src/transformers_tuning/methods.py` and expose them through the
  existing registry instead of creating a separate training script.
- Use defaults that run on a small decoder-only language model and document architecture-specific
  assumptions.
- Add a configuration test for every method and a forward-pass test for each new method family.
- Update both `README.md` and the relevant page under `site/src/pages` when user behavior changes.
- Never commit model weights, datasets, credentials, `.venv`, `site/node_modules`, or `artifacts`.

Run all checks before opening a pull request:

```bash
uv lock --check
uv run ruff check .
uv run ruff format --check .
uv run pytest
npm --prefix site run check
npm --prefix site run build
```

## Pull requests and releases

Pull requests should describe the motivation, validation performed, and any model-specific
limitations. CI tests all supported Python versions and builds the documentation.

Maintainers create releases by updating `project.version` in `pyproject.toml`, updating
`CHANGELOG.md`, and pushing a matching annotated tag such as `v0.2.0`. The release workflow checks
the version, builds the wheel and source distribution, uploads them as workflow artifacts, and
creates a GitHub Release. It does not publish to PyPI.

