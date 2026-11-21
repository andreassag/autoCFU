.PHONY: help install test lint typecheck format docs docs-clean pre-commit clean rust-fmt rust-lint rust-test rust-build rust-doc all-checks update-deps act

PYTHON ?= .venv/bin/python
RUFF ?= .venv/bin/ruff
TY ?= .venv/bin/ty
PYTEST ?= .venv/bin/pytest
SPHINXBUILD ?= .venv/bin/sphinx-build
CARGO ?= cargo

help: ## Show available targets
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Create virtualenv, install package with dev/docs extras, and configure pre-commit
	@if command -v uv >/dev/null 2>&1; then \
		uv venv .venv && uv pip install --python .venv/bin/python -e ".[dev,docs]"; \
	else \
		python3 -m venv .venv && .venv/bin/pip install --upgrade pip && .venv/bin/pip install -e ".[dev,docs]"; \
	fi
	@if [ -d .git ]; then .venv/bin/pre-commit install || true; fi

format: ## Format Python code with ruff
	$(RUFF) format src/ tests/

lint: ## Run ruff lint + ty static type checks
	$(RUFF) format --check src/ tests/
	$(RUFF) check src/ tests/
	$(TY) check src/ tests/

typecheck: ## Run ty static type checking
	$(TY) check src/ tests/

test: ## Run Python unit and parity test suite
	$(PYTEST) tests/ -v

docs: ## Build Sphinx HTML documentation with warnings treated as errors
	$(SPHINXBUILD) -b html -W docs docs/_build/html

docs-clean: ## Remove built Sphinx HTML documentation
	rm -rf docs/_build

rust-fmt: ## Check Rust formatting with rustfmt
	$(CARGO) fmt --all -- --check

rust-lint: ## Lint Rust codebase with clippy (-D warnings)
	$(CARGO) clippy --all-targets -- -D warnings

rust-test: ## Run Rust unit tests across lib and binary targets
	$(CARGO) test --all-targets

rust-build: ## Build release binary of Rust CLI
	$(CARGO) build --release

rust-doc: ## Generate Rust crate documentation
	$(CARGO) doc --no-deps

all-checks: lint test docs rust-fmt rust-lint rust-test rust-doc ## Run full Python & Rust verification suite

pre-commit: ## Run all pre-commit hooks across repository files
	.venv/bin/pre-commit run --all-files

clean: ## Remove Python/Rust/Sphinx build artifacts and caches
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache docs/_build
	find . -type d -name "__pycache__" -exec rm -rf {} +

update-deps: ## Run automated dependency update check
	./scripts/update_dependencies.sh

act: ## Execute GitHub Actions workflow locally using gh act
	gh act pull_request
