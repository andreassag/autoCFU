.PHONY: help install pre-commit test lint format clean act update-deps

VENV ?= .venv
PYTHON ?= $(VENV)/bin/python
UV ?= uv
PRECOMMIT ?= $(VENV)/bin/pre-commit
PYTEST ?= $(VENV)/bin/pytest
RUFF ?= $(VENV)/bin/ruff

help:
	@echo "Available commands:"
	@echo "  make install         - Create virtualenv, install dependencies, and setup pre-commit"
	@echo "  make pre-commit      - Run pre-commit checks on all files"
	@echo "  make test            - Run unit tests on the inference engine"
	@echo "  make lint            - Run linter checks (ruff)"
	@echo "  make format          - Format python code (ruff)"
	@echo "  make clean           - Remove build, cache, and test artifacts"
	@echo "  make act             - Run GitHub Actions CI locally using act"
	@echo "  make update-deps     - Upgrade dependencies and run formatting, linting, and tests"

install:
	@if command -v $(UV) >/dev/null 2>&1; then \
		echo "Creating virtual environment and installing with $(UV)..."; \
		$(UV) venv --allow-existing $(VENV) --python 3.12 2>/dev/null || $(UV) venv --allow-existing $(VENV); \
		$(UV) pip install --python $(PYTHON) -e "./inference[dev]"; \
	else \
		echo "Creating virtual environment with python3..."; \
		python3 -m venv $(VENV); \
		$(PYTHON) -m pip install --upgrade pip; \
		$(PYTHON) -m pip install -e "./inference[dev]"; \
	fi
	@echo "Installing pre-commit git hooks..."
	@if [ -f $(PRECOMMIT) ]; then \
		$(PRECOMMIT) install; \
	elif command -v pre-commit >/dev/null 2>&1; then \
		pre-commit install; \
	fi

pre-commit:
	@if [ -f $(PRECOMMIT) ]; then \
		$(PRECOMMIT) run --all-files; \
	elif command -v pre-commit >/dev/null 2>&1; then \
		pre-commit run --all-files; \
	else \
		echo "pre-commit not found. Run 'make install' first."; exit 1; \
	fi

test:
	@if [ -f $(PYTEST) ]; then \
		$(PYTEST) inference/tests; \
	else \
		PYTHONPATH=inference python3 -m unittest discover -s inference/tests; \
	fi

lint:
	@if [ -f $(RUFF) ]; then \
		$(RUFF) check inference/; \
	elif command -v ruff >/dev/null 2>&1; then \
		ruff check inference/; \
	else \
		echo "ruff not found. Run 'make install' first."; exit 1; \
	fi

format:
	@if [ -f $(RUFF) ]; then \
		$(RUFF) format inference/; \
	elif command -v ruff >/dev/null 2>&1; then \
		ruff format inference/; \
	else \
		echo "ruff not found. Run 'make install' first."; exit 1; \
	fi

clean:
	@rm -rf build/ dist/ *.egg-info inference/*.egg-info .pytest_cache .ruff_cache inference/.pytest_cache inference/.ruff_cache
	@find . -type d -name "__pycache__" -exec rm -rf {} +

act:
	@if command -v gh >/dev/null 2>&1 && gh act --version >/dev/null 2>&1; then \
		gh act pull_request; \
	elif command -v act >/dev/null 2>&1; then \
		act pull_request; \
	else \
		echo "Neither 'act' nor 'gh act' extension found. Please install act (https://github.com/nektos/act) or run 'gh extension install nektos/gh-act'."; exit 1; \
	fi

update-deps:
	@chmod +x scripts/update_dependencies.sh
	@./scripts/update_dependencies.sh
