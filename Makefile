.PHONY: help test test-verbose coverage coverage-html clean install lint format check build deploy tag install-ci generate-examples

# Default target
.DEFAULT_GOAL := help

# Python and virtualenv
PYTHON := python3
VENV := venv
VENV_BIN := $(VENV)/bin
PIP := $(VENV_BIN)/pip
PYTEST := $(VENV_BIN)/pytest
PYTHONPATH := .
PROJECT_VERSION := $(shell $(PYTHON) -W ignore setup.py --version)

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install package and dependencies in virtualenv
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -e .
	$(PIP) install -e ".[dev]"
	@echo "Virtual environment created. Activate with: source $(VENV)/bin/activate"

test: ## Run tests with coverage
	@source $(VENV_BIN)/activate && PYTHONPATH=$(PYTHONPATH) $(PYTEST)

test-verbose: ## Run tests with verbose output and coverage
	@source $(VENV_BIN)/activate && PYTHONPATH=$(PYTHONPATH) $(PYTEST) -v

test-watch: ## Run tests in watch mode (requires pytest-watch)
	@source $(VENV_BIN)/activate && PYTHONPATH=$(PYTHONPATH) $(VENV_BIN)/ptw

coverage: ## Run tests and show coverage report
	@source $(VENV_BIN)/activate && PYTHONPATH=$(PYTHONPATH) $(PYTEST) --cov-report=term-missing

coverage-html: ## Generate HTML coverage report
	@source $(VENV_BIN)/activate && PYTHONPATH=$(PYTHONPATH) $(PYTEST) --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"
	@open htmlcov/index.html 2>/dev/null || xdg-open htmlcov/index.html 2>/dev/null || echo "Open htmlcov/index.html in your browser"

lint: ## Run code linters (flake8)
	@source $(VENV_BIN)/activate && $(VENV_BIN)/flake8 parcel_ai_json/ tests/ --max-line-length=100 --extend-ignore=E203,W503

format: ## Format code with black
	@source $(VENV_BIN)/activate && $(VENV_BIN)/black parcel_ai_json/ tests/ examples/ scripts/

format-check: ## Check code formatting without making changes
	@source $(VENV_BIN)/activate && $(VENV_BIN)/black --check parcel_ai_json/ tests/ examples/ scripts/

check: format-check lint test ## Run all checks (format, lint, test)

clean: ## Clean up generated files
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	rm -rf parcel_ai_json/__pycache__/
	rm -rf tests/__pycache__/
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

clean-all: clean ## Clean everything including virtualenv
	rm -rf $(VENV)

generate-examples: ## Generate vehicle detection examples
	@source $(VENV_BIN)/activate && PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/generate_examples.py

dev-setup: install ## Set up development environment
	@echo "Development environment ready!"
	@echo "Run 'source $(VENV)/bin/activate' to activate the virtualenv"
	@echo "Run 'make test' to run tests"
	@echo "Run 'make coverage-html' to see coverage report"

# Package build and deployment targets

install-ci: ## Install CI dependencies (twine, wheel)
	$(PIP) install -r requirements_ci.txt
	$(PIP) install setuptools

build: ## Build source and wheel distributions
	$(VENV_BIN)/python setup.py sdist bdist_wheel

deploy: install-ci ## Build and deploy package to internal PyPI
	rm -rf dist/*
	$(VENV_BIN)/python setup.py sdist bdist_wheel
	$(VENV_BIN)/twine upload -r pypi-ergeon dist/* --config-file=./pypirc.conf
	@echo ""
	@echo "Package v$(PROJECT_VERSION) successfully published to https://pypi.ergeon.in/"
	@echo "Install with: pip install parcel-ai-json --extra-index-url=https://erg-bot:q8zgdmot3@pypi.ergeon.in/simple/"

tag: ## Create and push git tag for current version (must be on master)
	@if [ "$$(git rev-parse --abbrev-ref HEAD)" != "master" ]; then \
		echo "Error: Must be on master branch to tag"; \
		exit 1; \
	fi
	@echo "Creating tag v$(PROJECT_VERSION)..."
	git tag -a v$(PROJECT_VERSION) -m '$(PROJECT_VERSION) Release'
	git push --tags
	@echo "Tag v$(PROJECT_VERSION) created and pushed"
