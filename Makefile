.PHONY: help test test-verbose coverage coverage-html clean install lint format check build deploy tag install-ci generate-examples generate-examples-10 generate-examples-20 docker-build docker-build-clean docker-run docker-stop docker-logs docker-shell docker-push docker-clean docker-up docker-down

# Default target
.DEFAULT_GOAL := help

# Python and virtualenv
PYTHON := python3
VENV := venv
VENV_BIN := $(VENV)/bin
PIP := $(VENV_BIN)/pip
PYTEST := $(VENV_BIN)/pytest
PYTHONPATH := .
PROJECT_VERSION := $(shell $(PYTHON) -W ignore setup.py --version 2>/dev/null || echo "0.1.0")
NUM_EXAMPLES ?= 3

# Docker configuration
DOCKER_IMAGE := parcel-ai-json
DOCKER_TAG := latest
DOCKER_REGISTRY := # Set to your registry (e.g., account.dkr.ecr.us-west-2.amazonaws.com)
DOCKER_PORT := 8000

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
	@source $(VENV_BIN)/activate && $(VENV_BIN)/flake8

format: ## Format code with black
	@source $(VENV_BIN)/activate && $(VENV_BIN)/black parcel_ai_json/ tests/ examples/ scripts/

format-check: ## Check code formatting without making changes
	@source $(VENV_BIN)/activate && $(VENV_BIN)/black --check parcel_ai_json/ tests/ examples/ scripts/

check: format-check lint test ## Run all checks (format, lint, test)

clean: ## Clean up generated files, build artifacts, and cache
	@echo "Cleaning up build artifacts and cache files..."
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".DS_Store" -delete 2>/dev/null || true
	find . -type d -name ".idea" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleanup complete!"

clean-all: clean ## Clean everything including virtualenv
	rm -rf $(VENV)

generate-examples: ## Generate detection examples with labeled SAM segmentation (default: 3 examples)
	@source $(VENV_BIN)/activate && PYTHONPATH=$(PYTHONPATH) $(PYTHON) -u scripts/generate_examples.py --num-examples $(NUM_EXAMPLES) 2>&1 | tee /tmp/generate_examples.log

generate-examples-10: ## Generate 10 detection examples with labeled SAM segmentation
	@$(MAKE) generate-examples NUM_EXAMPLES=10

generate-examples-20: ## Generate 20 detection examples with labeled SAM segmentation
	@$(MAKE) generate-examples NUM_EXAMPLES=20

dev-setup: install ## Set up development environment
	@echo "Development environment ready!"
	@echo "Run 'source $(VENV)/bin/activate' to activate the virtualenv"
	@echo "Run 'make test' to run tests"
	@echo "Run 'make coverage-html' to see coverage report"

# Package build and deployment targets

install-ci: ## Install deployment dependencies (twine, wheel)
	$(PIP) install -e ".[deploy]"
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

# Docker targets (RECOMMENDED DEPLOYMENT METHOD)

docker-build: ## Build Docker image
	@echo "Building Docker image $(DOCKER_IMAGE):$(DOCKER_TAG)..."
	docker build -f docker/Dockerfile -t $(DOCKER_IMAGE):$(DOCKER_TAG) .
	@echo "Docker image built successfully!"
	@echo "Run 'make docker-run' to start the service"

docker-build-clean: ## Build Docker image without cache (forces full rebuild)
	@echo "Building Docker image $(DOCKER_IMAGE):$(DOCKER_TAG) without cache..."
	docker build --no-cache -f docker/Dockerfile -t $(DOCKER_IMAGE):$(DOCKER_TAG) .
	@echo "Docker image built successfully!"
	@echo "Run 'make docker-run' to start the service"

docker-run: ## Run Docker container locally
	@echo "Starting $(DOCKER_IMAGE) on port $(DOCKER_PORT)..."
	docker run -d \
		--name $(DOCKER_IMAGE) \
		-p $(DOCKER_PORT):8000 \
		$(DOCKER_IMAGE):$(DOCKER_TAG)
	@echo ""
	@echo "Service started!"
	@echo "  API:      http://localhost:$(DOCKER_PORT)"
	@echo "  Docs:     http://localhost:$(DOCKER_PORT)/docs"
	@echo "  Health:   http://localhost:$(DOCKER_PORT)/health"
	@echo ""
	@echo "View logs:  make docker-logs"
	@echo "Stop:       make docker-stop"

docker-stop: ## Stop and remove Docker container
	@echo "Stopping $(DOCKER_IMAGE)..."
	docker stop $(DOCKER_IMAGE) 2>/dev/null || true
	docker rm $(DOCKER_IMAGE) 2>/dev/null || true
	@echo "Container stopped and removed"

docker-logs: ## Show Docker container logs
	docker logs -f $(DOCKER_IMAGE)

docker-shell: ## Open shell in running Docker container
	docker exec -it $(DOCKER_IMAGE) /bin/bash

docker-push: ## Push Docker image to registry (requires DOCKER_REGISTRY to be set)
	@if [ -z "$(DOCKER_REGISTRY)" ]; then \
		echo "Error: DOCKER_REGISTRY not set. Set it in Makefile or use:"; \
		echo "  make docker-push DOCKER_REGISTRY=your-registry.com"; \
		exit 1; \
	fi
	@echo "Tagging image for registry..."
	docker tag $(DOCKER_IMAGE):$(DOCKER_TAG) $(DOCKER_REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG)
	docker tag $(DOCKER_IMAGE):$(DOCKER_TAG) $(DOCKER_REGISTRY)/$(DOCKER_IMAGE):v$(PROJECT_VERSION)
	@echo "Pushing to registry..."
	docker push $(DOCKER_REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG)
	docker push $(DOCKER_REGISTRY)/$(DOCKER_IMAGE):v$(PROJECT_VERSION)
	@echo "Image pushed successfully!"

docker-clean: ## Remove Docker image and clean build cache
	docker rmi $(DOCKER_IMAGE):$(DOCKER_TAG) 2>/dev/null || true
	docker image prune -f
	@echo "Docker images cleaned"

docker-up: ## Start services with Docker Compose
	@echo "Starting services with Docker Compose..."
	docker-compose -f docker/docker-compose.yml up -d
	@echo ""
	@echo "Services started!"
	@echo "  API:      http://localhost:8000"
	@echo "  Docs:     http://localhost:8000/docs"
	@echo ""
	@echo "View logs:  docker-compose -f docker/docker-compose.yml logs -f"
	@echo "Stop:       make docker-down"

docker-down: ## Stop Docker Compose services
	@echo "Stopping Docker Compose services..."
	docker-compose -f docker/docker-compose.yml down
	@echo "Services stopped"

docker-restart: docker-stop docker-run ## Restart Docker container

docker-rebuild: docker-stop docker-build docker-run ## Rebuild and restart Docker container
