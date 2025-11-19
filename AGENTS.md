# Repository Guidelines

## Project Structure & Module Organization
Core FastAPI services and detectors live in `parcel_ai_json/`, with `api.py` exposing HTTP routes and modules such as `vehicle_detector.py`, `grounded_sam_detector.py`, and `property_detector.py` handling inference logic. Tests mirror this layout under `tests/` (e.g., `tests/test_api.py`) and rely on fixtures for device mocks. Reference datasets, prompts, and walkthroughs are under `examples/`, while automation helpers sit in `scripts/` such as `scripts/generate_examples.py`. Container assets stay in `docker/`, generated assets in `output/` and `htmlcov/`, and `docs/` stores architecture notes. Treat `models/` as large binary storage and avoid committing new weights unless coordinated.

## Build, Test, and Development Commands
Run `make install` (or `make dev-setup`) once to create `venv/` and install editable dependencies. Activate the virtualenv (`source venv/bin/activate`) before local work. `make test` executes the Pytest suite with coverage, while `make coverage-html` produces `htmlcov/index.html` for interactive review. Use `make lint` for `flake8` and `make format` for `black` auto-formatting. To exercise the service end-to-end, `make docker-build && make docker-run` launches the API at `http://localhost:8000`; verify with `curl http://localhost:8000/health` or open `/docs`.

## Coding Style & Naming Conventions
This repo targets Python 3.10+. Follow PEP 8 with `black`’s defaults (88-char lines, 4-space indentation) and keep imports sorted logically (stdlib, third-party, local). Use snake_case for functions, PascalCase for classes, and keep module names descriptive (`*_detector.py`). Prefer type hints and FastAPI `pydantic` models for request/response schemas, and log via the shared `logging` module instead of print statements. When adding CLI helpers or scripts, route them through the Makefile to keep developer workflows consistent.

## Testing Guidelines
All tests run through Pytest; place new suites under `tests/` and name files `test_<feature>.py` with functions like `test_endpoint_returns_geojson`. Mock heavyweight detectors via FastAPI dependency overrides or fixture factories to keep tests GPU-agnostic. Maintain the current ~87 % coverage badge by adding assertions for edge cases (invalid coordinates, unsupported MIME types, etc.) and regenerating reports with `make coverage` before opening a PR. Document any new fixtures or sample assets inside the `tests/fixtures` hierarchy.

## Commit & Pull Request Guidelines
Git history favors short, imperative subjects (“Add Grounded-SAM API endpoint”) followed by optional detail in the body; keep subjects under ~72 characters and mention affected modules when relevant. Each PR should describe the change, link to the tracking issue or task ID, spell out validation steps (`make test`, `make docker-run` smoke), and attach screenshots or sample GeoJSON snippets for user-facing updates. Request review once CI is green and note any follow-ups or TODOs explicitly rather than leaving silent gaps.
