# Convenience targets. Everything here is a thin wrapper around a command that
# also works on its own; nothing is hidden behind make.

.DEFAULT_GOAL := help
.PHONY: help install test evals evals-full models report clean-report up down logs health

help: ## Show this list
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[1m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Sync dependencies from uv.lock
	uv sync

test: ## Run the pytest suite (no network, no models)
	uv run pytest -q

evals: ## Run the offline eval cases (local models only)
	uv run python evals/run_evals.py --offline

evals-full: ## Run every eval case (needs OpenRouter quota)
	uv run python evals/run_evals.py

models: ## Compare candidate LLMs for the router role (local, free)
	uv run python evals/compare_models.py --router-only

report: ## Build report/report.pdf from docs/ (needs Docker)
	uv run python scripts/build_report.py

clean-report: ## Remove the report build directory, including the diagram cache
	rm -rf report/build

up: ## Start the full stack (app, ollama, observability)
	docker compose up -d --build

down: ## Stop the stack, keeping volumes
	docker compose down

logs: ## Follow the app's logs
	docker compose logs -f app

health: ## Print the app's health endpoint
	curl -s http://127.0.0.1:8000/health | python3 -m json.tool
