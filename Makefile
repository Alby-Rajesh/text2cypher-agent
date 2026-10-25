.PHONY: dev lint test examples eval serve up

dev:
	pip install -e ".[dev,data]"

lint:
	ruff check src tests finetune
	ruff format --check src tests finetune

test:
	python -m pytest -q

examples:
	text2cypher examples

eval:
	text2cypher evaluate --limit 200

serve:
	text2cypher serve

up:
	docker compose up --build
