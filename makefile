.PHONY: setup-dev train test run

export PYTHONPATH := $(PWD)/src:$(PYTHONPATH)

format:
	@poetry run ruff check --select I --fix . && ruff format .

train:
	@poetry run python scripts/train_model.py

setup-dev:
	@poetry run python scripts/setup_dev.py

test:
	pytest tests/

run:
	uvicorn dr_llama.main:app --reload --host 0.0.0.0 --port 8000
