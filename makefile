format:
	@poetry run ruff check --select I --fix . && ruff format .
