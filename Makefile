.PHONY: format
format:
	poetry run ruff check  . --fix
