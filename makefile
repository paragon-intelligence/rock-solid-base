.PHONY: release

release:
	uv run release.py

pytest:
	uv run pytest tests/unit/
