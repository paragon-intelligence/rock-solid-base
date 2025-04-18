.PHONY: release clean pytest

release:
	uv run release.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	echo "All __pycache__ folders removed."

pytest:
	uv run pytest tests/unit/
