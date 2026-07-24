# Contributing

## Setup

```shell
# Install dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Lint
uv run ruff check .

# Type check
uv run mypy src
```

## Workflow

1. Create a branch from `main`
2. Make changes and ensure all checks pass
3. Commit with descriptive messages
4. Push and open a merge request

## Code Style

- Line length: 100 characters
- Python 3.11+ syntax
- Type annotations required (mypy strict mode)
- All public functions need type hints
