# AGENTS.md — commands for AI agents working in this repo

## Environment
- Python 3.11+ managed by `uv`.
- Install deps: `uv sync --extra dev`
- Run commands via: `uv run <cmd>`

## Lint / Typecheck / Test (run all before finishing work)
- Lint: `uv run ruff check .`
- Typecheck: `uv run mypy src`
- Tests: `uv run pytest`

## Run the pipeline
- Single file: `uv run tdv-vectorize <input.png|jpg|pdf> -o data/results/runs/<name>`
- Batch: `uv run tdv-vectorize data/fixtures/synthetic -o data/results/runs/batch`
- Evaluate: `uv run tdv-evaluate data/fixtures -o data/results/runs/eval`
- Regenerate fixtures: `uv run tdv-make-fixtures -o data/fixtures/synthetic`

## Determinism note
The pipeline MUST produce byte-identical JSON+SVG for identical input+config.
Do not introduce nondeterministic ordering, RNG without a fixed seed, or unsorted outputs.
