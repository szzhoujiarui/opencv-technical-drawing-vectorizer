from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def ensure_fixtures():
    fixtures_dir = Path("data/fixtures/synthetic")
    manifest = fixtures_dir / "_manifest.json"
    if not manifest.exists():
        from tdv.fixtures.synth import generate_all

        generate_all(fixtures_dir)
