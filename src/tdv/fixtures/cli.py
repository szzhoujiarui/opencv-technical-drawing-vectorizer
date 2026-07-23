import argparse
from pathlib import Path

from tdv.fixtures.synth import generate_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic fixture drawings")
    parser.add_argument("-o", "--output-dir", type=Path, default="data/fixtures/synthetic")
    args = parser.parse_args()
    manifests = generate_all(args.output_dir)
    print(f"Generated {len(manifests)} synthetic fixtures in {args.output_dir}")
    for m in manifests:
        print(f"  {m['id']}: {m['description']}")
