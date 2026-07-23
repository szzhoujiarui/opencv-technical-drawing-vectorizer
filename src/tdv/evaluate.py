from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from tdv.cli import vectorize
from tdv.config import PipelineConfig
from tdv.io.save import save_json
from tdv.report.metrics import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate vectorizer against fixtures")
    parser.add_argument(
        "fixtures_dir",
        type=Path,
        help="Fixtures directory with _manifest.json",
    )
    parser.add_argument("-c", "--config", type=Path, default=None, help="Config YAML path")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output directory")
    args = parser.parse_args()

    manifest_path = args.fixtures_dir / "_manifest.json"
    if not manifest_path.exists():
        print(f"No manifest found at {manifest_path}")
        return

    with open(manifest_path) as f:
        fixtures = json.load(f)

    config = (
        PipelineConfig.from_yaml(Path(args.config)) if args.config else PipelineConfig.default()
    )
    out_dir = Path(args.output) if args.output else Path("data/results/runs/eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict[str, Any]] = []

    for fixture in fixtures:
        fixture_id = fixture["id"]
        image_path = Path(fixture["image"])
        gt_path = Path(fixture["ground_truth"])

        print(f"  Evaluating: {fixture_id}")

        if not gt_path.exists():
            print(f"    No ground truth at {gt_path}, skipping metrics")
            continue

        with open(gt_path) as f:
            gt = json.load(f)

        t0 = time.time()
        try:
            result = vectorize(image_path, args.config, out_dir / fixture_id)
            elapsed = time.time() - t0
        except Exception as e:
            print(f"    FAILED: {e}")
            continue

        eval_result = evaluate(
            result["primitives"],
            gt,
            config.metrics,
        )

        fixture_result = {
            "id": fixture_id,
            "description": fixture.get("description", ""),
            "elapsed_s": round(elapsed, 3),
            "metrics": eval_result,
        }
        all_results.append(fixture_result)

        print(f"    {elapsed:.2f}s | {eval_result}")

    report = {
        "config": config.model_dump(mode="json"),
        "fixtures": all_results,
    }

    report_path = out_dir / "eval_report.json"
    save_json(report_path, report, config.precision)

    md_path = out_dir / "eval_report.md"
    with open(md_path, "w") as f:
        f.write("# Evaluation Report\n\n")
        for fr in all_results:
            f.write(f"## {fr['id']}: {fr.get('description', '')}\n")
            f.write(f"- Time: {fr['elapsed_s']:.2f}s\n")
            for prim_type, prim_metrics in fr.get("metrics", {}).items():
                f.write(f"- {prim_type}:\n")
                for k, v in prim_metrics.items():
                    f.write(f"  - {k}: {v}\n")
            f.write("\n")

    print(f"Report written to {report_path}")
    print(f"Markdown report at {md_path}")
