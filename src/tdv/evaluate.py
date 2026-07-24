from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

from tdv.cli import vectorize
from tdv.config import PipelineConfig
from tdv.io.save import save_json
from tdv.report.metrics import evaluate

logger = logging.getLogger("tdv")


def _macro_average_f1(results: list[dict[str, Any]]) -> float:
    per_fixture_f1: list[float] = []
    for r in results:
        f1_scores = [
            m["f1"] for m in r.get("metrics", {}).values() if "f1" in m
        ]
        if f1_scores:
            per_fixture_f1.append(sum(f1_scores) / len(f1_scores))
    return sum(per_fixture_f1) / len(per_fixture_f1) if per_fixture_f1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate vectorizer against fixtures")
    parser.add_argument(
        "fixtures_dir",
        type=Path,
        help="Fixtures directory with _manifest.json",
    )
    parser.add_argument("-c", "--config", type=Path, default=None, help="Config YAML path")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    manifest_path = args.fixtures_dir / "_manifest.json"
    if not manifest_path.exists():
        logger.error("No manifest found at %s", manifest_path)
        return

    with open(manifest_path) as f:
        fixtures = json.load(f)

    config = (
        PipelineConfig.from_yaml(Path(args.config)) if args.config else PipelineConfig.default()
    )
    out_dir = Path(args.output) if args.output else Path("data/results/runs/eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict[str, Any]] = []
    base = args.fixtures_dir.resolve()

    for fixture in fixtures:
        fixture_id = fixture["id"]
        image_path = (base / fixture["image"]).resolve()
        gt_path = (base / fixture["ground_truth"]).resolve()

        if not gt_path.is_relative_to(base):
            logger.warning("    Path escapes fixtures dir: %s", gt_path)
            continue
        if not image_path.is_relative_to(base):
            logger.warning("    Path escapes fixtures dir: %s", image_path)
            continue

        logger.info("  Evaluating: %s", fixture_id)

        if not gt_path.exists():
            logger.warning("    No ground truth at %s, skipping metrics", gt_path)
            continue

        with open(gt_path) as f:
            gt = json.load(f)

        t0 = time.time()
        try:
            result = vectorize(image_path, args.config, out_dir / fixture_id)
            elapsed = time.time() - t0
        except Exception as e:
            logger.error("    FAILED: %s", e)
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

        logger.info("    %.2fs | %s", elapsed, eval_result)

    report: dict[str, Any] = {
        "config": config.model_dump(mode="json"),
        "fixtures": all_results,
    }

    if all_results:
        macro_f1 = _macro_average_f1(all_results)
        report["macro_avg_f1"] = macro_f1

    report_path = out_dir / "eval_report.json"
    save_json(report_path, report, config.precision)

    md_path = out_dir / "eval_report.md"
    with open(md_path, "w") as f:
        f.write("# Evaluation Report\n\n")
        if all_results:
            macro_f1 = _macro_average_f1(all_results)
            f.write(f"## Macro Average F1: {macro_f1:.4f}\n\n")
        for fr in all_results:
            f.write(f"## {fr['id']}: {fr.get('description', '')}\n")
            f.write(f"- Time: {fr['elapsed_s']:.2f}s\n")
            for prim_type, prim_metrics in fr.get("metrics", {}).items():
                f.write(f"- {prim_type}:\n")
                for k, v in prim_metrics.items():
                    f.write(f"  - {k}: {v}\n")
            f.write("\n")

    logger.info("Report written to %s", report_path)
    logger.info("Markdown report at %s", md_path)
