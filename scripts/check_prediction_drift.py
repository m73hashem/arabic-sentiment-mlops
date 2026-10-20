"""Compare confidence-score CSVs using PSI and write a text-free evidence report."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from arabic_sentiment.monitoring import (
    compare_confidence_files,
)


def write_dashboard(report: dict[str, object], path: Path) -> None:
    status = "ALERT" if report["alert"] else "No alert"
    body = (
        "<!doctype html><meta charset='utf-8'><title>Local prediction drift</title>"
        "<h1>Prediction confidence drift</h1>"
        f"<p>Status: <strong>{html.escape(status)}</strong></p>"
        f"<p>PSI: {report['psi']:.6f}; alert threshold: > {report['threshold']}</p>"
        f"<p>Reference: {html.escape(str(report['reference']['file']))} "
        f"({report['reference']['rows']} rows)</p>"
        f"<p>Current: {html.escape(str(report['current']['file']))} "
        f"({report['current']['rows']} rows)</p>"
        "<p>Text-free local summary; no review text is stored in this report.</p>"
    )
    path.write_text(body, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="JSON evidence report path"
    )
    args = parser.parse_args()
    report = compare_confidence_files(args.reference, args.current)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_dashboard(report, args.output.with_suffix(".html"))
    print(json.dumps(report, indent=2))
    print(f"Dashboard: {args.output.with_suffix('.html')}")
    if report["alert"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
