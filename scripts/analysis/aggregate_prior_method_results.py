#!/usr/bin/env python3
"""Aggregate prior-method evaluation metrics into long and macro CSV files."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


LONG_FIELDS = (
    "method",
    "accent",
    "decoder",
    "dataset",
    "wer_percent",
    "errors",
    "substitutions",
    "deletions",
    "insertions",
    "reference_words",
    "utterances",
    "checkpoint_sha256",
)
MACRO_FIELDS = (
    "method",
    "decoder",
    "dataset",
    "folds",
    "accents",
    "macro_wer_percent",
)


def parse_identity(path: Path, base: Path) -> tuple[str, str, str, str]:
    relative = path.relative_to(base)
    parts = relative.parts
    if len(parts) != 8 or parts[2:5] != ("full-transformer", "outputs", "seed=13"):
        raise ValueError(f"Unexpected metrics path: {path}")
    method, accent = parts[:2]
    decoder, dataset, filename = parts[5:]
    if filename != "metrics.json":
        raise ValueError(f"Unexpected metrics filename: {path}")
    return method, accent, decoder, dataset


def collect_rows(base: Path) -> list[dict]:
    rows = []
    pattern = "*/*/full-transformer/outputs/seed=13/*/*/metrics.json"
    for path in sorted(base.glob(pattern)):
        method, accent, decoder, dataset = parse_identity(path, base)
        metrics = json.loads(path.read_text(encoding="utf-8"))
        if metrics.get("smoke"):
            continue
        rows.append(
            {
                "method": method,
                "accent": accent,
                "decoder": decoder,
                "dataset": dataset,
                "wer_percent": 100 * float(metrics["wer"]),
                "errors": int(metrics["errors"]),
                "substitutions": int(metrics["substitutions"]),
                "deletions": int(metrics["deletions"]),
                "insertions": int(metrics["insertions"]),
                "reference_words": int(metrics["reference_words"]),
                "utterances": int(metrics["utterances"]),
                "checkpoint_sha256": metrics["checkpoint_sha256"],
            }
        )
    return rows


def macro_rows(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["decoder"], row["dataset"])].append(row)

    output = []
    for (method, decoder, dataset), records in sorted(groups.items()):
        records.sort(key=lambda record: record["accent"])
        accents = [record["accent"] for record in records]
        if len(accents) != len(set(accents)):
            raise ValueError(f"Duplicate accent in {method}/{decoder}/{dataset}")
        output.append(
            {
                "method": method,
                "decoder": decoder,
                "dataset": dataset,
                "folds": len(records),
                "accents": "|".join(accents),
                "macro_wer_percent": sum(
                    record["wer_percent"] for record in records
                )
                / len(records),
            }
        )
    return output


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = collect_rows(args.base)
    if not rows:
        raise SystemExit(f"No evaluation metrics found below {args.base}")
    rows.sort(key=lambda row: tuple(row[field] for field in LONG_FIELDS[:4]))
    write_csv(args.output_dir / "prior-method-results-long.csv", LONG_FIELDS, rows)
    write_csv(
        args.output_dir / "prior-method-results-macro.csv",
        MACRO_FIELDS,
        macro_rows(rows),
    )
    print(f"Wrote {len(rows)} evaluation rows to {args.output_dir}")


if __name__ == "__main__":
    main()
