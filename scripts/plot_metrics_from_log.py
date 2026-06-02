#!/usr/bin/env python3
"""Plot verl console scalar metrics from a training log.

The verl console logger prints records like:
  step:1 - actor/entropy:0.25 - actor/lr:np.float64(1e-6)

This script extracts those scalar records, writes a CSV, and saves one PNG per
selected metric.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path


PAIR_RE = re.compile(
    r"(?P<key>[A-Za-z0-9_./-]+):"
    r"(?P<value>np\.float64\([^)]+\)|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|nan|inf|-inf)"
)


def parse_value(raw: str) -> float | None:
    if raw.startswith("np.float64(") and raw.endswith(")"):
        raw = raw[len("np.float64(") : -1]
    try:
        value = float(raw)
    except ValueError:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def parse_logs(paths: list[Path]) -> tuple[list[dict[str, float]], set[str]]:
    rows: list[dict[str, float]] = []
    keys: set[str] = set()

    for path in paths:
        with path.open("r", errors="ignore") as f:
            for line in f:
                if "step:" not in line:
                    continue
                pairs = {}
                for match in PAIR_RE.finditer(line):
                    key = match.group("key")
                    value = parse_value(match.group("value"))
                    if value is None:
                        continue
                    pairs[key] = value
                if "step" not in pairs:
                    continue
                rows.append(pairs)
                keys.update(pairs)

    rows.sort(key=lambda row: row.get("step", row.get("training/global_step", 0.0)))
    return rows, keys


def metric_selected(metric: str, includes: list[str], excludes: list[str]) -> bool:
    if metric == "step":
        return False
    if includes and not any(re.search(pattern, metric) for pattern in includes):
        return False
    if excludes and any(re.search(pattern, metric) for pattern in excludes):
        return False
    return True


def write_csv(rows: list[dict[str, float]], metrics: list[str], output_dir: Path) -> Path:
    csv_path = output_dir / "metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", *metrics])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in ["step", *metrics]})
    return csv_path


def safe_name(metric: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", metric.replace("/", "__"))


def plot_metrics(rows: list[dict[str, float]], metrics: list[str], output_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit("matplotlib is not installed. Run: pip install matplotlib") from exc

    by_metric: dict[str, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
    for row in rows:
        step = row.get("step", row.get("training/global_step"))
        if step is None:
            continue
        for metric in metrics:
            if metric not in row:
                continue
            xs, ys = by_metric[metric]
            xs.append(step)
            ys.append(row[metric])

    for metric, (xs, ys) in by_metric.items():
        if not xs:
            continue
        plt.figure(figsize=(8, 4.5))
        plt.plot(xs, ys, linewidth=1.6)
        plt.xlabel("step")
        plt.ylabel(metric)
        plt.title(metric)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"{safe_name(metric)}.png", dpi=180)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path, help="Training log files, for example slurm-123.out")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("figs/metrics"))
    parser.add_argument(
        "-m",
        "--metric",
        action="append",
        default=[],
        help="Regex for metrics to include. Can be passed multiple times. Default: all scalar metrics.",
    )
    parser.add_argument(
        "-x",
        "--exclude",
        action="append",
        default=[],
        help="Regex for metrics to exclude. Can be passed multiple times.",
    )
    args = parser.parse_args()

    missing = [str(path) for path in args.logs if not path.exists()]
    if missing:
        raise SystemExit(f"Missing log file(s): {', '.join(missing)}")

    rows, keys = parse_logs(args.logs)
    if not rows:
        raise SystemExit("No console scalar rows found. Expected lines containing 'step:... - metric:value'.")

    metrics = sorted(key for key in keys if metric_selected(key, args.metric, args.exclude))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = write_csv(rows, metrics, args.output_dir)
    plot_metrics(rows, metrics, args.output_dir)

    print(f"Parsed {len(rows)} scalar rows")
    print(f"Wrote {csv_path}")
    print(f"Wrote PNG files to {args.output_dir}")


if __name__ == "__main__":
    main()
