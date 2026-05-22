import argparse
import csv
import json
import math
import os
from typing import Any, Dict, List, Optional

DEFAULT_METRICS = [
    "avg_episode_time_sec",
    "total_training_time_sec",
    "time_to_convergence_sec",
    "episodes_to_convergence",
    "fitness",
]


def find_run_summaries(runs_dir: str) -> List[str]:
    summaries = []
    for root, _, files in os.walk(runs_dir):
        if "run_summary.json" in files:
            summaries.append(os.path.join(root, "run_summary.json"))
    return summaries


def load_summary(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def safe_std(values: List[float]) -> Optional[float]:
    if len(values) < 2:
        return 0.0 if values else None
    mean_val = safe_mean(values)
    if mean_val is None:
        return None
    variance = sum((v - mean_val) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def ci95(mean_val: float, std_val: float, n: int) -> (float, float):
    if n <= 1:
        return mean_val, mean_val
    z = 1.96
    half_width = z * (std_val / math.sqrt(n))
    return mean_val - half_width, mean_val + half_width


def build_rows(
    summaries: List[Dict[str, Any]],
    metrics: List[str],
) -> List[Dict[str, Any]]:
    per_method: Dict[str, Dict[str, List[float]]] = {}
    for summary in summaries:
        method = summary.get("method", "unknown")
        metrics_data = summary.get("metrics", {})
        per_method.setdefault(method, {})
        for metric in metrics:
            value = metrics_data.get(metric)
            if value is None:
                continue
            per_method[method].setdefault(metric, []).append(float(value))

    rows: List[Dict[str, Any]] = []
    for method, metric_map in sorted(per_method.items()):
        for metric in metrics:
            values = metric_map.get(metric, [])
            n = len(values)
            if n == 0:
                continue
            mean_val = safe_mean(values)
            std_val = safe_std(values)
            if mean_val is None or std_val is None:
                continue
            ci_low, ci_high = ci95(mean_val, std_val, n)
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "n": n,
                    "mean": mean_val,
                    "std": std_val,
                    "min": min(values),
                    "max": max(values),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                }
            )
    return rows


def print_table(rows: List[Dict[str, Any]]) -> None:
    headers = [
        "method",
        "metric",
        "n",
        "mean",
        "std",
        "min",
        "max",
        "ci95_low",
        "ci95_high",
    ]
    col_widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            col_widths[h] = max(col_widths[h], len(str(row[h])))

    header_line = "  ".join(h.ljust(col_widths[h]) for h in headers)
    print(header_line)
    print("  ".join("-" * col_widths[h] for h in headers))
    for row in rows:
        line = "  ".join(str(row[h]).ljust(col_widths[h]) for h in headers)
        print(line)


def write_csv(rows: List[Dict[str, Any]], output_path: str) -> None:
    if not rows:
        return
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(rows: List[Dict[str, Any]], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute statistical summaries from run logs.")
    parser.add_argument("--runs-dir", default="runs", help="Root directory with run outputs.")
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated metric names to summarize.",
    )
    parser.add_argument(
        "--format",
        default="console",
        choices=["console", "csv", "json"],
        help="Output format.",
    )
    parser.add_argument("--output", default="", help="Optional output file path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    summaries = [load_summary(p) for p in find_run_summaries(args.runs_dir)]
    rows = build_rows(summaries, metrics)

    if args.format == "console":
        if not rows:
            print("No metrics found. Check the runs directory or metrics list.")
            return
        print_table(rows)
        if args.output:
            write_csv(rows, args.output)
        return

    if args.format == "csv":
        if args.output:
            write_csv(rows, args.output)
        else:
            writer = csv.DictWriter(
                os.sys.stdout,
                fieldnames=list(rows[0].keys()) if rows else [],
            )
            if rows:
                writer.writeheader()
                writer.writerows(rows)
        return

    if args.format == "json":
        if args.output:
            write_json(rows, args.output)
        else:
            print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
