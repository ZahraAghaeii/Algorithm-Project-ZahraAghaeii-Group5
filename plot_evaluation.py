from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def read_results(csv_path: Path) -> List[dict]:
    rows: List[dict] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def pivot(rows: List[dict], metric: str) -> Tuple[List[str], List[str], Dict[str, Dict[str, float]]]:
    """
    Returns:
      text_ids: sorted list of text ids
      methods: sorted list of methods (keeps preferred order if present)
      data[text_id][method] -> metric value
    """
    preferred = ["textrank", "llm", "hybrid"]

    text_ids = sorted({r["text_id"] for r in rows})
    methods_found = sorted({r["method"] for r in rows})
    methods = [m for m in preferred if m in methods_found] + [m for m in methods_found if m not in preferred]

    data: Dict[str, Dict[str, float]] = defaultdict(dict)
    for r in rows:
        t = r["text_id"]
        m = r["method"]
        val = float(r[metric])
        data[t][m] = val

    return text_ids, methods, data


def plot_grouped_bars(
    rows: List[dict],
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    text_ids, methods, data = pivot(rows, metric)

    # Grouped bar positions
    group_count = len(text_ids)
    method_count = len(methods)
    bar_width = 0.25 if method_count <= 3 else 0.18

    x_positions = list(range(group_count))

    plt.figure()
    for j, method in enumerate(methods):
        vals = [data[t].get(method, 0.0) for t in text_ids]
        # Shift each method bar in the group
        xs = [x + (j - (method_count - 1) / 2) * bar_width for x in x_positions]
        plt.bar(xs, vals, width=bar_width, label=method)

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Input text")
    plt.xticks(x_positions, text_ids)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    csv_path = Path("evaluation_results.csv")
    if not csv_path.exists():
        raise FileNotFoundError("evaluation_results.csv not found. Run run_evaluation.py first.")

    rows = read_results(csv_path)

    out_dir = Path("plots")
    out_dir.mkdir(exist_ok=True)

    # Runtime chart
    plot_grouped_bars(
        rows=rows,
        metric="runtime_sec",
        title="Runtime Comparison (seconds)",
        ylabel="Seconds",
        output_path=out_dir / "runtime_sec.png",
    )

    # Coverage chart
    plot_grouped_bars(
        rows=rows,
        metric="coverage",
        title="Token Coverage Comparison",
        ylabel="Coverage (0-1)",
        output_path=out_dir / "coverage.png",
    )

    # Words chart
    plot_grouped_bars(
        rows=rows,
        metric="words",
        title="Summary Length Comparison (words)",
        ylabel="Words",
        output_path=out_dir / "words.png",
    )

    # Redundancy chart
    plot_grouped_bars(
        rows=rows,
        metric="redundancy",
        title="Redundancy Comparison (avg cosine similarity)",
        ylabel="Redundancy",
        output_path=out_dir / "redundancy.png",
    )

    print("Saved plots to:", out_dir.resolve())


if __name__ == "__main__":
    main()
