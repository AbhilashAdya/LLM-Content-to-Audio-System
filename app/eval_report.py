"""Generate graphs for deduplication quality evaluation.

Outputs standalone SVG files (no HTML dashboard, no extra plotting dependencies).

Must-have graphs:
- Raw vs Distinct
- Cluster size distribution
- Noise ratio (clustered vs noise)
- eps sweep stability curves

Output layout:
  {out_dir}/{day_key}/
    raw_vs_distinct.svg
    cluster_size_distribution.svg
    noise_ratio.svg
    eps_sweep_distinct_count.svg        (if enabled)
    eps_sweep_noise_rate.svg            (if enabled)
    eps_sweep_max_cluster_size.svg      (if enabled)
    eps_sweep.csv                       (if enabled)
"""

from __future__ import annotations

import argparse
import csv
import html
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _ensure_project_root_on_syspath() -> None:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


_ensure_project_root_on_syspath()

from app.eval import (  # noqa: E402
    _get_day_data,
    _missing_dependency_message,
    compute_cluster_size_stats,
    compute_compression_rate,
    compute_noise_rate,
    compute_within_cluster_similarity,
)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text, encoding="utf-8")


def _svg_bar_chart(
    *,
    title: str,
    labels: List[str],
    values: List[float],
    width: int = 520,
    height: int = 240,
    bar_color: str = "#4C78A8",
) -> str:
    if not labels or not values or len(labels) != len(values):
        return ""

    margin_l, margin_r, margin_t, margin_b = 48, 16, 28, 42
    plot_w = max(1, width - margin_l - margin_r)
    plot_h = max(1, height - margin_t - margin_b)

    vmax = max(values) if max(values) > 0 else 1.0
    n = len(values)
    gap = 12
    bar_w = max(8, int((plot_w - gap * (n - 1)) / max(1, n)))

    def x0(i: int) -> int:
        return margin_l + i * (bar_w + gap)

    def y(v: float) -> int:
        return margin_t + int((1 - (v / vmax)) * plot_h)

    parts: List[str] = []
    parts.append(f'<svg viewBox="0 0 {width} {height}" role="img">')
    parts.append(
        f'<text x="{margin_l}" y="{margin_t - 10}" font-size="14" font-weight="600">{html.escape(title)}</text>'
    )

    parts.append(
        f'<line x1="{margin_l}" y1="{margin_t + plot_h}" x2="{margin_l + plot_w}" y2="{margin_t + plot_h}" stroke="#999"/>'
    )
    parts.append(
        f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t + plot_h}" stroke="#999"/>'
    )

    for i, (lab, v) in enumerate(zip(labels, values)):
        left = x0(i)
        top = y(v)
        h = (margin_t + plot_h) - top
        parts.append(
            f'<rect x="{left}" y="{top}" width="{bar_w}" height="{h}" fill="{bar_color}" rx="3"/>'
        )
        parts.append(
            f'<text x="{left + bar_w / 2}" y="{top - 6}" text-anchor="middle" font-size="12" fill="#333">{int(v)}</text>'
        )
        parts.append(
            f'<text x="{left + bar_w / 2}" y="{margin_t + plot_h + 18}" text-anchor="middle" font-size="12" fill="#333">{html.escape(lab)}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


def _svg_stacked_bar(
    *,
    title: str,
    a_label: str,
    a_value: int,
    b_label: str,
    b_value: int,
    width: int = 520,
    height: int = 160,
    a_color: str = "#54A24B",
    b_color: str = "#E45756",
) -> str:
    total = max(1, a_value + b_value)
    a_w = int((a_value / total) * (width - 64))
    b_w = int((b_value / total) * (width - 64))

    parts: List[str] = []
    parts.append(f'<svg viewBox="0 0 {width} {height}" role="img">')
    parts.append(
        f'<text x="32" y="20" font-size="14" font-weight="600">{html.escape(title)}</text>'
    )

    x = 32
    y = 46
    h = 26
    parts.append(
        f'<rect x="{x}" y="{y}" width="{a_w}" height="{h}" fill="{a_color}" rx="3"/>'
    )
    parts.append(
        f'<rect x="{x + a_w}" y="{y}" width="{b_w}" height="{h}" fill="{b_color}" rx="3"/>'
    )

    a_pct = 100.0 * (a_value / total)
    b_pct = 100.0 * (b_value / total)

    parts.append(
        f'<text x="{x}" y="{y + 56}" font-size="12" fill="#333">{html.escape(a_label)}: {a_value} ({a_pct:.1f}%)</text>'
    )
    parts.append(
        f'<text x="{x}" y="{y + 76}" font-size="12" fill="#333">{html.escape(b_label)}: {b_value} ({b_pct:.1f}%)</text>'
    )

    parts.append("</svg>")
    return "".join(parts)


def _bucket_cluster_sizes(sizes: List[int]) -> Tuple[List[str], List[int]]:
    buckets = [
        ("1", lambda s: s == 1),
        ("2", lambda s: s == 2),
        ("3", lambda s: s == 3),
        ("4", lambda s: s == 4),
        ("5", lambda s: s == 5),
        ("6-10", lambda s: 6 <= s <= 10),
        ("11-20", lambda s: 11 <= s <= 20),
        ("21+", lambda s: s >= 21),
    ]

    labels: List[str] = []
    counts: List[int] = []
    for name, pred in buckets:
        labels.append(name)
        counts.append(sum(1 for s in sizes if pred(s)))
    return labels, counts


def _svg_line_chart(
    *,
    title: str,
    xs: List[float],
    ys: List[float],
    y_label: str,
    width: int = 520,
    height: int = 240,
    color: str = "#4C78A8",
) -> str:
    if not xs or not ys or len(xs) != len(ys):
        return ""

    margin_l, margin_r, margin_t, margin_b = 48, 16, 28, 42
    plot_w = max(1, width - margin_l - margin_r)
    plot_h = max(1, height - margin_t - margin_b)

    xmin, xmax = float(min(xs)), float(max(xs))
    ymin, ymax = float(min(ys)), float(max(ys))
    if xmax == xmin:
        xmax = xmin + 1.0
    if ymax == ymin:
        ymax = ymin + 1.0

    def x(v: float) -> int:
        return margin_l + int(((v - xmin) / (xmax - xmin)) * plot_w)

    def y(v: float) -> int:
        return margin_t + int((1 - ((v - ymin) / (ymax - ymin))) * plot_h)

    points = " ".join(f"{x(a)},{y(b)}" for a, b in zip(xs, ys))

    parts: List[str] = []
    parts.append(f'<svg viewBox="0 0 {width} {height}" role="img">')
    parts.append(
        f'<text x="{margin_l}" y="{margin_t - 10}" font-size="14" font-weight="600">{html.escape(title)}</text>'
    )

    parts.append(
        f'<line x1="{margin_l}" y1="{margin_t + plot_h}" x2="{margin_l + plot_w}" y2="{margin_t + plot_h}" stroke="#999"/>'
    )
    parts.append(
        f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t + plot_h}" stroke="#999"/>'
    )

    parts.append(f'<text x="{margin_l}" y="{height - 10}" font-size="12" fill="#333">eps</text>')
    parts.append(f'<text x="8" y="{margin_t + 10}" font-size="12" fill="#333">{html.escape(y_label)}</text>')

    parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
    for a, b in zip(xs, ys):
        parts.append(f'<circle cx="{x(a)}" cy="{y(b)}" r="3" fill="{color}"/>')

    parts.append("</svg>")
    return "".join(parts)


def _default_eps_sweep(center: float) -> List[float]:
    deltas = [-0.08, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.08]
    return sorted({round(max(0.01, center + d), 3) for d in deltas})


def _parse_eps_sweep(raw: str, *, center: float) -> List[float]:
    raw = (raw or "").strip()
    if not raw:
        return _default_eps_sweep(center)

    vals: List[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            vals.append(float(part))
        except Exception:
            continue
    return sorted({round(v, 3) for v in vals if v > 0})


def compute_day_metrics(
    *,
    day_key: str,
    eps: float,
    min_samples: int,
    rebuild_cache: bool,
) -> Dict[str, Any]:
    store, raw_count, distinct_stories = _get_day_data(
        day_key=day_key,
        eps=eps,
        min_samples=min_samples,
        rebuild_cache=rebuild_cache,
    )

    distinct_count = len(distinct_stories)
    compression = compute_compression_rate(raw_count=raw_count, distinct_count=distinct_count)
    noise_rate = compute_noise_rate(raw_count=raw_count, distinct_stories=distinct_stories)
    cluster_stats = compute_cluster_size_stats(distinct_stories=distinct_stories)
    cohesion = compute_within_cluster_similarity(store=store, distinct_stories=distinct_stories)

    cluster_sizes = [_safe_int(s.get("cluster_size") or 1, 1) for s in distinct_stories]
    singleton_count = sum(1 for s in cluster_sizes if s == 1)

    return {
        "day_key": day_key,
        "eps": float(eps),
        "min_samples": int(min_samples),
        "raw_count": int(raw_count),
        "distinct_count": int(distinct_count),
        "compression_rate": float(compression),
        "noise_rate": float(noise_rate),
        "singleton_count": int(singleton_count),
        "cluster_sizes": cluster_sizes,
        "cluster_size_stats": cluster_stats,
        "within_cluster_similarity": cohesion,
    }


def run_eps_sweep(
    *,
    day_key: str,
    eps_values: List[float],
    min_samples: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for eps in eps_values:
        out.append(
            compute_day_metrics(
                day_key=day_key,
                eps=float(eps),
                min_samples=min_samples,
                rebuild_cache=True,
            )
        )
    return out


def write_graphs(
    *,
    out_dir: Path,
    base: Dict[str, Any],
    sweep: List[Dict[str, Any]],
) -> Dict[str, Path]:
    day_key = str(base.get("day_key"))
    report_dir = out_dir / day_key
    report_dir.mkdir(parents=True, exist_ok=True)

    raw_count = int(base.get("raw_count") or 0)
    distinct_count = int(base.get("distinct_count") or 0)

    cluster_sizes: List[int] = list(base.get("cluster_sizes") or [])
    bucket_labels, bucket_counts = _bucket_cluster_sizes(cluster_sizes)

    singleton_count = int(base.get("singleton_count") or 0)
    clustered_count = max(0, raw_count - singleton_count)

    outputs: Dict[str, Path] = {}

    outputs["raw_vs_distinct"] = report_dir / "raw_vs_distinct.svg"
    _write_text(
        outputs["raw_vs_distinct"],
        _svg_bar_chart(
            title="Raw Articles vs Distinct Stories",
            labels=["Raw", "Distinct"],
            values=[float(raw_count), float(distinct_count)],
        ),
    )

    outputs["cluster_size_distribution"] = report_dir / "cluster_size_distribution.svg"
    _write_text(
        outputs["cluster_size_distribution"],
        _svg_bar_chart(
            title="Cluster Size Distribution (bucketed)",
            labels=bucket_labels,
            values=[float(v) for v in bucket_counts],
            bar_color="#F58518",
        ),
    )

    outputs["noise_ratio"] = report_dir / "noise_ratio.svg"
    _write_text(
        outputs["noise_ratio"],
        _svg_stacked_bar(
            title="Noise vs Clustered (raw articles)",
            a_label="Clustered",
            a_value=int(clustered_count),
            b_label="Noise (singletons)",
            b_value=int(singleton_count),
        ),
    )

    if sweep:
        xs = [float(r["eps"]) for r in sweep]

        outputs["eps_sweep_distinct_count"] = report_dir / "eps_sweep_distinct_count.svg"
        _write_text(
            outputs["eps_sweep_distinct_count"],
            _svg_line_chart(
                title="eps sweep: distinct_count",
                xs=xs,
                ys=[float(r["distinct_count"]) for r in sweep],
                y_label="distinct",
                color="#4C78A8",
            ),
        )

        outputs["eps_sweep_noise_rate"] = report_dir / "eps_sweep_noise_rate.svg"
        _write_text(
            outputs["eps_sweep_noise_rate"],
            _svg_line_chart(
                title="eps sweep: noise_rate",
                xs=xs,
                ys=[float(r["noise_rate"]) for r in sweep],
                y_label="noise",
                color="#E45756",
            ),
        )

        outputs["eps_sweep_max_cluster_size"] = report_dir / "eps_sweep_max_cluster_size.svg"
        _write_text(
            outputs["eps_sweep_max_cluster_size"],
            _svg_line_chart(
                title="eps sweep: max_cluster_size",
                xs=xs,
                ys=[float(r["cluster_size_stats"]["max"]) for r in sweep],
                y_label="max",
                color="#54A24B",
            ),
        )

        outputs["eps_sweep_csv"] = report_dir / "eps_sweep.csv"
        with outputs["eps_sweep_csv"].open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["eps", "raw_count", "distinct_count", "noise_rate", "max_cluster_size"])
            for r in sweep:
                w.writerow(
                    [
                        f"{float(r['eps']):.3f}",
                        int(r.get("raw_count") or 0),
                        int(r.get("distinct_count") or 0),
                        f"{float(r.get('noise_rate') or 0):.6f}",
                        int(float(r.get("cluster_size_stats", {}).get("max") or 0)),
                    ]
                )

    return outputs


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate dedup graphs (SVG)")
    parser.add_argument(
        "--day-key",
        type=str,
        default=datetime.now().astimezone().strftime("%Y-%m-%d"),
        help="Local day in YYYY-MM-DD (default: today).",
    )
    parser.add_argument("--eps", type=float, default=0.28)
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Delete cached distinct stories for the day before computing metrics.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(Path("reports") / "dedup_eval"),
        help="Output directory for the graphs (default: reports/dedup_eval).",
    )
    parser.add_argument(
        "--no-eps-sweep",
        action="store_true",
        help="Skip eps sweep (faster).",
    )
    parser.add_argument(
        "--eps-sweep",
        type=str,
        default="",
        help="Comma-separated eps values (overrides default sweep around --eps).",
    )

    args = parser.parse_args(argv)

    try:
        base = compute_day_metrics(
            day_key=str(args.day_key),
            eps=float(args.eps),
            min_samples=int(args.min_samples),
            rebuild_cache=bool(args.rebuild_cache),
        )

        sweep: List[Dict[str, Any]] = []
        if not bool(args.no_eps_sweep):
            eps_values = _parse_eps_sweep(str(args.eps_sweep), center=float(args.eps))
            sweep = run_eps_sweep(
                day_key=str(args.day_key),
                eps_values=eps_values,
                min_samples=int(args.min_samples),
            )

    except ModuleNotFoundError as exc:
        print(_missing_dependency_message(exc), file=sys.stderr)
        return 2

    outputs = write_graphs(out_dir=Path(str(args.out_dir)), base=base, sweep=sweep)
    for k in sorted(outputs):
        print(f"{k}={outputs[k]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())