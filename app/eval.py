"""Dedup evaluation metrics.

This file is intentionally a thin wrapper around the *real* dedup pipeline.
It calls `RetrievalEngine.retrieve_distinct_stories_for_day(...)` and computes
metrics from the returned distinct stories.

Metrics implemented so far:
1) Compression rate = distinct_count / raw_count
2) Noise rate (singleton share) = singleton_count / raw_count
3) Cluster size stats = max / median / p95 of `cluster_size`
4) Within-cluster similarity (cohesion) = similarity(member, rep) summary stats

Notes on caching:
- `retrieve_distinct_stories_for_day` returns a cached result if one exists for
  that `day_key`. In that case, the `--eps/--min-samples` arguments may NOT take
  effect unless the cache is rebuilt (use --rebuild-cache).

Notes on cohesion metric:
- We do not re-run DBSCAN.
- We use cached cluster membership (`member_article_ids_json`) + the cached
  representative raw article ID (`rep_article_id`) stored in distinct-story
  metadata.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _ensure_project_root_on_syspath() -> None:
    """Make this file runnable via IDE Run/F5 without manual PYTHONPATH tweaks.

    When executed as a script (e.g., `python app/eval.py`) some IDEs set the
    working directory such that `import app...` fails. We fix that by adding the
    repository root (parent of the `app/` directory) to `sys.path` based on this
    file's location.
    """

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


_ensure_project_root_on_syspath()

if TYPE_CHECKING:
    from app.memory.vector_store import VectorStore
    from app.retrieval.retrieval_engine import RetrievalEngine


def _get_distinct_stories_for_day(
    engine: RetrievalEngine,
    day_key: str,
    eps: float,
    min_samples: int,
) -> List[Dict[str, Any]]:
    """Return distinct stories for the day using the project's retrieval path."""
    return engine.retrieve_distinct_stories_for_day(
        day_key=day_key,
        eps=eps,
        min_samples=min_samples,
    )


def _get_raw_count_from_distinct_metadata(
    store: VectorStore,
    distinct_stories: List[Dict[str, Any]],
) -> int:
    """Get the raw article count that the pipeline clustered over.

    Preferred source is the distinct-story cache metadata (written by
    `VectorStore.cache_distinct_stories` from `run_metadata["raw_count"]`).

    If that value is missing, fall back to counting raw_fetched articles using
    the cached UTC window (`since_utc`, `until_utc`) from the same metadata.

    Returns 0 if it cannot be determined.
    """
    if not distinct_stories:
        return 0

    md = (distinct_stories[0].get("metadata") or {})

    raw_count = int(md.get("raw_count") or 0)
    if raw_count > 0:
        return raw_count

    since_utc = md.get("since_utc")
    until_utc = md.get("until_utc")
    if not since_utc or not until_utc:
        return 0

    try:
        since = datetime.fromisoformat(str(since_utc))
        until = datetime.fromisoformat(str(until_utc))
    except Exception:
        return 0

    raw = store.get_raw_articles(since=since, until=until, include_embeddings=False)
    return len(raw.get("ids") or [])


def _get_day_data(
    *,
    day_key: str,
    eps: float,
    min_samples: int,
    rebuild_cache: bool = False,
) -> Tuple[VectorStore, int, List[Dict[str, Any]]]:
    """Fetch distinct stories and the raw_count denominator for the requested day."""
    from app.memory.vector_store import VectorStore
    from app.retrieval.retrieval_engine import RetrievalEngine

    store = VectorStore()
    engine = RetrievalEngine(store)

    if rebuild_cache:
        cached = store.get_distinct_stories(day_key)
        cached_ids = [
            s.get("distinct_story_id") for s in cached if s.get("distinct_story_id")
        ]
        if cached_ids:
            store.delete_distinct_stories_by_ids([str(i) for i in cached_ids])
            print(
                f"Rebuild cache: deleted {len(cached_ids)} cached distinct stories "
                f"for {day_key}."
            )

    distinct_stories = _get_distinct_stories_for_day(
        engine=engine,
        day_key=day_key,
        eps=eps,
        min_samples=min_samples,
    )

    raw_count = _get_raw_count_from_distinct_metadata(store, distinct_stories)
    return store, raw_count, distinct_stories


def compute_compression_rate(*, raw_count: int, distinct_count: int) -> float:
    """Compute compression rate = distinct_count / raw_count."""
    if raw_count <= 0:
        return 0.0
    return float(distinct_count / raw_count)


def compute_noise_rate(
    *, raw_count: int, distinct_stories: List[Dict[str, Any]]
) -> float:
    """Compute noise rate (singleton share) = singleton_count / raw_count.

    With DBSCAN(min_samples=2), singleton distinct stories correspond to DBSCAN
    noise points (articles that did not cluster with any other article).

    We use the pipeline's cached `cluster_size` field; we do not re-run DBSCAN.
    """
    if raw_count <= 0:
        return 0.0

    singleton_count = sum(
        1 for s in distinct_stories if int(s.get("cluster_size") or 1) == 1
    )
    return float(singleton_count / raw_count)


def compute_cluster_size_stats(
    *, distinct_stories: List[Dict[str, Any]]
) -> Dict[str, float]:
    """Compute simple cluster-size distribution stats from `cluster_size`.

    Returns a dict with:
    - max: maximum cluster_size
    - median: 50th percentile of cluster_size
    - p95: 95th percentile of cluster_size
    """
    sizes = [int(s.get("cluster_size") or 1) for s in distinct_stories]
    if not sizes:
        return {"max": 0.0, "median": 0.0, "p95": 0.0}

    arr = np.array(sizes, dtype=float)
    return {
        "max": float(np.max(arr)),
        "median": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    denom = (float(np.linalg.norm(a)) * float(np.linalg.norm(b))) or 1e-12
    return float(np.dot(a, b) / denom)


def _get_raw_embeddings_by_id(
    store: VectorStore,
    article_ids: List[str],
) -> Dict[str, np.ndarray]:
    """Fetch raw_fetched embeddings for specific raw article IDs.

    This reads from the existing Chroma raw collection; it does not compute
    embeddings.
    """
    if not article_ids:
        return {}

    try:
        res = store.raw_collection.get(ids=article_ids, include=["embeddings"])
    except Exception:
        return {}

    ids = res.get("ids") or []
    embs = res.get("embeddings")
    if embs is None:
        embs_list = []
    elif hasattr(embs, "tolist"):
        embs_list = embs.tolist()
    else:
        embs_list = embs

    out: Dict[str, np.ndarray] = {}
    for i, emb in zip(ids, embs_list):
        if i and emb:
            out[str(i)] = np.array(emb, dtype=float)
    return out


def compute_within_cluster_similarity(
    *,
    store: VectorStore,
    distinct_stories: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Compute within-cluster similarity (cohesion).

    For each cluster with size >= 2, compute cosine similarity of each member to
    the representative (excluding the rep itself). Then summarize across
    clusters.

    Returns a dict with:
    - mean_of_means: mean over clusters of mean(sim(member, rep))
    - p05_of_mins: 5th percentile over clusters of min(sim(member, rep))
    - min_of_mins: minimum over clusters of min(sim(member, rep))
    """
    distinct_ids = [
        s.get("distinct_story_id")
        for s in distinct_stories
        if s.get("distinct_story_id")
    ]
    if not distinct_ids:
        return {"mean_of_means": 0.0, "p05_of_mins": 0.0, "min_of_mins": 0.0}

    member_map = store.get_distinct_member_article_ids_map(
        [str(i) for i in distinct_ids]
    )

    # Build one raw-embedding map for all members we might need.
    all_member_ids: List[str] = []
    for member_ids in member_map.values():
        all_member_ids.extend([str(x) for x in (member_ids or []) if x])

    raw_emb_by_id = _get_raw_embeddings_by_id(store, sorted(set(all_member_ids)))

    per_cluster_means: List[float] = []
    per_cluster_mins: List[float] = []

    for story in distinct_stories:
        if int(story.get("cluster_size") or 1) < 2:
            continue

        distinct_story_id = story.get("distinct_story_id")
        if not distinct_story_id:
            continue

        md = story.get("metadata") or {}
        rep_article_id = md.get("rep_article_id")
        if not rep_article_id:
            continue

        rep_vec = raw_emb_by_id.get(str(rep_article_id))
        if rep_vec is None:
            continue

        member_ids = member_map.get(str(distinct_story_id)) or []
        member_vecs: List[np.ndarray] = []
        for mid in member_ids:
            if not mid or str(mid) == str(rep_article_id):
                continue
            v = raw_emb_by_id.get(str(mid))
            if v is not None:
                member_vecs.append(v)

        if not member_vecs:
            continue

        sims = [_cosine_similarity(v, rep_vec) for v in member_vecs]
        per_cluster_means.append(float(np.mean(np.array(sims, dtype=float))))
        per_cluster_mins.append(float(np.min(np.array(sims, dtype=float))))

    if not per_cluster_means or not per_cluster_mins:
        return {"mean_of_means": 0.0, "p05_of_mins": 0.0, "min_of_mins": 0.0}

    means_arr = np.array(per_cluster_means, dtype=float)
    mins_arr = np.array(per_cluster_mins, dtype=float)

    return {
        "mean_of_means": float(np.mean(means_arr)),
        "p05_of_mins": float(np.percentile(mins_arr, 5)),
        "min_of_mins": float(np.min(mins_arr)),
    }


def run_eval(
    *,
    day_key: str,
    eps: float = 0.28,
    min_samples: int = 2,
    rebuild_cache: bool = False,
) -> Dict[str, Any]:
    """Run the evaluation and return metrics as a dict (IDE-friendly)."""
    store, raw_count, distinct_stories = _get_day_data(
        day_key=str(day_key),
        eps=float(eps),
        min_samples=int(min_samples),
        rebuild_cache=bool(rebuild_cache),
    )

    distinct_count = len(distinct_stories)

    compression = compute_compression_rate(
        raw_count=raw_count,
        distinct_count=distinct_count,
    )
    noise_rate = compute_noise_rate(
        raw_count=raw_count,
        distinct_stories=distinct_stories,
    )
    cluster_stats = compute_cluster_size_stats(distinct_stories=distinct_stories)
    cohesion = compute_within_cluster_similarity(
        store=store,
        distinct_stories=distinct_stories,
    )

    return {
        "day_key": str(day_key),
        "eps": float(eps),
        "min_samples": int(min_samples),
        "raw_count": int(raw_count),
        "distinct_count": int(distinct_count),
        "compression_rate": float(compression),
        "noise_rate": float(noise_rate),
        "cluster_size_stats": cluster_stats,
        "within_cluster_similarity": cohesion,
    }


def _missing_dependency_message(exc: ModuleNotFoundError) -> str:
    missing = getattr(exc, "name", None) or "(unknown module)"

    project_root = Path(__file__).resolve().parents[1]
    win_venv = project_root / ".venv" / "Scripts" / "python.exe"
    nix_venv = project_root / ".venv" / "bin" / "python"

    venv_hint = ""
    if win_venv.exists():
        venv_hint = str(win_venv)
    elif nix_venv.exists():
        venv_hint = str(nix_venv)

    lines = [
        f"Missing Python dependency: {missing}",
        f"Interpreter used: {sys.executable}",
        "",
        "Fix (recommended): in VS Code select the project interpreter:",
        (
            f"  - Python: Select Interpreter -> {venv_hint}"
            if venv_hint
            else "  - Python: Select Interpreter -> .venv"
        ),
        "",
        "If you're using the Code Runner extension, it may run the system Python.",
        "Configure it to use the `.venv` interpreter (or enable 'Run In Terminal').",
        "",
        (
            "If the package is genuinely missing from your environment, install "
            "deps from `requirements.txt`."
        ),
    ]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entrypoint: runnable via Run/F5 (defaults) or as a CLI with flags."""
    parser = argparse.ArgumentParser(description="Evaluate dedup metrics for a day")
    parser.add_argument(
        "--day-key",
        type=str,
        default=datetime.now().astimezone().strftime("%Y-%m-%d"),
        help="Local day in YYYY-MM-DD (default: today).",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.28,
        help="DBSCAN eps (cosine distance). Smaller = more conservative.",
    )
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Delete cached distinct stories for the day before computing metrics.",
    )

    args = parser.parse_args(argv)

    try:
        result = run_eval(
            day_key=str(args.day_key),
            eps=float(args.eps),
            min_samples=int(args.min_samples),
            rebuild_cache=bool(args.rebuild_cache),
        )
    except ModuleNotFoundError as exc:
        print(_missing_dependency_message(exc), file=sys.stderr)
        return 2

    print(f"day_key={args.day_key} eps={args.eps} min_samples={args.min_samples}")
    print(f"raw_count={result['raw_count']} distinct_count={result['distinct_count']}")
    print(f"compression_rate(distinct/raw)={result['compression_rate']:.4f}")
    print(f"noise_rate(singleton/raw)={result['noise_rate']:.4f}")
    print(
        "cluster_size_stats="
        f"max:{result['cluster_size_stats']['max']:.1f} "
        f"median:{result['cluster_size_stats']['median']:.1f} "
        f"p95:{result['cluster_size_stats']['p95']:.1f}"
    )
    print(
        "within_cluster_similarity="
        f"meanOfMeans:{result['within_cluster_similarity']['mean_of_means']:.3f} "
        f"p05OfMins:{result['within_cluster_similarity']['p05_of_mins']:.3f} "
        f"minOfMins:{result['within_cluster_similarity']['min_of_mins']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
