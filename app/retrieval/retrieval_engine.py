import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, List, Optional
from sklearn.cluster import DBSCAN
from datetime import datetime, timedelta, timezone
from app.memory.vector_store import VectorStore


class RetrievalEngine:
    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
    
    @staticmethod
    def _cluster_distinct_stories(
        embeddings: np.ndarray,
        metadatas: List[dict],
        documents: List[str],
        eps: float = 0.35,
        min_samples: int = 2,
    ) -> List[dict]:
        if embeddings.size == 0:
            return []

        # DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
        labels = clustering.fit_predict(embeddings)

        # Build clusters: each DBSCAN cluster label is a story; noise points (-1) are singletons.
        clusters: List[List[int]] = []
        unique_labels = set(labels)

        for label in sorted(unique_labels):
            if label == -1:
                noise_indices = [i for i, l in enumerate(labels) if l == -1]
                clusters.extend([[i] for i in noise_indices])
            else:
                clusters.append([i for i, l in enumerate(labels) if l == label])

        stories: List[dict] = []

        for indices in clusters:
            if not indices:
                continue

            if len(indices) == 1:
                best_idx = indices[0]
            else:
                cluster_embeddings = embeddings[indices]

                # Centrality scoring using cosine similarity (normalize then dot-product).
                norms = np.linalg.norm(cluster_embeddings, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-12)
                normed = cluster_embeddings / norms

                similarity_matrix = normed @ normed.T
                centrality_scores = similarity_matrix.mean(axis=1)

                best_idx = indices[int(np.argmax(centrality_scores))]

            stories.append(
                {
                    "headline": metadatas[best_idx].get("title"),
                    "document": documents[best_idx],
                    "metadata": metadatas[best_idx],
                    "cluster_size": len(indices),
                }
            )

        # Present most-recent first when available.
        def _ts(story: dict) -> str:
            return (story.get("metadata") or {}).get("timestamp_added") or ""

        stories.sort(key=_ts, reverse=True)
        return stories

    @staticmethod
    def _cluster_distinct_stories_with_members(
        ids: List[str],
        embeddings: np.ndarray,
        metadatas: List[dict],
        documents: List[str],
        eps: float = 0.35,
        min_samples: int = 2,
    ) -> List[dict]:
        """
        Like `_cluster_distinct_stories`, but also returns member_article_ids and
        the representative's embedding for caching.
        """
        if embeddings.size == 0:
            return []

        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
        labels = clustering.fit_predict(embeddings)

        clusters: List[List[int]] = []
        unique_labels = set(labels)

        for label in sorted(unique_labels):
            if label == -1:
                noise_indices = [i for i, l in enumerate(labels) if l == -1]
                clusters.extend([[i] for i in noise_indices])
            else:
                clusters.append([i for i, l in enumerate(labels) if l == label])

        stories: List[dict] = []

        for indices in clusters:
            if not indices:
                continue

            if len(indices) == 1:
                best_idx = indices[0]
            else:
                cluster_embeddings = embeddings[indices]
                norms = np.linalg.norm(cluster_embeddings, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-12)
                normed = cluster_embeddings / norms

                similarity_matrix = normed @ normed.T
                centrality_scores = similarity_matrix.mean(axis=1)
                best_idx = indices[int(np.argmax(centrality_scores))]

            member_ids = [ids[i] for i in indices if i < len(ids)]
            rep_id = ids[best_idx] if best_idx < len(ids) else None
            rep_embedding = embeddings[best_idx].tolist() if best_idx < len(embeddings) else []

            stories.append(
                {
                    "headline": metadatas[best_idx].get("title"),
                    "document": documents[best_idx],
                    "metadata": metadatas[best_idx],
                    "cluster_size": len(indices),
                    "rep_article_id": rep_id,
                    "member_article_ids": member_ids,
                    "embedding": rep_embedding,
                }
            )

        def _ts(story: dict) -> str:
            return (story.get("metadata") or {}).get("ingested_at") or ""

        stories.sort(key=_ts, reverse=True)
        return stories

    def retrieve_distinct_stories(
    self,
    query: str,
    dense_k: int = 20,
    eps: float = 0.35,
    min_samples: int = 2,
):
        # Prefer cached representatives first for speed and stable IDs.
        since_day_key = (datetime.now().astimezone().date() - timedelta(days=30)).strftime("%Y-%m-%d")
        distinct = self.store.query_distinct(query_text=query, top_k=dense_k, since_day_key=since_day_key)

        if distinct.get("documents") and distinct["documents"][0]:
            ids = distinct.get("ids", [[]])[0]
            metadatas = distinct.get("metadatas", [[]])[0]
            documents = distinct.get("documents", [[]])[0]

            stories: List[dict] = []
            for idx, metadata in enumerate(metadatas):
                stories.append(
                    {
                        "headline": (metadata or {}).get("headline") or "",
                        "document": documents[idx] if idx < len(documents) else "",
                        "metadata": metadata,
                        "cluster_size": int((metadata or {}).get("cluster_size") or 1),
                        "distinct_story_id": ids[idx] if idx < len(ids) else None,
                    }
                )
            return stories

        # Cache miss path: use raw_fetched search to discover relevant day_keys,
        # build daily caches for those days, then re-run distinct search.
        raw = self.store.query_raw(query_text=query, top_k=max(10, dense_k))
        raw_metas = (raw.get("metadatas") or [[]])[0]
        day_keys: List[str] = []

        for metadata in raw_metas:
            ingested_at = (metadata or {}).get("ingested_at")
            if not ingested_at:
                continue
            try:
                dt = datetime.fromisoformat(ingested_at)
            except Exception:
                continue
            day_keys.append(dt.astimezone().strftime("%Y-%m-%d"))

        for day_key in sorted(set(day_keys)):
            self.retrieve_distinct_stories_for_day(day_key=day_key, eps=eps, min_samples=min_samples)

        distinct2 = self.store.query_distinct(query_text=query, top_k=dense_k, since_day_key=since_day_key)
        if distinct2.get("documents") and distinct2["documents"][0]:
            ids = distinct2.get("ids", [[]])[0]
            metadatas = distinct2.get("metadatas", [[]])[0]
            documents = distinct2.get("documents", [[]])[0]

            stories: List[dict] = []
            for idx, metadata in enumerate(metadatas):
                stories.append(
                    {
                        "headline": (metadata or {}).get("headline") or "",
                        "document": documents[idx] if idx < len(documents) else "",
                        "metadata": metadata,
                        "cluster_size": int((metadata or {}).get("cluster_size") or 1),
                        "distinct_story_id": ids[idx] if idx < len(ids) else None,
                    }
                )
            return stories

        return []

    def retrieve_distinct_stories_by_recency(
        self,
        since: datetime,
        until: Optional[datetime] = None,
        eps: float = 0.35,
        min_samples: int = 2,
    ) -> List[dict]:
        results = self.store.get_recent_articles(
            since=since,
            until=until,
            include_embeddings=True,
        )

        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or []
        embeddings_list = results.get("embeddings") or []

        if not documents:
            return []

        embeddings = np.array(embeddings_list)

        return self._cluster_distinct_stories(
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
            eps=eps,
            min_samples=min_samples,
        )

    def retrieve_distinct_stories_today(
        self,
        eps: float = 0.35,
        min_samples: int = 2,
    ) -> List[dict]:
        now = datetime.now().astimezone()
        start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)

        day_key = start_of_day.strftime("%Y-%m-%d")
        return self.retrieve_distinct_stories_for_day(day_key=day_key, eps=eps, min_samples=min_samples)

    def retrieve_distinct_stories_for_day(
        self,
        day_key: str,
        eps: float = 0.35,
        min_samples: int = 2,
    ) -> List[dict]:
        """
        Fetch distinct stories for a given local day (YYYY-MM-DD).

        Uses the distinct_stories cache when present; otherwise builds it by
        clustering all raw_fetched articles ingested within that day window.
        """
        cached = self.store.get_distinct_stories(day_key=day_key)
        if cached:
            return cached

        try:
            day = datetime.strptime(day_key, "%Y-%m-%d").date()
        except ValueError:
            return []

        local_tz = datetime.now().astimezone().tzinfo
        start = datetime(day.year, day.month, day.day, tzinfo=local_tz)
        end = start + timedelta(days=1)

        raw = self.store.get_raw_articles(since=start, until=end, include_embeddings=True)

        raw_ids = raw.get("ids") or []
        raw_docs = raw.get("documents") or []
        raw_metas = raw.get("metadatas") or []
        raw_embs = raw.get("embeddings") or []

        if not raw_docs:
            return []

        embeddings = np.array(raw_embs)

        clustered = self._cluster_distinct_stories_with_members(
            ids=raw_ids,
            embeddings=embeddings,
            metadatas=raw_metas,
            documents=raw_docs,
            eps=eps,
            min_samples=min_samples,
        )

        run_metadata = {
            "since_utc": start.astimezone(timezone.utc).isoformat(),
            "until_utc": end.astimezone(timezone.utc).isoformat(),
            "selection_basis": "ingested_at",
            "embedding_model": "all-MiniLM-L6-v2",
            "cluster_algo": "dbscan",
            "eps": eps,
            "min_samples": min_samples,
            "metric": "cosine",
            "raw_count": len(raw_ids),
            "distinct_count": len(clustered),
        }

        self.store.cache_distinct_stories(
            day_key=day_key,
            stories=clustered,
            run_metadata=run_metadata,
            replace=True,
        )

        return self.store.get_distinct_stories(day_key=day_key)

    def retrieve_distinct_stories_last_hours(
        self,
        hours: int = 24,
        eps: float = 0.35,
        min_samples: int = 2,
    ) -> List[dict]:
        now = datetime.now().astimezone()
        since = now - timedelta(hours=hours)

        return self.retrieve_distinct_stories_by_recency(
            since=since,
            until=now,
            eps=eps,
            min_samples=min_samples,
        )

    
