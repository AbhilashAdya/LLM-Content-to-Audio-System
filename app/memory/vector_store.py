import json
import hashlib
from datetime import datetime, timedelta, timezone
import os
from typing import List, Optional, Dict, Any

import chromadb
from sentence_transformers import SentenceTransformer

from app.data_ingestion.schemas import Article


class VectorStore:
    def __init__(self):
        # Embedding model (can change later)
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Persistent Chroma client
        self.client = chromadb.PersistentClient(path="chroma_db")
        if not os.path.exists("chroma_db"):
            print("ChromaDB directory not created.")
        else:
            print("ChromaDB directory created successfully.")

        # Collections
        # - raw_fetched: everything ingested (exact de-dupe only)
        # - distinct_stories: one representative per cluster/day (cached)
        # - recent/important: legacy paths kept for backwards compatibility
        self.raw_collection = self.client.get_or_create_collection(
            name="ai_news_raw_fetched",
            metadata={"hnsw:space": "cosine"},
        )

        self.distinct_collection = self.client.get_or_create_collection(
            name="ai_news_distinct_stories",
            metadata={"hnsw:space": "cosine"},
        )

        # Two collections: recent + important
        self.recent_collection = self.client.get_or_create_collection(
            name="ai_news_recent",
            metadata={"hnsw:space": "cosine"},
        )

        self.important_collection = self.client.get_or_create_collection(
            name="ai_news_important",
            metadata={"hnsw:space": "cosine"},
        )

    # -----------------------------
    # ID GENERATION
    # -----------------------------
    # Does not depend on input state; transforms input -> output.
    @staticmethod
    def generate_article_id(article: Article) -> str:
        unique_string = f"{article.title}|" f"{article.url}|" f"{article.published_at}"
        return hashlib.sha256(unique_string.encode()).hexdigest()

    # -----------------------------
    # ADD ARTICLES (RECENT)
    # -----------------------------
    def add_articles(self, articles: List[Article]) -> int:
        added = 0
        for article in articles:
            if article.summary is None:
                continue

            article_id = self.generate_article_id(article)

            # Check if already exists
            if self._exists(article_id):
                continue

            embedding = self.embedding_model.encode(article.summary).tolist()

            self.recent_collection.add(
                ids=[article_id],
                embeddings=[embedding],
                documents=[article.summary],
                metadatas=[
                    {
                        "title": article.title,
                        "url": article.url,
                        "source": article.source,
                        "published_at": str(article.published_at),
                        "timestamp_added": datetime.now(timezone.utc).isoformat(),
                        "raw_text": article.summary,
                    }
                ],
            )
            added += 1

        return added

    # -----------------------------
    # ADD ARTICLES (RAW FETCHED)
    # -----------------------------
    def add_raw_articles(self, articles: List[Article]) -> int:
        """
        Store all fetched articles in the raw collection.

        This is the long-lived source of truth for ingestion and clustering.
        We only skip exact duplicates based on the deterministic article_id.
        """
        added = 0

        for article in articles:
            if article.summary is None:
                continue

            article_id = self.generate_article_id(article)

            # Exact de-dupe: do not re-add identical IDs.
            try:
                exists = bool(self.raw_collection.get(ids=[article_id]).get("ids"))
            except Exception:
                exists = False

            if exists:
                continue

            embedding = self.embedding_model.encode(article.summary).tolist()
            ingested_at = datetime.now(timezone.utc).isoformat()

            self.raw_collection.add(
                ids=[article_id],
                embeddings=[embedding],
                documents=[article.summary],
                metadatas=[
                    {
                        "title": article.title,
                        "url": article.url,
                        "source": article.source,
                        "published_at": str(article.published_at),
                        "ingested_at": ingested_at,
                        "raw_text": article.summary,
                    }
                ],
            )
            added += 1

        return added

    # -----------------------------
    # MOVE TO IMPORTANT
    # -----------------------------
    def mark_as_important(self, article_id: str) -> None:
        """
        Move an article from recent to the important collection by ID.
        """
        result = self.recent_collection.get(ids=[article_id])

        # No such article. Prevent error by checking existence first.
        if not result["documents"]:
            return

        self.important_collection.add(
            ids=result["ids"],
            embeddings=result["embeddings"],
            documents=result["documents"],
            metadatas=result["metadatas"],
        )

    # -----------------------------
    # QUERY RECENT
    # -----------------------------
    def query_recent(self, query_text: str, top_k: int = 3):
        query_embedding = self.embedding_model.encode(query_text).tolist()

        return self.recent_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "embeddings", "distances"],
        )

    def query_raw(self, query_text: str, top_k: int = 20):
        query_embedding = self.embedding_model.encode(query_text).tolist()

        return self.raw_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "embeddings", "distances"],
        )

    def query_distinct(
        self,
        query_text: str,
        top_k: int = 20,
        since_day_key: Optional[str] = None,
    ):
        query_embedding = self.embedding_model.encode(query_text).tolist()

        where = None
        if since_day_key is not None:
            where = {"day_key": {"$gte": since_day_key}}

        try:
            if where is not None:
                return self.distinct_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=where,
                    include=["documents", "metadatas", "embeddings", "distances"],
                )
        except Exception:
            pass

        return self.distinct_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "embeddings", "distances"],
        )

    # -----------------------------
    # GET RECENT (RECENCY WINDOW)
    # -----------------------------
    def get_recent_articles(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        include_embeddings: bool = True,
    ) -> Dict[str, Any]:
        """
        Fetch all recent articles (not top_k) optionally filtered by timestamp_added.

        `timestamp_added` is stored as an ISO-8601 string (UTC). We attempt a
        server-side metadata filter first, then fall back to filtering in Python
        for compatibility.
        """

        def _ensure_aware(dt: Optional[datetime]) -> Optional[datetime]:
            if dt is None:
                return None
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt

        since = _ensure_aware(since)
        until = _ensure_aware(until)

        include = ["ids", "documents", "metadatas"]
        if include_embeddings:
            include.append("embeddings")

        results: Dict[str, Any]

        # Best-effort: use Chroma's `where` filter when available.
        where = None
        if since is not None or until is not None:
            cond: Dict[str, str] = {}
            if since is not None:
                cond["$gte"] = since.astimezone(timezone.utc).isoformat()
            if until is not None:
                cond["$lt"] = until.astimezone(timezone.utc).isoformat()
            if cond:
                where = {"timestamp_added": cond}

        try:
            if where is not None:
                results = self.recent_collection.get(where=where, include=include)
            else:
                results = self.recent_collection.get(include=include)
        except Exception:
            # Fallback for older Chroma versions / metadata filter limitations.
            results = self.recent_collection.get(include=include)

        # Always apply a local filter to be safe and consistent.
        if since is None and until is None:
            return results

        ids = results.get("ids") or []
        metadatas = results.get("metadatas") or []
        documents = results.get("documents") or []
        embeddings = results.get("embeddings") if include_embeddings else None

        keep_indices: List[int] = []
        for idx, metadata in enumerate(metadatas):
            timestamp_str = (metadata or {}).get("timestamp_added")
            if not timestamp_str:
                continue

            try:
                timestamp = datetime.fromisoformat(timestamp_str)
            except Exception:
                continue

            if since is not None and timestamp < since:
                continue
            if until is not None and timestamp >= until:
                continue

            keep_indices.append(idx)

        filtered: Dict[str, Any] = {
            "ids": [ids[i] for i in keep_indices],
            "documents": [documents[i] for i in keep_indices],
            "metadatas": [metadatas[i] for i in keep_indices],
        }

        if include_embeddings:
            filtered["embeddings"] = (
                [embeddings[i] for i in keep_indices] if embeddings else []
            )

        return filtered

    # -----------------------------
    # GET RAW (RECENCY WINDOW)
    # -----------------------------
    def get_raw_articles(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        include_embeddings: bool = True,
    ) -> Dict[str, Any]:
        """
        Fetch all raw-fetched articles optionally filtered by ingested_at.

        `ingested_at` is stored as an ISO-8601 string (UTC). We attempt a
        server-side metadata filter first, then fall back to filtering in Python
        for compatibility.
        """

        def _ensure_aware(dt: Optional[datetime]) -> Optional[datetime]:
            if dt is None:
                return None
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt

        since = _ensure_aware(since)
        until = _ensure_aware(until)

        include = ["ids", "documents", "metadatas"]
        if include_embeddings:
            include.append("embeddings")

        where = None
        if since is not None or until is not None:
            cond: Dict[str, str] = {}
            if since is not None:
                cond["$gte"] = since.astimezone(timezone.utc).isoformat()
            if until is not None:
                cond["$lt"] = until.astimezone(timezone.utc).isoformat()
            if cond:
                where = {"ingested_at": cond}

        try:
            if where is not None:
                results = self.raw_collection.get(where=where, include=include)
            else:
                results = self.raw_collection.get(include=include)
        except Exception:
            results = self.raw_collection.get(include=include)

        if since is None and until is None:
            return results

        ids = results.get("ids") or []
        metadatas = results.get("metadatas") or []
        documents = results.get("documents") or []
        embeddings = results.get("embeddings") if include_embeddings else None

        keep_indices: List[int] = []
        for idx, metadata in enumerate(metadatas):
            timestamp_str = (metadata or {}).get("ingested_at")
            if not timestamp_str:
                continue

            try:
                timestamp = datetime.fromisoformat(timestamp_str)
            except Exception:
                continue

            if since is not None and timestamp < since:
                continue
            if until is not None and timestamp >= until:
                continue

            keep_indices.append(idx)

        filtered: Dict[str, Any] = {
            "ids": [ids[i] for i in keep_indices],
            "documents": [documents[i] for i in keep_indices],
            "metadatas": [metadatas[i] for i in keep_indices],
        }

        if include_embeddings:
            filtered["embeddings"] = (
                [embeddings[i] for i in keep_indices] if embeddings else []
            )

        return filtered

    # -----------------------------
    # DISTINCT STORIES (CACHE)
    # -----------------------------
    def _delete_distinct_for_day(self, day_key: str) -> int:
        """
        Delete all cached distinct stories for a given day_key.
        Best-effort server-side filter, with local fallback.
        """
        ids_to_delete: List[str] = []

        try:
            existing = self.distinct_collection.get(
                where={"day_key": day_key},
                include=["ids"],
            )
            ids_to_delete = existing.get("ids") or []
        except Exception:
            existing = self.distinct_collection.get(include=["ids", "metadatas"])
            ids = existing.get("ids") or []
            metadatas = existing.get("metadatas") or []
            for idx, metadata in enumerate(metadatas):
                if (metadata or {}).get("day_key") == day_key:
                    ids_to_delete.append(ids[idx])

        if ids_to_delete:
            self.distinct_collection.delete(ids=ids_to_delete)

        return len(ids_to_delete)

    def cache_distinct_stories(
        self,
        day_key: str,
        stories: List[Dict[str, Any]],
        run_metadata: Dict[str, Any],
        replace: bool = True,
    ) -> int:
        """
        Cache a day's distinct stories into `ai_news_distinct_stories`.

        `stories` is expected to include:
        - rep_article_id: str
        - member_article_ids: List[str]
        - headline: str
        - document: str
        - metadata: dict (representative metadata)
        - embedding: List[float] (representative embedding)
        """
        existing_member_to_id: Dict[str, str] = {}

        # If we're replacing, build a map from previous cluster members -> previous
        # distinct_story_id so reclustering doesn't orphan story_state IDs
        # (best-effort stability across runs).
        if replace:
            try:
                prev = self.distinct_collection.get(
                    where={"day_key": day_key},
                    include=["ids", "metadatas"],
                )
                prev_ids = prev.get("ids") or []
                prev_metas = prev.get("metadatas") or []
                for prev_id, meta in zip(prev_ids, prev_metas):
                    json_str = (meta or {}).get("member_article_ids_json")
                    if not json_str:
                        continue
                    try:
                        members = list(json.loads(json_str))
                    except Exception:
                        continue
                    for member_id in members:
                        if member_id:
                            existing_member_to_id[str(member_id)] = str(prev_id)
            except Exception:
                pass

        if replace:
            self._delete_distinct_for_day(day_key)

        added = 0
        created_at = datetime.now(timezone.utc).isoformat()

        for story in stories:
            rep_article_id = story.get("rep_article_id")
            member_article_ids = story.get("member_article_ids") or []

            # Stable IDs across reclustering:
            # - Prefer reusing the previous distinct_story_id if any member existed
            #   before.
            # - Otherwise, derive a new ID from (day_key, rep_article_id).
            reused_id = None
            for member_id in sorted(member_article_ids):
                reused_id = existing_member_to_id.get(str(member_id))
                if reused_id:
                    break

            if reused_id:
                distinct_story_id = reused_id
            else:
                distinct_id_src = f"{day_key}|{rep_article_id}"
                distinct_story_id = hashlib.sha256(distinct_id_src.encode()).hexdigest()

            rep_metadata = story.get("metadata") or {}
            rep_source = rep_metadata.get("source") or ""
            sources = sorted({rep_source} | set(story.get("sources") or []))
            cluster_size = int(
                story.get("cluster_size") or len(member_article_ids) or 1
            )
            rep_ingested_at = (
                rep_metadata.get("ingested_at")
                or rep_metadata.get("timestamp_added")
                or ""
            )

            metadata: Dict[str, Any] = {
                "day_key": day_key,
                "rep_article_id": rep_article_id,
                "cluster_size": cluster_size,
                "member_article_ids_json": json.dumps(member_article_ids),
                "headline": story.get("headline") or rep_metadata.get("title") or "",
                "sources_csv": ",".join([s for s in sources if s]),
                "rep_url": rep_metadata.get("url") or "",
                "rep_source": rep_source,
                "rep_published_at": rep_metadata.get("published_at") or "",
                "rep_ingested_at": rep_ingested_at,
                "created_at": created_at,
            }

            # Keep minimal clustering context for evaluation/debugging.
            for k in [
                "since_utc",
                "until_utc",
                "selection_basis",
                "embedding_model",
                "cluster_algo",
                "eps",
                "min_samples",
                "metric",
                "raw_count",
                "distinct_count",
            ]:
                if k in run_metadata:
                    metadata[k] = run_metadata[k]

            embedding = story.get("embedding") or []
            document = story.get("document") or ""

            self.distinct_collection.add(
                ids=[distinct_story_id],
                embeddings=[embedding],
                documents=[document],
                metadatas=[metadata],
            )
            added += 1

        return added

    def get_distinct_stories(
        self,
        day_key: str,
    ) -> List[Dict[str, Any]]:
        """
        Fetch cached distinct stories for a given day_key.
        Returns the story dict shape expected by CLIAssistant.
        """
        try:
            results = self.distinct_collection.get(
                where={"day_key": day_key},
                include=["documents", "metadatas"],
            )
        except Exception:
            results = self.distinct_collection.get(include=["documents", "metadatas"])

        ids = results.get("ids") or []
        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or []

        stories: List[Dict[str, Any]] = []
        for idx, metadata in enumerate(metadatas):
            if (metadata or {}).get("day_key") != day_key:
                continue

            headline = (
                (metadata or {}).get("headline")
                or (metadata or {}).get("rep_title")
                or ""
            )
            stories.append(
                {
                    "headline": headline,
                    "document": documents[idx] if idx < len(documents) else "",
                    "metadata": metadata,
                    "cluster_size": int((metadata or {}).get("cluster_size") or 1),
                    "distinct_story_id": ids[idx] if idx < len(ids) else None,
                }
            )

        def _ts(story: Dict[str, Any]) -> str:
            md = story.get("metadata") or {}
            return md.get("rep_ingested_at") or md.get("created_at") or ""

        stories.sort(key=_ts, reverse=True)
        return stories

    def get_distinct_member_article_ids_map(
        self, distinct_story_ids: List[str]
    ) -> Dict[str, List[str]]:
        """
        Returns a map of distinct_story_id -> member_article_ids (may be empty if
        not found).
        """
        if not distinct_story_ids:
            return {}

        try:
            results = self.distinct_collection.get(
                ids=distinct_story_ids,
                include=["metadatas"],
            )
        except Exception:
            return {}

        ids = results.get("ids") or []
        metadatas = results.get("metadatas") or []

        out: Dict[str, List[str]] = {}
        for story_id, metadata in zip(ids, metadatas):
            members: List[str] = []
            json_str = (metadata or {}).get("member_article_ids_json")
            if json_str:
                try:
                    members = list(json.loads(json_str))
                except Exception:
                    members = []
            out[story_id] = members

        return out

    def delete_raw_articles_by_ids(self, article_ids: List[str]) -> int:
        if not article_ids:
            return 0
        try:
            self.raw_collection.delete(ids=article_ids)
        except Exception:
            return 0
        return len(article_ids)

    def delete_distinct_stories_by_ids(self, distinct_story_ids: List[str]) -> int:
        if not distinct_story_ids:
            return 0
        try:
            self.distinct_collection.delete(ids=distinct_story_ids)
        except Exception:
            return 0
        return len(distinct_story_ids)

    # -----------------------------
    # QUERY IMPORTANT
    # -----------------------------

    def query_important(self, query_text: str, top_k: int = 3):
        query_embedding = self.embedding_model.encode(query_text).tolist()

        return self.important_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "embeddings", "distances"],
        )

    # -----------------------------
    # CLEANUP OLD ARTICLES
    # -----------------------------
    def cleanup_expired(self, days: int = 15) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        results = self.recent_collection.get(include=["ids", "metadatas"])

        ids_to_delete = []

        metadatas = results.get("metadatas") or []
        ids = results.get("ids") or []

        for idx, metadata in enumerate(metadatas):
            timestamp_str = (metadata or {}).get("timestamp_added")
            if not timestamp_str:
                continue

            timestamp = datetime.fromisoformat(timestamp_str)

            if timestamp < cutoff:
                ids_to_delete.append(ids[idx])

        if ids_to_delete:
            self.recent_collection.delete(ids=ids_to_delete)

        return len(ids_to_delete)

    # -----------------------------
    # CHECK EXISTENCE
    # -----------------------------
    def _exists(self, article_id: str) -> bool:
        result = self.recent_collection.get(ids=[article_id])
        return bool(result["ids"])
