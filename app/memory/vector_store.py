import hashlib
from datetime import datetime, timedelta, timezone
from typing import List

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from app.data_ingestion.schemas import Article


class VectorStore:
    def __init__(self):
        # Embedding model (can change later)
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Persistent Chroma client
        self.client = chromadb.Client(
            Settings(
                persist_directory="chroma_db",
                anonymized_telemetry=False,
            )
        )

        # Two collections: recent + important
        self.recent_collection = self.client.get_or_create_collection(
            name="ai_news_recent"
        )

        self.important_collection = self.client.get_or_create_collection(
            name="ai_news_important"
        )

    # -----------------------------
    # ID GENERATION
    # -----------------------------
    @staticmethod
    def generate_article_id(article: Article) -> str:
        unique_string = f"{article.title}|" f"{article.url}|" f"{article.published_at}"
        return hashlib.sha256(unique_string.encode()).hexdigest()

    # -----------------------------
    # ADD ARTICLES (RECENT)
    # -----------------------------
    def add_articles(self, articles: List[Article]):
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

    # -----------------------------
    # MOVE TO IMPORTANT
    # -----------------------------
    def mark_as_important(self, article_id: str):
        result = self.recent_collection.get(ids=[article_id])

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
        )

    # -----------------------------
    # QUERY IMPORTANT
    # -----------------------------
    def query_important(self, query_text: str, top_k: int = 3):
        query_embedding = self.embedding_model.encode(query_text).tolist()

        return self.important_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

    # -----------------------------
    # CLEANUP OLD ARTICLES
    # -----------------------------
    def cleanup_expired(self, days: int = 15):
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        results = self.recent_collection.get()

        ids_to_delete = []

        for idx, metadata in enumerate(results["metadatas"]):
            timestamp_str = metadata.get("timestamp_added")
            if not timestamp_str:
                continue

            timestamp = timestamp = datetime.fromisoformat(timestamp_str)

            if timestamp < cutoff:
                ids_to_delete.append(results["ids"][idx])

        if ids_to_delete:
            self.recent_collection.delete(ids=ids_to_delete)

    # -----------------------------
    # CHECK EXISTENCE
    # -----------------------------
    def _exists(self, article_id: str) -> bool:
        result = self.recent_collection.get(ids=[article_id])
        return bool(result["ids"])
