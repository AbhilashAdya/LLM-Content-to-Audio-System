# from app.data_ingestion.rss_ingestor import fetch_articles


# def main():
#     articles = fetch_articles()
#     print(f"Fetched {len(articles)} articles")

#     for article in articles[:3]:
#         print(f"- [{article.source}] {article.title}")


# if __name__ == "__main__":
#     main()

from app.data_ingestion.rss_ingestor import fetch_articles
from app.memory.vector_store import VectorStore


def main():
    print("Fetching latest articles...")
    articles = fetch_articles()
    print(f"Fetched {len(articles)} articles")

    print("Initializing vector store...")
    store = VectorStore()

    print("Cleaning up expired articles (15 days)...")
    store.cleanup_expired(days=15)

    print("Adding new articles to vector store...")
    store.add_articles(articles)

    print("\nTesting semantic retrieval...\n")

    query = "latest multimodal AI research developments"

    results = store.query_recent(query, top_k=3)

    if results["documents"]:
        for i in range(len(results["documents"][0])):
            metadata = results["metadatas"][0][i]
            print(f"\nResult {i+1}")
            print(f"Title: {metadata['title']}")
            print(f"Source: {metadata['source']}")
            print(f"URL: {metadata['url']}")
            print(f"Published At: {metadata['published_at']}")
    else:
        print("No relevant articles found.")


if __name__ == "__main__":
    main()
