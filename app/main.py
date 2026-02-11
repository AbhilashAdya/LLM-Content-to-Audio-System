from app.data_ingestion.rss_ingestor import fetch_articles


def main():
    articles = fetch_articles()
    print(f"Fetched {len(articles)} articles")

    for article in articles[:3]:
        print(f"- [{article.source}] {article.title}")


if __name__ == "__main__":
    main()
