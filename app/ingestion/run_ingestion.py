from app.data_ingestion.rss_ingestor import fetch_articles
from app.memory.vector_store import VectorStore
from app.memory.story_state_store import StoryStateStore
from app.maintenance.cleanup import cleanup_expired_story_data


def run_ingestion(
    store=None,
    days_to_keep: int = 15,
    heard_ttl_days: int = 15,
    backlog_ttl_days: int = 30,
):
    """
    Fetch RSS articles, remove expired ones, then add new ones.

    If `store` is not provided, this function will create a VectorStore so that
    running this file directly still performs ingestion.
    """
    if store is None:
        store = VectorStore()

    print("Updating AI news database...")

    articles = fetch_articles()
    print(f"Fetched {len(articles)} articles.")

    print("Adding new articles to raw_fetched...")
    raw_added = store.add_raw_articles(articles)
    print(f"Added {raw_added} new raw articles.")

    # Legacy cache (ai_news_recent) is kept in VectorStore for backwards compatibility,
    # but ingestion is now raw_fetched -> distinct_stories, so we don't populate it.
    # If you want to re-enable it later, uncomment this block.
    #
    # print(f"Cleaning up legacy recent cache older than {days_to_keep} days...")
    # deleted = store.cleanup_expired(days=days_to_keep)
    # print(f"Deleted {deleted} expired articles from legacy recent cache.")
    #
    # print("Adding new articles to legacy recent cache...")
    # added = store.add_articles(articles)
    # print(f"Added {added} new articles to legacy recent cache.")

    print(
        "Cleaning up expired story data "
        f"(heard>{heard_ttl_days}d, backlog>{backlog_ttl_days}d)..."
    )
    state = StoryStateStore()
    result = cleanup_expired_story_data(
        store=store,
        state=state,
        heard_ttl_days=heard_ttl_days,
        backlog_ttl_days=backlog_ttl_days,
    )
    print(
        "Expired cleanup: "
        f"heard={result.expired_heard}, unheard={result.expired_unheard}, "
        f"distinct_deleted={result.distinct_deleted}, "
        f"raw_deleted={result.raw_deleted}, "
        f"state_deleted={result.state_deleted}"
    )

    print("Database updated.\n")
    return store


if __name__ == "__main__":
    run_ingestion()
