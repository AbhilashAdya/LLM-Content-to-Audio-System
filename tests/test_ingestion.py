from datetime import datetime, timedelta, timezone
from app.data_ingestion.schemas import Article


def test_article_schema_with_published_date():
    published_time = datetime.now(timezone.utc)

    article = Article(
        title="Test",
        summary="Test summary",
        url="https://example.com",
        source="Test Source",
        published_at=published_time,
    )

    assert isinstance(article.title, str)
    assert isinstance(article.source, str)
    assert isinstance(article.published_at, datetime)


def test_article_recency_comparison():
    now = datetime.now(timezone.utc)

    older_article = Article(
        title="Old",
        summary="Old summary",
        url="https://old.com",
        source="Test Source",
        published_at= now - timedelta(days=1),
    )

    newer_article = Article(
        title="New",
        summary="New summary",
        url="https://new.com",
        source="Test Source",
        published_at=now,
    )

    assert newer_article.published_at > older_article.published_at

def test_article_without_published_date():
    article = Article(
        title="No date",
        summary="Missing date",
        url="https://nodate.com",
        source="Test Source",
        published_at=None,
    )

    assert article.published_at is None

