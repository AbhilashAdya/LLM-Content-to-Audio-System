from datetime import datetime
from typing import List, Optional

import feedparser
from dateutil import parser as date_parser

from app.data_ingestion.schemas import Article


RSS_SOURCES = [
    {
        "name": "VentureBeat AI",
        "url": "https://venturebeat.com/category/ai/feed/",
    },
    {
        "name": "MIT Technology Review AI",
        "url": "https://www.technologyreview.com/topic/artificial-intelligence/feed/",
    },
    {
        "name": "Towards Data Science",
        "url": "https://towardsdatascience.com/feed",
    },
]



def parse_published_date(entry) -> Optional[datetime]:
    """
    Parse the published date from an RSS entry.
    Returns None if unavailable or unparsable.
    """
    if hasattr(entry, "published"):
        try:
            return date_parser.parse(entry.published)
        except Exception:
            return None
    return None


def fetch_articles() -> List[Article]:
    """
    Fetch articles from all configured RSS sources
    and normalize them into Article objects.
    """
    articles: List[Article] = []

    for source in RSS_SOURCES:
        feed = feedparser.parse(source["url"])

        for entry in feed.entries:
            article = Article(
                title=entry.get("title", "").strip(),
                summary=entry.get("summary", "").strip(),
                url=entry.get("link", ""),
                source=source["name"],
                published_at=parse_published_date(entry),
            )
            articles.append(article)

    return articles
