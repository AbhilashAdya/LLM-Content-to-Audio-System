from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Article:
    title: str
    summary: str
    url: str
    source: str
    published_at: Optional[datetime]
