from __future__ import annotations

from dataclasses import dataclass
from typing import Set

from app.memory.story_state_store import StoryStateStore
from app.memory.vector_store import VectorStore


@dataclass(frozen=True)
class CleanupResult:
    expired_heard: int
    expired_unheard: int
    distinct_deleted: int
    raw_deleted: int
    state_deleted: int


def cleanup_expired_story_data(
    store: VectorStore,
    state: StoryStateStore,
    heard_ttl_days: int = 15,
    backlog_ttl_days: int = 30,
) -> CleanupResult:
    """
    Deletes expired non-important stories:
    - heard stories older than `heard_ttl_days`
    - unheard stories (backlog) older than `backlog_ttl_days` based on day_key

    Deletion includes:
    - cached distinct story records
    - underlying raw_fetched member articles (best-effort)
    - story_state rows
    """
    state.init()
    expired = state.list_expired_ids(
        heard_ttl_days=heard_ttl_days,
        backlog_ttl_days=backlog_ttl_days,
    )

    heard_ids = list(expired.get("heard") or [])
    unheard_ids = list(expired.get("unheard") or [])
    all_distinct_ids = heard_ids + unheard_ids

    if not all_distinct_ids:
        return CleanupResult(
            expired_heard=0,
            expired_unheard=0,
            distinct_deleted=0,
            raw_deleted=0,
            state_deleted=0,
        )

    member_map = store.get_distinct_member_article_ids_map(all_distinct_ids)

    raw_ids: Set[str] = set()
    for member_ids in member_map.values():
        raw_ids.update(member_ids or [])

    raw_deleted = store.delete_raw_articles_by_ids(sorted(raw_ids))
    distinct_deleted = store.delete_distinct_stories_by_ids(all_distinct_ids)
    state_deleted = state.delete_story_states(all_distinct_ids)

    return CleanupResult(
        expired_heard=len(heard_ids),
        expired_unheard=len(unheard_ids),
        distinct_deleted=distinct_deleted,
        raw_deleted=raw_deleted,
        state_deleted=state_deleted,
    )
