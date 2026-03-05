import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict


@dataclass(frozen=True)
class ImportantNotifyStatus:
    should_notify: bool
    current_count: int
    limit: int
    next_notify_at: int


class StoryStateStore:
    """
    Stores per-user state for distinct stories:
    - unheard: in backlog
    - heard: consumed (eligible for TTL cleanup later)
    - important: saved (never auto-deleted)
    """

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            project_root = Path(__file__).resolve().parents[2]
            db_path = project_root / "assistant_state.sqlite3"
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def init(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS story_state (
                    distinct_story_id TEXT PRIMARY KEY,
                    day_key TEXT NOT NULL,
                    status TEXT NOT NULL
                        CHECK (status IN ('unheard','heard','important')),
                    first_seen_at TEXT,
                    last_presented_at TEXT,
                    heard_at TEXT,
                    important_at TEXT
                )
                """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """)
            # Default: notify once important count reaches the limit.
            conn.execute(
                "INSERT OR IGNORE INTO settings(key, value) VALUES(?, ?)",
                ("important_next_notify_at", "20"),
            )

    @staticmethod
    def _now_utc_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def ensure_unheard(self, distinct_story_id: str, day_key: str) -> None:
        now = self._now_utc_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO story_state(
                    distinct_story_id, day_key, status, first_seen_at, last_presented_at
                ) VALUES(?, ?, 'unheard', ?, ?)
                """,
                (distinct_story_id, day_key, now, now),
            )

    def mark_presented(self, distinct_story_id: str) -> None:
        now = self._now_utc_iso()
        with self._connect() as conn:
            conn.execute(
                "UPDATE story_state SET last_presented_at=? WHERE distinct_story_id=?",
                (now, distinct_story_id),
            )

    def mark_heard(self, distinct_story_id: str) -> None:
        now = self._now_utc_iso()
        with self._connect() as conn:
            # Do not downgrade important.
            conn.execute(
                """
                UPDATE story_state
                SET status='heard',
                    heard_at=COALESCE(heard_at, ?)
                WHERE distinct_story_id=?
                  AND status != 'important'
                """,
                (now, distinct_story_id),
            )

    def mark_important(self, distinct_story_id: str) -> None:
        now = self._now_utc_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE story_state
                SET status='important',
                    important_at=COALESCE(important_at, ?)
                WHERE distinct_story_id=?
                """,
                (now, distinct_story_id),
            )

    def list_unheard_ids(self, limit: int = 200) -> List[Tuple[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT distinct_story_id, day_key
                FROM story_state
                WHERE status='unheard'
                ORDER BY day_key ASC, first_seen_at ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [(r["distinct_story_id"], r["day_key"]) for r in rows]

    def count_important(self) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM story_state WHERE status='important'"
            ).fetchone()
        return int(row["c"] if row else 0)

    def get_important_notify_status(self, limit: int = 20) -> ImportantNotifyStatus:
        current = self.count_important()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM settings WHERE key='important_next_notify_at'"
            ).fetchone()
            next_notify_at = int(row["value"]) if row else limit

        should_notify = current >= limit and current >= next_notify_at
        return ImportantNotifyStatus(
            should_notify=should_notify,
            current_count=current,
            limit=limit,
            next_notify_at=next_notify_at,
        )

    def ack_important_notification(self, current_count: int, step: int = 15) -> None:
        """
        Move the notification threshold forward by `step` items from the current count.
        """
        next_notify_at = current_count + step
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO settings(key, value) VALUES(?, ?)",
                ("important_next_notify_at", str(next_notify_at)),
            )

    def list_oldest_important_ids(self, limit: int = 15) -> List[str]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT distinct_story_id
                FROM story_state
                WHERE status='important'
                ORDER BY important_at ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [r["distinct_story_id"] for r in rows]

    def list_expired_ids(
        self,
        heard_ttl_days: int = 15,
        backlog_ttl_days: int = 30,
    ) -> Dict[str, List[str]]:
        """
        Returns a dict with keys:
        - heard: distinct_story_ids eligible for deletion (heard_at older than TTL)
        - unheard: distinct_story_ids eligible for deletion (day_key older than TTL)

        Important stories are never auto-deleted here.
        """
        now_utc = datetime.now(timezone.utc)
        heard_cutoff = (now_utc - timedelta(days=heard_ttl_days)).isoformat()

        today_local = datetime.now().astimezone().date()
        cutoff_day = today_local - timedelta(days=backlog_ttl_days)
        cutoff_day_key = cutoff_day.strftime("%Y-%m-%d")

        with self._connect() as conn:
            heard_rows = conn.execute(
                """
                SELECT distinct_story_id
                FROM story_state
                WHERE status='heard'
                  AND heard_at IS NOT NULL
                  AND heard_at < ?
                """,
                (heard_cutoff,),
            ).fetchall()

            unheard_rows = conn.execute(
                """
                SELECT distinct_story_id
                FROM story_state
                WHERE status='unheard'
                  AND day_key < ?
                """,
                (cutoff_day_key,),
            ).fetchall()

        return {
            "heard": [r["distinct_story_id"] for r in heard_rows],
            "unheard": [r["distinct_story_id"] for r in unheard_rows],
        }

    def delete_story_states(self, distinct_story_ids: Iterable[str]) -> int:
        ids = list(distinct_story_ids)
        if not ids:
            return 0

        with self._connect() as conn:
            cur = conn.executemany(
                "DELETE FROM story_state WHERE distinct_story_id=?",
                [(i,) for i in ids],
            )
        return cur.rowcount if cur is not None else 0
