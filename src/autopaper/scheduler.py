from __future__ import annotations

from datetime import datetime, timedelta, timezone

from autopaper.models import Topic


def is_due(topic: Topic, now: datetime | None = None) -> bool:
    """手动主题不会到期。每天按 20 小时、每周按 6.5 天判断，避免卡在整点。"""
    if topic.schedule not in {"daily", "weekly"}:
        return False
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    if topic.last_run_at is None:
        return True
    last = topic.last_run_at
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    elapsed = current - last
    if topic.schedule == "daily":
        return elapsed >= timedelta(hours=20)
    return elapsed >= timedelta(days=6, hours=12)


def due_topics(topics: list[Topic], now: datetime | None = None, *, include_all: bool = False) -> list[Topic]:
    if include_all:
        return list(topics)
    return [topic for topic in topics if is_due(topic, now)]
