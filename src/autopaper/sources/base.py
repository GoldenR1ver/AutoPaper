from __future__ import annotations

from typing import Protocol

from autopaper.models import Paper


class LiteratureSource(Protocol):
    name: str

    def search(
        self,
        query: str,
        *,
        limit: int,
        since_year: int | None = None,
        sort: str = "relevance",
    ) -> list[Paper]:
        """按主题检索，返回统一的论文记录。"""
