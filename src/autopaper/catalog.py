from __future__ import annotations

from autopaper.errors import AutoPaperError, NotFoundError
from autopaper.models import Author, Topic
from autopaper.sources import available_source_names
from autopaper.storage.db import KnowledgeBase
from autopaper.textutil import collapse_ws, normalize_match, slugify


def format_author(author: Author) -> str:
    """主名在前，别名放在全角括号里，用 / 隔开。"""
    aliases = [
        alias
        for alias in author.aliases
        if normalize_match(alias) and normalize_match(alias) != normalize_match(author.name)
    ]
    if not aliases:
        return author.name
    return f"{author.name}（{' / '.join(aliases)}）"


def split_choices(raw: str | None) -> list[str]:
    if not raw:
        return []
    parts = raw.replace("；", ";").split(";")
    return [collapse_ws(part) for part in parts if collapse_ws(part)]


def sorted_authors(kb: KnowledgeBase) -> list[Author]:
    return sorted(kb.list_authors(), key=lambda author: author.name.casefold())


def sorted_topics(kb: KnowledgeBase) -> list[Topic]:
    return sorted(kb.list_topics(), key=lambda topic: topic.name.casefold())


def _by_index(items: list, token: str):
    if token.isdigit():
        index = int(token)
        if 1 <= index <= len(items):
            return items[index - 1]
    return None


def _match_author_name(authors: list[Author], token: str) -> Author | None:
    folded = normalize_match(token)
    for author in authors:
        labels = [author.name, author.id, format_author(author), *author.aliases]
        if folded in {normalize_match(label) for label in labels}:
            return author
    return None


def _match_topic_name(topics: list[Topic], token: str) -> Topic | None:
    folded = normalize_match(token)
    slug = slugify(token)
    for topic in topics:
        if folded in {normalize_match(topic.name), normalize_match(topic.id)} or topic.id == slug:
            return topic
    return None


def resolve_authors(kb: KnowledgeBase, raw: str) -> list[Author]:
    original = sorted_authors(kb)
    authors = list(original)
    chosen: list[Author] = []
    seen: set[str] = set()
    for token in split_choices(raw):
        author = _by_index(original, token)
        if author is None:
            author = _match_author_name(authors, token)
        if author is None:
            author = kb.create_author(token)
            authors.append(author)
        if author.id not in seen:
            chosen.append(author)
            seen.add(author.id)
    if not chosen:
        raise AutoPaperError("请至少指定一位作者")
    return chosen


def ensure_topic(kb: KnowledgeBase, name: str) -> Topic:
    cleaned = collapse_ws(name)
    if not cleaned:
        raise AutoPaperError("主题名为空")
    existing = _match_topic_name(kb.list_topics(), cleaned)
    if existing is not None:
        return existing
    topic_id = slugify(cleaned)
    taken = {topic.id for topic in kb.list_topics()}
    suffix = 2
    while topic_id in taken:
        topic_id = f"{slugify(cleaned)}-{suffix}"
        suffix += 1
    topic = Topic(
        id=topic_id,
        name=cleaned,
        query=cleaned,
        sources=available_source_names(),
        schedule="manual",
    )
    return kb.save_topic(topic)


def resolve_topics(kb: KnowledgeBase, raw: str) -> list[Topic]:
    original = sorted_topics(kb)
    topics = list(original)
    chosen: list[Topic] = []
    seen: set[str] = set()
    for token in split_choices(raw):
        topic = _by_index(original, token)
        if topic is None:
            topic = _match_topic_name(topics, token)
        if topic is None:
            topic = ensure_topic(kb, token)
            topics.append(topic)
        if topic.id not in seen:
            chosen.append(topic)
            seen.add(topic.id)
    if not chosen:
        raise AutoPaperError("请至少指定一个研究主题")
    return chosen


def require_existing_author(kb: KnowledgeBase, token: str) -> Author:
    authors = sorted_authors(kb)
    author = _by_index(authors, token)
    if author is None:
        author = _match_author_name(authors, token)
    if author is None:
        author = kb.find_author(token)
    if author is None:
        raise NotFoundError(f"没有找到作者：{token}")
    return author
