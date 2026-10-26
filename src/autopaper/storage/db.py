from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from autopaper.errors import AmbiguousError, AutoPaperError, NotFoundError
from autopaper.models import (
    Author,
    Paper,
    Relation,
    ResearcherProfile,
    ScoredPaper,
    Topic,
    make_key,
    merge_papers,
)
from autopaper.textutil import collapse_ws, like_pattern, normalize_match, normalize_title, slugify

SCHEMA = """
CREATE TABLE IF NOT EXISTS topics (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  query TEXT NOT NULL,
  keywords TEXT NOT NULL DEFAULT '[]',
  exclude_keywords TEXT NOT NULL DEFAULT '[]',
  domain_tags TEXT NOT NULL DEFAULT '[]',
  domain_mode TEXT NOT NULL DEFAULT 'soft',
  sources TEXT NOT NULL,
  schedule TEXT NOT NULL DEFAULT 'manual',
  sort TEXT NOT NULL DEFAULT 'relevance',
  max_per_source INTEGER NOT NULL DEFAULT 20,
  since_year INTEGER,
  min_score REAL NOT NULL DEFAULT 0,
  llm_top_n INTEGER NOT NULL DEFAULT 20,
  keyword_mode TEXT NOT NULL DEFAULT 'any',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  last_run_at TEXT
);

CREATE TABLE IF NOT EXISTS papers (
  key TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  title_norm TEXT NOT NULL,
  abstract TEXT NOT NULL DEFAULT '',
  authors TEXT NOT NULL DEFAULT '[]',
  year INTEGER,
  doi TEXT,
  arxiv_id TEXT,
  pubmed_id TEXT,
  s2_id TEXT,
  openalex_id TEXT,
  url TEXT,
  pdf_url TEXT,
  venue TEXT,
  citation_count INTEGER,
  categories TEXT NOT NULL DEFAULT '[]',
  sources TEXT NOT NULL DEFAULT '[]',
  search_text TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_papers_doi ON papers(doi) WHERE doi IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_papers_title_norm ON papers(title_norm);
CREATE INDEX IF NOT EXISTS idx_papers_arxiv ON papers(arxiv_id);
CREATE INDEX IF NOT EXISTS idx_papers_pmid ON papers(pubmed_id);
CREATE INDEX IF NOT EXISTS idx_papers_year ON papers(year);

CREATE TABLE IF NOT EXISTS topic_papers (
  topic_id TEXT NOT NULL,
  paper_key TEXT NOT NULL,
  heuristic_score REAL,
  llm_score REAL,
  score REAL NOT NULL,
  reason TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL DEFAULT 'candidate',
  collected_at TEXT NOT NULL,
  PRIMARY KEY (topic_id, paper_key),
  FOREIGN KEY (topic_id) REFERENCES topics(id) ON DELETE CASCADE,
  FOREIGN KEY (paper_key) REFERENCES papers(key) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS readings (
  paper_key TEXT PRIMARY KEY,
  payload TEXT NOT NULL,
  markdown TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY (paper_key) REFERENCES papers(key) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS researchers (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  orcid TEXT,
  affiliation TEXT,
  openalex_id TEXT,
  s2_id TEXT,
  h_index INTEGER,
  cited_by_count INTEGER,
  works_count INTEGER,
  concepts TEXT NOT NULL DEFAULT '[]',
  summary TEXT NOT NULL DEFAULT '',
  directions TEXT NOT NULL DEFAULT '[]',
  report_path TEXT,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS researcher_papers (
  researcher_id TEXT NOT NULL,
  paper_key TEXT NOT NULL,
  PRIMARY KEY (researcher_id, paper_key),
  FOREIGN KEY (researcher_id) REFERENCES researchers(id) ON DELETE CASCADE,
  FOREIGN KEY (paper_key) REFERENCES papers(key) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS researcher_relations (
  researcher_id TEXT NOT NULL,
  other_name TEXT NOT NULL,
  other_id TEXT,
  relation TEXT NOT NULL,
  weight REAL NOT NULL DEFAULT 1,
  detail TEXT NOT NULL DEFAULT '',
  PRIMARY KEY (researcher_id, other_name, relation),
  FOREIGN KEY (researcher_id) REFERENCES researchers(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  topic_id TEXT,
  started_at TEXT NOT NULL,
  finished_at TEXT,
  status TEXT NOT NULL,
  stats TEXT NOT NULL DEFAULT '{}'
);
"""

SCHEMA_V2 = """
ALTER TABLE papers ADD COLUMN local_path TEXT;

CREATE TABLE IF NOT EXISTS authors (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  aliases TEXT NOT NULL DEFAULT '[]',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS author_names (
  name_norm TEXT PRIMARY KEY,
  author_id TEXT NOT NULL,
  name TEXT NOT NULL,
  is_primary INTEGER NOT NULL DEFAULT 0,
  FOREIGN KEY (author_id) REFERENCES authors(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS author_papers (
  author_id TEXT NOT NULL,
  paper_key TEXT NOT NULL,
  PRIMARY KEY (author_id, paper_key),
  FOREIGN KEY (author_id) REFERENCES authors(id) ON DELETE CASCADE,
  FOREIGN KEY (paper_key) REFERENCES papers(key) ON DELETE CASCADE
);
"""

SCHEMA_V3 = """
CREATE TABLE IF NOT EXISTS highlights (
  id TEXT PRIMARY KEY,
  scope TEXT NOT NULL,
  scope_id TEXT NOT NULL,
  scope_label TEXT NOT NULL DEFAULT '',
  kind TEXT NOT NULL,
  color TEXT NOT NULL DEFAULT '',
  paragraph TEXT NOT NULL,
  selected_text TEXT NOT NULL DEFAULT '',
  comment TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_highlights_scope ON highlights(scope, scope_id);
"""

SCHEMA_V4 = """
CREATE TABLE IF NOT EXISTS topic_parents (
  child_id TEXT NOT NULL,
  parent_id TEXT NOT NULL,
  PRIMARY KEY (child_id, parent_id),
  FOREIGN KEY (child_id) REFERENCES topics(id) ON DELETE CASCADE,
  FOREIGN KEY (parent_id) REFERENCES topics(id) ON DELETE CASCADE,
  CHECK (child_id != parent_id)
);

CREATE TABLE IF NOT EXISTS topic_related (
  left_id TEXT NOT NULL,
  right_id TEXT NOT NULL,
  PRIMARY KEY (left_id, right_id),
  FOREIGN KEY (left_id) REFERENCES topics(id) ON DELETE CASCADE,
  FOREIGN KEY (right_id) REFERENCES topics(id) ON DELETE CASCADE,
  CHECK (left_id < right_id)
);
"""


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _kept_local_path(papers: list[Paper]) -> str:
    ranked = sorted(
        papers,
        key=lambda paper: (
            0 if paper.local_path and Path(paper.local_path).exists() else 1,
            -len(normalize_title(paper.title)),
        ),
    )
    for paper in ranked:
        if paper.local_path:
            return paper.local_path
    return ""


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


def _highlight_row(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "scope": row["scope"],
        "scope_id": row["scope_id"],
        "scope_label": row["scope_label"],
        "kind": row["kind"],
        "color": row["color"],
        "paragraph": row["paragraph"],
        "selected_text": row["selected_text"],
        "comment": row["comment"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False)


def _loads(value: str | None, default):
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


class KnowledgeBase:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA busy_timeout = 5000")
        self._migrate()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> KnowledgeBase:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _migrate(self) -> None:
        self.conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
        row = self.conn.execute("SELECT MAX(version) AS version FROM schema_version").fetchone()
        version = row["version"] or 0
        if version < 1:
            self.conn.executescript(SCHEMA)
            self.conn.execute("INSERT INTO schema_version(version) VALUES (1)")
            version = 1
        if version < 2:
            self.conn.executescript(SCHEMA_V2)
            self.conn.execute("INSERT INTO schema_version(version) VALUES (2)")
            version = 2
        if version < 3:
            self.conn.executescript(SCHEMA_V3)
            self.conn.execute("INSERT INTO schema_version(version) VALUES (3)")
            version = 3
        if version < 4:
            self.conn.executescript(SCHEMA_V4)
            self.conn.execute("INSERT INTO schema_version(version) VALUES (4)")
        self.conn.commit()

    def save_topic(self, topic: Topic) -> Topic:
        now = utcnow()
        if topic.created_at is None:
            topic.created_at = now
        topic.updated_at = now
        self.conn.execute(
            """
            INSERT INTO topics (
              id, name, query, keywords, exclude_keywords, domain_tags, domain_mode,
              sources, schedule, sort, max_per_source, since_year, min_score,
              llm_top_n, keyword_mode, created_at, updated_at, last_run_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              name = excluded.name,
              query = excluded.query,
              keywords = excluded.keywords,
              exclude_keywords = excluded.exclude_keywords,
              domain_tags = excluded.domain_tags,
              domain_mode = excluded.domain_mode,
              sources = excluded.sources,
              schedule = excluded.schedule,
              sort = excluded.sort,
              max_per_source = excluded.max_per_source,
              since_year = excluded.since_year,
              min_score = excluded.min_score,
              llm_top_n = excluded.llm_top_n,
              keyword_mode = excluded.keyword_mode,
              updated_at = excluded.updated_at,
              last_run_at = excluded.last_run_at,
              created_at = topics.created_at
            """,
            (
                topic.id,
                topic.name,
                topic.query,
                _dumps(topic.keywords),
                _dumps(topic.exclude_keywords),
                _dumps(topic.domain_tags),
                topic.domain_mode,
                _dumps(topic.sources),
                topic.schedule,
                topic.sort,
                topic.max_per_source,
                topic.since_year,
                topic.min_score,
                topic.llm_top_n,
                topic.keyword_mode,
                _iso(topic.created_at),
                _iso(topic.updated_at),
                _iso(topic.last_run_at),
            ),
        )
        self.conn.commit()
        return topic

    def list_topics(self) -> list[Topic]:
        rows = self.conn.execute("SELECT * FROM topics ORDER BY updated_at DESC").fetchall()
        return [self._topic(row) for row in rows]

    def get_topic(self, token: str) -> Topic:
        text = token.strip()
        row = self.conn.execute("SELECT * FROM topics WHERE id = ?", (text,)).fetchone()
        if row:
            return self._topic(row)
        folded = normalize_match(text)
        matches = [
            item
            for item in self.list_topics()
            if normalize_match(item.name) == folded or item.id == slugify(text)
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            names = "、".join(item.id for item in matches)
            raise AmbiguousError(f"主题名称对应多条记录，请使用 id：{names}")
        raise NotFoundError(f"没有找到主题：{token}")

    def delete_topic(self, topic_id: str) -> None:
        self.conn.execute("DELETE FROM topics WHERE id = ?", (topic_id,))
        self.conn.commit()

    def mark_topic_ran(self, topic_id: str, when: datetime | None = None) -> None:
        self.conn.execute(
            "UPDATE topics SET last_run_at = ? WHERE id = ?",
            (_iso(when or utcnow()), topic_id),
        )
        self.conn.commit()

    def clear_topic_papers(self, topic_id: str) -> None:
        self.conn.execute("DELETE FROM topic_papers WHERE topic_id = ?", (topic_id,))
        self.conn.commit()

    def upsert_paper(self, paper: Paper) -> str:
        paper = paper.cleanup()
        if not normalize_title(paper.title):
            raise ValueError("论文标题为空")
        existing = self._find_paper_row(paper)
        now = _iso(utcnow())
        if existing is None:
            key = make_key(paper)
            clash = self.conn.execute("SELECT * FROM papers WHERE key = ?", (key,)).fetchone()
            if clash is not None and self._compatible(clash, paper):
                existing = clash
            elif clash is not None:
                key = self._disambiguate_key(key)
                self._write_paper(key, paper, now, now, insert=True)
                return key
            else:
                self._write_paper(key, paper, now, now, insert=True)
                return key
        merged = merge_papers(self._paper(existing), paper)
        canonical = make_key(merged)
        created_at = existing["created_at"]
        target_key = existing["key"]
        if canonical != target_key:
            occupant = self.conn.execute("SELECT * FROM papers WHERE key = ?", (canonical,)).fetchone()
            if occupant is not None and occupant["key"] != target_key and self._compatible(occupant, merged):
                merged = merge_papers(self._paper(occupant), merged)
                canonical = make_key(merged)
                created_at = min(created_at, occupant["created_at"])
                self._adopt_key(target_key, canonical)
                target_key = canonical
            elif occupant is None:
                self._adopt_key(target_key, canonical)
                target_key = canonical
        self._write_paper(target_key, merged, created_at, now, insert=False)
        if created_at:
            self.conn.execute(
                "UPDATE papers SET created_at = ? WHERE key = ? AND created_at > ?",
                (created_at, target_key, created_at),
            )
            self.conn.commit()
        return target_key

    def _disambiguate_key(self, key: str) -> str:
        suffix = 2
        while self.conn.execute("SELECT 1 FROM papers WHERE key = ?", (f"{key}:{suffix}",)).fetchone():
            suffix += 1
        return f"{key}:{suffix}"

    def _adopt_key(self, old: str, new: str) -> None:
        """把一篇论文的主键改成合并后的规范键，并带走主题、笔记和作者关联。"""
        if not old or not new or old == new:
            return

        def move() -> None:
            self._relink_paper(old, new)
            occupant = self.conn.execute("SELECT 1 FROM papers WHERE key = ?", (new,)).fetchone()
            if occupant is None:
                self.conn.execute("UPDATE papers SET key = ? WHERE key = ?", (new, old))
            else:
                self.conn.execute("DELETE FROM papers WHERE key = ?", (old,))

        self._with_fk_off(move)

    def _relink_paper(self, old: str, new: str) -> None:
        for table, owner in (
            ("topic_papers", "topic_id"),
            ("researcher_papers", "researcher_id"),
            ("author_papers", "author_id"),
        ):
            self.conn.execute(
                f"""
                DELETE FROM {table}
                WHERE paper_key = ?
                  AND {owner} IN (SELECT {owner} FROM {table} WHERE paper_key = ?)
                """,
                (old, new),
            )
            self.conn.execute(
                f"UPDATE {table} SET paper_key = ? WHERE paper_key = ?",
                (new, old),
            )
        self.conn.execute(
            """
            DELETE FROM readings
            WHERE paper_key = ? AND EXISTS (SELECT 1 FROM readings AS kept WHERE kept.paper_key = ?)
            """,
            (old, new),
        )
        self.conn.execute("UPDATE readings SET paper_key = ? WHERE paper_key = ?", (new, old))
        self.conn.execute(
            "UPDATE highlights SET scope_id = ? WHERE scope = 'paper' AND scope_id = ?",
            (new, old),
        )

    def _with_fk_off(self, action) -> None:
        self.conn.commit()
        self.conn.execute("PRAGMA foreign_keys = OFF")
        try:
            action()
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        finally:
            self.conn.execute("PRAGMA foreign_keys = ON")

    def link_topic_paper(self, topic_id: str, scored: ScoredPaper, paper_key: str) -> None:
        self.conn.execute(
            """
            INSERT INTO topic_papers (
              topic_id, paper_key, heuristic_score, llm_score, score, reason, status, collected_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(topic_id, paper_key) DO UPDATE SET
              heuristic_score = excluded.heuristic_score,
              llm_score = excluded.llm_score,
              score = excluded.score,
              reason = excluded.reason,
              status = excluded.status,
              collected_at = excluded.collected_at
            """,
            (
                topic_id,
                paper_key,
                scored.heuristic_score,
                scored.llm_score,
                scored.score,
                scored.reason,
                scored.status,
                _iso(utcnow()),
            ),
        )
        self.conn.commit()

    def descendant_ids(self, topic_id: str) -> list[str]:
        rows = self.conn.execute("SELECT child_id, parent_id FROM topic_parents").fetchall()
        children: dict[str, list[str]] = {}
        for row in rows:
            children.setdefault(row["parent_id"], []).append(row["child_id"])
        found: list[str] = []
        seen: set[str] = set()
        stack = list(children.get(topic_id, []))
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            found.append(current)
            stack.extend(children.get(current, []))
        return found

    def ancestor_ids(self, topic_id: str) -> list[str]:
        rows = self.conn.execute("SELECT child_id, parent_id FROM topic_parents").fetchall()
        parents: dict[str, list[str]] = {}
        for row in rows:
            parents.setdefault(row["child_id"], []).append(row["parent_id"])
        found: list[str] = []
        seen: set[str] = set()
        stack = list(parents.get(topic_id, []))
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            found.append(current)
            stack.extend(parents.get(current, []))
        return found

    def list_topic_papers(
        self,
        topic_id: str,
        *,
        include_held: bool = True,
        min_score: float | None = None,
    ) -> list[ScoredPaper]:
        topic_ids = [topic_id, *self.descendant_ids(topic_id)]
        placeholders = ", ".join("?" for _ in topic_ids)
        sql = f"""
            SELECT papers.*, topic_papers.topic_id AS link_topic_id,
                   topic_papers.heuristic_score, topic_papers.llm_score,
                   topic_papers.score, topic_papers.reason, topic_papers.status
            FROM topic_papers
            JOIN papers ON papers.key = topic_papers.paper_key
            WHERE topic_papers.topic_id IN ({placeholders})
        """
        params: list = list(topic_ids)
        if not include_held:
            sql += " AND topic_papers.status = 'candidate'"
        if min_score is not None:
            sql += " AND topic_papers.score >= ?"
            params.append(min_score)
        rows = self.conn.execute(sql, params).fetchall()
        chosen: dict[str, ScoredPaper] = {}
        direct: dict[str, bool] = {}
        for row in rows:
            scored = self._scored(row)
            is_direct = row["link_topic_id"] == topic_id
            if not is_direct:
                scored.inherited = True
            previous = chosen.get(scored.paper.key)
            if previous is None or (is_direct and not direct[scored.paper.key]) or (
                is_direct == direct.get(scored.paper.key) and scored.score > previous.score
            ):
                chosen[scored.paper.key] = scored
                direct[scored.paper.key] = is_direct
        return sorted(
            chosen.values(),
            key=lambda item: (-item.score, -(item.paper.citation_count or 0), -(item.paper.year or 0), item.paper.title.casefold()),
        )

    def set_topic_parent(self, child_id: str, parent_id: str) -> None:
        child = self.get_topic(child_id)
        parent = self.get_topic(parent_id)
        if child.id == parent.id or parent.id in self.descendant_ids(child.id):
            raise AutoPaperError("这样设置会让主题从属关系绕回自身。")
        self.conn.execute(
            "INSERT OR IGNORE INTO topic_parents (child_id, parent_id) VALUES (?, ?)",
            (child.id, parent.id),
        )
        self.conn.commit()

    def remove_topic_parent(self, child_id: str, parent_id: str) -> None:
        self.conn.execute(
            "DELETE FROM topic_parents WHERE child_id = ? AND parent_id = ?",
            (child_id, parent_id),
        )
        self.conn.commit()

    def topic_parent_rows(self) -> list[tuple[str, str]]:
        rows = self.conn.execute("SELECT child_id, parent_id FROM topic_parents").fetchall()
        return [(row["child_id"], row["parent_id"]) for row in rows]

    def add_topic_related(self, left_id: str, right_id: str) -> None:
        left = self.get_topic(left_id)
        right = self.get_topic(right_id)
        if left.id == right.id:
            raise AutoPaperError("主题不能和自己相关。")
        pair = tuple(sorted((left.id, right.id)))
        self.conn.execute(
            "INSERT OR IGNORE INTO topic_related (left_id, right_id) VALUES (?, ?)",
            pair,
        )
        self.conn.commit()

    def remove_topic_related(self, left_id: str, right_id: str) -> None:
        pair = tuple(sorted((left_id, right_id)))
        self.conn.execute("DELETE FROM topic_related WHERE left_id = ? AND right_id = ?", pair)
        self.conn.commit()

    def manual_related_pairs(self) -> set[tuple[str, str]]:
        rows = self.conn.execute("SELECT left_id, right_id FROM topic_related").fetchall()
        return {(row["left_id"], row["right_id"]) for row in rows}

    def shared_topic_pairs(self) -> set[tuple[str, str]]:
        rows = self.conn.execute("SELECT topic_id, paper_key FROM topic_papers").fetchall()
        by_paper: dict[str, set[str]] = {}
        for row in rows:
            by_paper.setdefault(row["paper_key"], set()).add(row["topic_id"])
        pairs: set[tuple[str, str]] = set()
        for topic_ids in by_paper.values():
            ordered = sorted(topic_ids)
            for index, left in enumerate(ordered):
                for right in ordered[index + 1 :]:
                    pairs.add((left, right))
        return pairs

    def direct_topic_ids(self, paper_key: str) -> set[str]:
        rows = self.conn.execute("SELECT topic_id FROM topic_papers WHERE paper_key = ?", (paper_key,)).fetchall()
        return {row["topic_id"] for row in rows}

    def resolve_paper(self, token: str) -> Paper | None:
        text = token.strip()
        from autopaper.textutil import normalize_arxiv_id, normalize_doi, normalize_pmid

        doi = normalize_doi(text)
        if doi:
            row = self.conn.execute("SELECT * FROM papers WHERE doi = ?", (doi,)).fetchone()
            if row:
                return self._paper(row)
        row = self.conn.execute("SELECT * FROM papers WHERE key = ?", (text,)).fetchone()
        if row:
            return self._paper(row)
        arxiv_id = normalize_arxiv_id(text)
        if arxiv_id:
            row = self.conn.execute("SELECT * FROM papers WHERE arxiv_id = ?", (arxiv_id,)).fetchone()
            if row:
                return self._paper(row)
        if text.lower().startswith("pmid"):
            pmid = normalize_pmid(text)
            if pmid:
                row = self.conn.execute("SELECT * FROM papers WHERE pubmed_id = ?", (pmid,)).fetchone()
                if row:
                    return self._paper(row)
        title_norm = normalize_title(text)
        if title_norm:
            rows = self.conn.execute("SELECT * FROM papers WHERE title_norm = ?", (title_norm,)).fetchall()
            if len(rows) == 1:
                return self._paper(rows[0])
        return None

    def search_papers(self, query: str, *, limit: int = 20) -> list[Paper]:
        from autopaper.textutil import normalize_doi

        doi = normalize_doi(query)
        terms = [normalize_match(part) for part in query.split() if part.strip()]
        clauses = []
        params: list = []
        if doi:
            clauses.append("doi = ?")
            params.append(doi)
        like_clauses = []
        for term in terms:
            like_clauses.append("search_text LIKE ? ESCAPE '\\'")
            params.append(like_pattern(term))
        if like_clauses:
            clauses.append("(" + " AND ".join(like_clauses) + ")")
        if not clauses:
            return []
        params.append(limit)
        rows = self.conn.execute(
            f"SELECT * FROM papers WHERE {' OR '.join(clauses)} LIMIT ?",
            params,
        ).fetchall()
        return [self._paper(row) for row in rows]

    def save_reading(self, paper_key: str, payload: dict, markdown: str) -> None:
        self.conn.execute(
            """
            INSERT INTO readings (paper_key, payload, markdown, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(paper_key) DO UPDATE SET
              payload = excluded.payload,
              markdown = excluded.markdown,
              created_at = excluded.created_at
            """,
            (paper_key, _dumps(payload), markdown, _iso(utcnow())),
        )
        self.conn.commit()

    def update_bibliography(self, paper: Paper) -> None:
        paper.cleanup()
        self.conn.execute(
            """
            UPDATE papers
            SET year = ?, venue = ?, doi = ?, url = ?, updated_at = ?
            WHERE key = ?
            """,
            (paper.year, paper.venue, paper.doi, paper.url, _iso(utcnow()), paper.key),
        )
        self.conn.commit()

    def replace_reading_markdown(self, paper_key: str, markdown: str) -> None:
        self.conn.execute(
            "UPDATE readings SET markdown = ? WHERE paper_key = ?",
            (markdown, paper_key),
        )
        self.conn.commit()

    def get_reading(self, paper_key: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM readings WHERE paper_key = ?", (paper_key,)).fetchone()
        if row is None:
            return None
        return {"payload": _loads(row["payload"], {}), "markdown": row["markdown"], "created_at": row["created_at"]}

    def add_highlight(
        self,
        *,
        scope: str,
        scope_id: str,
        scope_label: str,
        kind: str,
        color: str,
        paragraph: str,
        selected_text: str,
        comment: str,
    ) -> dict:
        existing = self.conn.execute(
            """
            SELECT * FROM highlights
            WHERE scope = ? AND scope_id = ? AND kind = ? AND color = ?
              AND paragraph = ? AND selected_text = ? AND comment = ?
            """,
            (scope, scope_id, kind, color, paragraph, selected_text, comment),
        ).fetchone()
        if existing is not None:
            return _highlight_row(existing)
        now = _iso(utcnow())
        ident = uuid.uuid4().hex
        self.conn.execute(
            """
            INSERT INTO highlights (
              id, scope, scope_id, scope_label, kind, color, paragraph, selected_text, comment, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (ident, scope, scope_id, scope_label, kind, color, paragraph, selected_text, comment, now, now),
        )
        self.conn.commit()
        row = self.conn.execute("SELECT * FROM highlights WHERE id = ?", (ident,)).fetchone()
        return _highlight_row(row)

    def list_highlights(self, scope: str = "", scope_id: str = "") -> list[dict]:
        if scope and scope_id:
            rows = self.conn.execute(
                """
                SELECT * FROM highlights
                WHERE scope = ? AND scope_id = ?
                ORDER BY created_at, id
                """,
                (scope, scope_id),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM highlights ORDER BY updated_at DESC, id"
            ).fetchall()
        return [_highlight_row(row) for row in rows]

    def delete_highlight(self, ident: str) -> None:
        cursor = self.conn.execute("DELETE FROM highlights WHERE id = ?", (ident,))
        self.conn.commit()
        if cursor.rowcount == 0:
            raise NotFoundError("没有这条高亮或批注")

    def upsert_researcher(
        self,
        researcher_id: str,
        profile: ResearcherProfile,
        summary: str,
        directions: list[str],
        report_path: str,
    ) -> str:
        self.conn.execute(
            """
            INSERT INTO researchers (
              id, name, orcid, affiliation, openalex_id, s2_id, h_index, cited_by_count,
              works_count, concepts, summary, directions, report_path, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              name = excluded.name,
              orcid = excluded.orcid,
              affiliation = excluded.affiliation,
              openalex_id = excluded.openalex_id,
              s2_id = excluded.s2_id,
              h_index = excluded.h_index,
              cited_by_count = excluded.cited_by_count,
              works_count = excluded.works_count,
              concepts = excluded.concepts,
              summary = excluded.summary,
              directions = excluded.directions,
              report_path = excluded.report_path,
              updated_at = excluded.updated_at
            """,
            (
                researcher_id,
                profile.name,
                profile.orcid,
                profile.affiliation,
                profile.openalex_id,
                profile.s2_id,
                profile.h_index,
                profile.cited_by_count,
                profile.works_count,
                _dumps(profile.concepts),
                summary,
                _dumps(directions),
                report_path,
                _iso(utcnow()),
            ),
        )
        self.conn.commit()
        return researcher_id

    def replace_relations(self, researcher_id: str, relations: list[Relation]) -> None:
        self.conn.execute("DELETE FROM researcher_relations WHERE researcher_id = ?", (researcher_id,))
        self.conn.executemany(
            """
            INSERT INTO researcher_relations (
              researcher_id, other_name, other_id, relation, weight, detail
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (researcher_id, item.other_name, item.other_id, item.relation, item.weight, item.detail)
                for item in relations
            ],
        )
        self.conn.commit()

    def link_researcher_paper(self, researcher_id: str, paper_key: str) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO researcher_papers (researcher_id, paper_key) VALUES (?, ?)",
            (researcher_id, paper_key),
        )
        self.conn.commit()

    def list_researchers(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM researchers ORDER BY updated_at DESC").fetchall()

    def list_relations(self, researcher_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT * FROM researcher_relations
            WHERE researcher_id = ?
            ORDER BY relation, weight DESC
            """,
            (researcher_id,),
        ).fetchall()

    def list_authors(self) -> list[Author]:
        rows = self.conn.execute("SELECT * FROM authors ORDER BY name").fetchall()
        return [self._author(row) for row in rows]

    def find_author(self, token: str) -> Author | None:
        text = collapse_ws(token)
        if not text:
            return None
        row = self.conn.execute("SELECT * FROM authors WHERE id = ?", (text,)).fetchone()
        if row is not None:
            return self._author(row)
        row = self.conn.execute(
            """
            SELECT authors.* FROM author_names
            JOIN authors ON authors.id = author_names.author_id
            WHERE author_names.name_norm = ?
            """,
            (normalize_match(text),),
        ).fetchone()
        if row is None:
            return None
        return self._author(row)

    def require_author(self, token: str) -> Author:
        author = self.find_author(token)
        if author is None:
            raise NotFoundError(f"没有找到作者：{token}")
        return author

    def create_author(self, name: str, aliases: list[str] | None = None) -> Author:
        primary = collapse_ws(name)
        if not primary:
            raise AutoPaperError("作者名为空")
        existing = self.find_author(primary)
        if existing is not None:
            raise AutoPaperError(f"作者已存在：{existing.name}")
        author_id = self._unique_author_id(primary)
        cleaned = self._clean_aliases(primary, aliases or [], owner_id=None)
        self._write_author(author_id, primary, cleaned, created_at=_iso(utcnow()))
        return self.require_author(author_id)

    def update_aliases(
        self,
        token: str,
        *,
        add: list[str] | None = None,
        remove: list[str] | None = None,
    ) -> Author:
        author = self.require_author(token)
        aliases = list(author.aliases)
        removed = {normalize_match(item) for item in (remove or []) if collapse_ws(item)}
        aliases = [item for item in aliases if normalize_match(item) not in removed]
        aliases = self._clean_aliases(author.name, [*aliases, *(add or [])], owner_id=author.id)
        self._write_author(author.id, author.name, aliases, created_at=None)
        return self.require_author(author.id)

    def link_author_paper(self, author_id: str, paper_key: str) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO author_papers (author_id, paper_key) VALUES (?, ?)",
            (author_id, paper_key),
        )
        self.conn.commit()

    def authors_of_paper(self, paper_key: str) -> list[Author]:
        rows = self.conn.execute(
            """
            SELECT authors.* FROM author_papers
            JOIN authors ON authors.id = author_papers.author_id
            WHERE author_papers.paper_key = ?
            ORDER BY authors.name
            """,
            (paper_key,),
        ).fetchall()
        return [self._author(row) for row in rows]

    def topics_of_paper(self, paper_key: str) -> list[Topic]:
        rows = self.conn.execute(
            """
            SELECT topics.* FROM topic_papers
            JOIN topics ON topics.id = topic_papers.topic_id
            WHERE topic_papers.paper_key = ?
            ORDER BY topics.name
            """,
            (paper_key,),
        ).fetchall()
        topics = [self._topic(row) for row in rows]
        seen = {topic.id for topic in topics}
        for topic in list(topics):
            for ancestor_id in self.ancestor_ids(topic.id):
                if ancestor_id in seen:
                    continue
                ancestor = self.get_topic(ancestor_id)
                topics.append(ancestor)
                seen.add(ancestor.id)
        topics.sort(key=lambda item: item.name.casefold())
        return topics

    def list_local_papers(self, *, author_id: str | None = None, topic_id: str | None = None) -> list[Paper]:
        sql = """
            SELECT DISTINCT papers.* FROM papers
            WHERE papers.local_path IS NOT NULL AND papers.local_path != ''
        """
        params: list = []
        if author_id:
            sql += " AND papers.key IN (SELECT paper_key FROM author_papers WHERE author_id = ?)"
            params.append(author_id)
        if topic_id:
            topic_ids = [topic_id, *self.descendant_ids(topic_id)]
            placeholders = ", ".join("?" for _ in topic_ids)
            sql += f" AND papers.key IN (SELECT paper_key FROM topic_papers WHERE topic_id IN ({placeholders}))"
            params.extend(topic_ids)
        sql += " ORDER BY papers.title"
        rows = self.conn.execute(sql, params).fetchall()
        return [self._paper(row) for row in rows]

    def has_title(self, title: str) -> bool:
        norm = normalize_title(title)
        if not norm:
            return False
        row = self.conn.execute("SELECT 1 FROM papers WHERE title_norm = ? LIMIT 1", (norm,)).fetchone()
        return row is not None

    def list_papers(self) -> list[Paper]:
        rows = self.conn.execute("SELECT * FROM papers ORDER BY title").fetchall()
        return [self._paper(row) for row in rows]

    def find_same_paper(self, paper: Paper, threshold: float = 92) -> Paper | None:
        from autopaper.models import copy_paper
        from autopaper.pipeline.dedup import same_paper

        current = copy_paper(paper).cleanup()
        for existing in self.list_papers():
            if existing.key and existing.key == current.key:
                continue
            if same_paper(existing, current, threshold):
                return existing
        return None

    def duplicate_groups(self, threshold: float = 92) -> list[list[Paper]]:
        from autopaper.pipeline.dedup import group_papers

        return group_papers(self.list_papers(), threshold)

    def merge_same_papers(self, threshold: float = 92) -> list[dict]:
        merged = []
        for group in self.duplicate_groups(threshold):
            merged.append(self._merge_paper_group(group))
        return merged

    def _merge_paper_group(self, group: list[Paper]) -> dict:
        ordered = sorted(group, key=self._paper_richness, reverse=True)
        combined = ordered[0]
        for other in ordered[1:]:
            combined = merge_papers(combined, other)
        combined.local_path = _kept_local_path(ordered)
        keeper = ordered[0]
        created_at = self._earliest_created([paper.key for paper in ordered])
        self._write_paper(keeper.key, combined, created_at, _iso(utcnow()) or "", insert=False)
        for other in ordered[1:]:
            self._adopt_key(other.key, keeper.key)
        canonical = make_key(combined)
        if canonical != keeper.key:
            occupant = self.conn.execute("SELECT 1 FROM papers WHERE key = ?", (canonical,)).fetchone()
            if occupant is None or canonical == keeper.key:
                self._adopt_key(keeper.key, canonical)
                keeper_key = canonical
            else:
                keeper_key = keeper.key
        else:
            keeper_key = keeper.key
        if combined.local_path:
            self.set_local_path(keeper_key, combined.local_path)
        removed = [paper.title for paper in ordered[1:]]
        extra_files = [
            paper.local_path
            for paper in ordered
            if paper.local_path and paper.local_path != combined.local_path
        ]
        return {
            "key": keeper_key,
            "title": combined.title,
            "removed": removed,
            "extra_files": extra_files,
        }

    def _paper_richness(self, paper: Paper) -> tuple:
        reading = self.get_reading(paper.key) or {}
        return (
            len(str(reading.get("markdown") or "")),
            len(paper.abstract or ""),
            len(normalize_title(paper.title)),
            1 if paper.doi else 0,
        )

    def _earliest_created(self, keys: list[str]) -> str:
        placeholders = ", ".join("?" for _ in keys)
        row = self.conn.execute(
            f"SELECT MIN(created_at) AS created_at FROM papers WHERE key IN ({placeholders})",
            keys,
        ).fetchone()
        return row["created_at"] if row and row["created_at"] else _iso(utcnow()) or ""

    def get_author(self, author_id: str) -> Author:
        row = self.conn.execute("SELECT * FROM authors WHERE id = ?", (author_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"没有找到作者：{author_id}")
        return self._author(row)

    def delete_paper(self, key: str) -> Paper:
        paper = self.resolve_paper(key)
        if paper is None:
            raise NotFoundError(f"没有找到论文：{key}")
        self.conn.execute("DELETE FROM highlights WHERE scope = 'paper' AND scope_id = ?", (paper.key,))
        self.conn.execute("DELETE FROM papers WHERE key = ?", (paper.key,))
        self.conn.commit()
        return paper

    def delete_author(self, author_id: str) -> Author:
        author = self.get_author(author_id)
        self.conn.execute("DELETE FROM highlights WHERE scope = 'author' AND scope_id = ?", (author.id,))
        self._delete_synthesis_highlights(author_id=author.id)
        self.conn.execute("DELETE FROM authors WHERE id = ?", (author.id,))
        self.conn.commit()
        return author

    def delete_topic_record(self, topic_id: str) -> Topic:
        topic = self.get_topic(topic_id)
        self.conn.execute("DELETE FROM highlights WHERE scope = 'topic' AND scope_id = ?", (topic.id,))
        self._delete_synthesis_highlights(topic_id=topic.id)
        self.delete_topic(topic.id)
        return topic

    def _delete_synthesis_highlights(self, *, author_id: str = "", topic_id: str = "") -> None:
        rows = self.conn.execute("SELECT id, scope_id FROM highlights WHERE scope = 'synthesis'").fetchall()
        for row in rows:
            left, _, right = str(row["scope_id"]).partition("|")
            if (author_id and left == author_id) or (topic_id and right == topic_id):
                self.conn.execute("DELETE FROM highlights WHERE id = ?", (row["id"],))

    def delete_reading(self, key: str) -> Paper:
        paper = self.resolve_paper(key)
        if paper is None:
            raise NotFoundError(f"没有找到论文：{key}")
        self.conn.execute("DELETE FROM readings WHERE paper_key = ?", (paper.key,))
        self.conn.commit()
        return paper

    def set_local_path(self, key: str, path: str) -> None:
        self.conn.execute(
            "UPDATE papers SET local_path = ?, updated_at = ? WHERE key = ?",
            (path, _iso(utcnow()), key),
        )
        self.conn.commit()

    def _unique_author_id(self, name: str) -> str:
        base = slugify(name)
        candidate = base
        suffix = 2
        while self.conn.execute("SELECT 1 FROM authors WHERE id = ?", (candidate,)).fetchone():
            candidate = f"{base}-{suffix}"
            suffix += 1
        return candidate

    def _clean_aliases(self, primary: str, aliases: list[str], *, owner_id: str | None) -> list[str]:
        cleaned: list[str] = []
        seen = {normalize_match(primary)}
        for item in aliases:
            alias = collapse_ws(item)
            norm = normalize_match(alias)
            if not alias or not norm or norm in seen:
                continue
            owner = self.find_author(alias)
            if owner is not None and owner.id != owner_id:
                raise AutoPaperError(f"“{alias}”已经属于 {owner.name}")
            seen.add(norm)
            cleaned.append(alias)
        return cleaned

    def _write_author(self, author_id: str, name: str, aliases: list[str], created_at: str | None) -> None:
        now = _iso(utcnow())
        if created_at is None:
            row = self.conn.execute("SELECT created_at FROM authors WHERE id = ?", (author_id,)).fetchone()
            created_at = row["created_at"] if row else now
        self.conn.execute(
            """
            INSERT INTO authors (id, name, aliases, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              name = excluded.name,
              aliases = excluded.aliases,
              updated_at = excluded.updated_at
            """,
            (author_id, name, _dumps(aliases), created_at, now),
        )
        self.conn.execute("DELETE FROM author_names WHERE author_id = ?", (author_id,))
        rows = [(normalize_match(name), author_id, name, 1)]
        seen = {normalize_match(name)}
        for alias in aliases:
            norm = normalize_match(alias)
            if not norm or norm in seen:
                continue
            rows.append((norm, author_id, alias, 0))
            seen.add(norm)
        self.conn.executemany(
            "INSERT INTO author_names (name_norm, author_id, name, is_primary) VALUES (?, ?, ?, ?)",
            rows,
        )
        self.conn.commit()

    def _author(self, row) -> Author:
        return Author(id=row["id"], name=row["name"], aliases=_loads(row["aliases"], []))

    def start_run(self, topic_id: str) -> int:
        cursor = self.conn.execute(
            "INSERT INTO runs (topic_id, started_at, status, stats) VALUES (?, ?, 'running', '{}')",
            (topic_id, _iso(utcnow())),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def finish_run(self, run_id: int, status: str, stats: dict) -> None:
        self.conn.execute(
            "UPDATE runs SET finished_at = ?, status = ?, stats = ? WHERE id = ?",
            (_iso(utcnow()), status, _dumps(stats), run_id),
        )
        self.conn.commit()

    def latest_run(self, topic_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM runs WHERE topic_id = ? ORDER BY id DESC LIMIT 1",
            (topic_id,),
        ).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["stats"] = _loads(row["stats"], {})
        return data

    def _find_paper_row(self, paper: Paper):
        checks = [
            ("doi", paper.doi),
            ("arxiv_id", paper.arxiv_id),
            ("pubmed_id", paper.pubmed_id),
            ("s2_id", paper.s2_id),
            ("openalex_id", paper.openalex_id),
        ]
        for column, value in checks:
            if not value:
                continue
            row = self.conn.execute(f"SELECT * FROM papers WHERE {column} = ?", (value,)).fetchone()
            if row is not None and self._compatible(row, paper):
                return row
        title_norm = normalize_title(paper.title)
        rows = self.conn.execute("SELECT * FROM papers WHERE title_norm = ?", (title_norm,)).fetchall()
        for row in rows:
            if self._compatible(row, paper):
                return row
        return None

    def _compatible(self, row, paper: Paper) -> bool:
        pairs = [
            ("doi", paper.doi),
            ("arxiv_id", paper.arxiv_id),
            ("pubmed_id", paper.pubmed_id),
            ("s2_id", paper.s2_id),
            ("openalex_id", paper.openalex_id),
        ]
        for column, value in pairs:
            existing = row[column]
            if existing and value and existing != value:
                return False
        return True

    def _write_paper(self, key: str, paper: Paper, created_at: str, updated_at: str, *, insert: bool) -> None:
        search_text = normalize_match(
            " ".join(
                [
                    paper.title,
                    paper.abstract,
                    " ".join(paper.authors),
                    paper.venue or "",
                    " ".join(paper.categories),
                    paper.doi or "",
                ]
            )
        )[:30000]
        values = (
            paper.title,
            normalize_title(paper.title),
            paper.abstract,
            _dumps(paper.authors),
            paper.year,
            paper.doi,
            paper.arxiv_id,
            paper.pubmed_id,
            paper.s2_id,
            paper.openalex_id,
            paper.url,
            paper.pdf_url,
            paper.venue,
            paper.citation_count,
            _dumps(paper.categories),
            _dumps(paper.sources),
            search_text,
            updated_at,
            paper.local_path or None,
        )
        if insert:
            self.conn.execute(
                """
                INSERT INTO papers (
                  title, title_norm, abstract, authors, year, doi, arxiv_id, pubmed_id,
                  s2_id, openalex_id, url, pdf_url, venue, citation_count, categories,
                  sources, search_text, updated_at, local_path, key, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (*values, key, created_at),
            )
        else:
            self.conn.execute(
                """
                UPDATE papers SET
                  title = ?, title_norm = ?, abstract = ?, authors = ?, year = ?, doi = ?,
                  arxiv_id = ?, pubmed_id = ?, s2_id = ?, openalex_id = ?, url = ?,
                  pdf_url = ?, venue = ?, citation_count = ?, categories = ?, sources = ?,
                  search_text = ?, updated_at = ?, local_path = ?
                WHERE key = ?
                """,
                (*values, key),
            )
        self.conn.commit()

    def _topic(self, row) -> Topic:
        return Topic(
            id=row["id"],
            name=row["name"],
            query=row["query"],
            keywords=_loads(row["keywords"], []),
            exclude_keywords=_loads(row["exclude_keywords"], []),
            domain_tags=_loads(row["domain_tags"], []),
            domain_mode=row["domain_mode"],
            sources=_loads(row["sources"], []),
            schedule=row["schedule"],
            sort=row["sort"],
            max_per_source=row["max_per_source"],
            since_year=row["since_year"],
            min_score=row["min_score"],
            llm_top_n=row["llm_top_n"],
            keyword_mode=row["keyword_mode"],
            created_at=_parse_iso(row["created_at"]),
            updated_at=_parse_iso(row["updated_at"]),
            last_run_at=_parse_iso(row["last_run_at"]),
        )

    def _paper(self, row) -> Paper:
        return Paper(
            key=row["key"],
            title=row["title"],
            abstract=row["abstract"] or "",
            authors=_loads(row["authors"], []),
            year=row["year"],
            doi=row["doi"],
            arxiv_id=row["arxiv_id"],
            pubmed_id=row["pubmed_id"],
            s2_id=row["s2_id"],
            openalex_id=row["openalex_id"],
            url=row["url"],
            pdf_url=row["pdf_url"],
            venue=row["venue"],
            citation_count=row["citation_count"],
            categories=_loads(row["categories"], []),
            sources=_loads(row["sources"], []),
            source=( _loads(row["sources"], []) or [""] )[0],
            local_path=(row["local_path"] or "") if "local_path" in row.keys() else "",
        )

    def _scored(self, row) -> ScoredPaper:
        return ScoredPaper(
            paper=self._paper(row),
            heuristic_score=row["heuristic_score"] or 0,
            llm_score=row["llm_score"],
            score=row["score"],
            reason=row["reason"] or "",
            status=row["status"],
        )


def make_new_key(paper: Paper) -> str:
    return make_key(paper)
