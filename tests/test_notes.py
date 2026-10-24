from pathlib import Path

from autopaper.analysis.notes import run_notes
from autopaper.catalog import resolve_authors, resolve_topics
from autopaper.library import plan_uploads, store_uploads
from autopaper.storage.db import KnowledgeBase


class FakeLLM:
    def __init__(self):
        self.settings = type("Settings", (), {"llm_model": "deepseek-flash"})()
        self.prompts = []

    def complete_json(self, system, user, **kwargs):
        self.prompts.append(user)
        if "一位作者" in system:
            return {
                "direction": "把难以观察的社会因素做成可测量工具。",
                "positioning": "位于科学学与行为软件工程之间。",
                "relationships": {
                    "among_papers": "两篇论文复用同一套测量思路。",
                    "scholars": [{"name": "Guizani", "relation": "笔记中被对照的研究者", "evidence": "Alpha Paper"}],
                    "topics": [{"topic": "Science of Science", "relation": "主要主题"}],
                },
                "works": [{"title": "Alpha Paper", "role": "给出测量例子"}],
                "open_questions": ["因果尚未确立"],
                "evidence_limits": "只覆盖已生成笔记的论文。",
            }
        if "研究主题" in system:
            return {
                "overview": "这个主题下的论文都在把软性因素操作化。",
                "authors": [{
                    "name": "Huilian Qiu",
                    "direction": "测量科学中的语言与态度。",
                    "positioning": "提供工具而不是因果证明。",
                    "relationships": "各篇之间是方法复用，笔记未说明外部合作者。",
                }],
                "paper_relations": [{"papers": "Alpha Paper；Beta Study", "relation": "同一测量思路"}],
                "tensions": ["样本范围不同"],
                "open_questions": ["指标能否迁移"],
            }
        if "综合" in system:
            return {
                "shared_problem": "宣传语言如何影响科学采纳",
                "papers": [{"title": "Alpha Paper", "role": "给出一个可测量的例子"}],
                "method_thread": "用文本指标描述语言",
                "direction_thread": "科学学",
                "theory_thread": "笔记未给出统一理论",
                "tensions": ["样本范围不同"],
                "open_questions": ["指标是否可迁移"],
                "next_steps": ["核对原文中的数字"],
            }
        return {
            "one_sentence": "这篇论文讨论宣传语言。",
            "field": "科学学",
            "field_status": "正在把语言特征做成可测指标。",
            "position": "位于科学传播和计量研究之间。",
            "background": "申请书和论文标题会使用宣传性措辞。",
            "problem": "这种措辞是否影响采纳。",
            "method": "人工标注加回归。",
            "experiments": "在公开文本上估计相关关系。",
            "results": "宣传性措辞与采纳指标相关。",
            "solves_problem": "提供了相关证据，没有证明因果。",
            "new_questions": ["其他领域是否一样"],
            "innovations": {"method": ["新的语言指标"], "direction": ["科学学"], "theory": []},
            "limitations": {"method": ["样本有限"], "direction": [], "theory": []},
            "reproduction": {"data": "公开文本", "steps": ["标注", "回归"], "risks": "标注不一致"},
            "claims": [{"claim": "措辞与采纳相关", "evidence": "回归系数", "where": "结果"}],
            "terms": [{"term": "promotional language", "meaning": "夸大贡献的措辞"}],
            "reader_checks": ["核对样本量"],
        }


def test_notes_are_written_beside_papers_and_collected_by_author(tmp_path: Path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "Alpha Paper.txt").write_text("promotional language in science", encoding="utf-8")
    with KnowledgeBase(tmp_path / "kb.sqlite") as kb:
        planned = plan_uploads(inbox, kb, tmp_path)
        authors = resolve_authors(kb, "Huilian Qiu")
        topics = resolve_topics(kb, "Science of Science with semantics")
        store_uploads(planned, kb, authors, topics)
        written = run_notes(
            kb,
            tmp_path,
            FakeLLM(),
            author=authors[0],
            topic=topics[0],
            echo=lambda *_args, **_kwargs: None,
        )
    note = tmp_path / "library" / "alpha-paper" / "note.md"
    synthesis = tmp_path / "notes" / "huilian-qiu" / "science-of-science-with-semantics" / "synthesis.md"
    index = synthesis.parent / "index.md"
    assert note in written
    assert synthesis in written
    text = note.read_text(encoding="utf-8")
    assert "DeepSeek-V4.1-Flash" in text
    assert "主张与证据" in text
    assert "新的语言指标" in text
    assert "共同问题" in synthesis.read_text(encoding="utf-8")
    assert "Alpha Paper" in index.read_text(encoding="utf-8")


def test_new_upload_note_refreshes_synthesis_with_existing_papers(tmp_path: Path):
    from autopaper.analysis.notes import note_papers

    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "Alpha Paper.txt").write_text("promotional language", encoding="utf-8")
    (inbox / "Beta Study.txt").write_text("grant success", encoding="utf-8")
    with KnowledgeBase(tmp_path / "kb.sqlite") as kb:
        planned = plan_uploads(inbox, kb, tmp_path)
        authors = resolve_authors(kb, "Huilian Qiu")
        topics = resolve_topics(kb, "Science of Science")
        store_uploads(planned, kb, authors, topics)
        stored = {paper.title: paper for paper in kb.list_local_papers()}
        kb.save_reading(stored["Alpha Paper"].key, {"title": "Alpha Paper", "one_sentence": "已有笔记"}, "# Alpha\n")
        (Path(stored["Alpha Paper"].local_path).parent / "note.md").write_text("# Alpha\n", encoding="utf-8")
        llm = FakeLLM()
        written = note_papers(kb, tmp_path, llm, [stored["Beta Study"]], echo=lambda *_args, **_kwargs: None)
    assert any(path.name == "note.md" and "beta-study" in path.as_posix() for path in written)
    assert any(path.name == "synthesis.md" for path in written)
    assert any("Alpha Paper" in prompt and "Beta Study" in prompt for prompt in llm.prompts)


def test_author_and_topic_profiles_cover_direction_position_and_relations(tmp_path: Path):
    from autopaper.analysis.notes import author_profile_path, generate_author_profile, generate_topic_profile, topic_profile_path

    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "Alpha Paper.txt").write_text("promotional language", encoding="utf-8")
    with KnowledgeBase(tmp_path / "kb.sqlite") as kb:
        planned = plan_uploads(inbox, kb, tmp_path)
        authors = resolve_authors(kb, "Huilian Qiu")
        topics = resolve_topics(kb, "Science of Science")
        store_uploads(planned, kb, authors, topics)
        paper = kb.list_local_papers()[0]
        kb.save_reading(paper.key, {"title": "Alpha Paper", "one_sentence": "推广性语言", "position": "科学学"}, "# Alpha\n")
        llm = FakeLLM()
        author_note = generate_author_profile(kb, tmp_path, llm, authors[0])
        topic_note = generate_topic_profile(kb, tmp_path, llm, topics[0])
    author_text = author_note.read_text(encoding="utf-8")
    topic_text = topic_note.read_text(encoding="utf-8")
    assert author_note == author_profile_path(tmp_path, authors[0])
    assert topic_note == topic_profile_path(tmp_path, topics[0])
    for heading in ("研究方向", "定位", "学术关系", "Guizani"):
        assert heading in author_text
    for heading in ("研究方向", "定位", "学术关系", "Huilian Qiu"):
        assert heading in topic_text
    assert any("研究方向" in prompt and "学术关系" in prompt for prompt in llm.prompts)
