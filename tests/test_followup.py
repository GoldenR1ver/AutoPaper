from autopaper.llm.prompts import FOLLOWUP_QUESTIONS, FOLLOWUP_SYSTEM, followup_user_prompt


def test_followup_prompt_covers_a_reading_pass():
    identifiers = [item[0] for item in FOLLOWUP_QUESTIONS]
    for ident in (
        "meaning",
        "whole",
        "author",
        "topic",
        "method",
        "evidence",
        "corpus",
        "gap",
        "term",
        "reproduce",
    ):
        assert ident in identifiers
    assert "两到四句" in FOLLOWUP_SYSTEM
    assert "笔记未说明" in FOLLOWUP_SYSTEM
    whole = next(item for item in FOLLOWUP_QUESTIONS if item[0] == "whole")
    prompt = followup_user_prompt(
        scope_name="论文笔记",
        label="Alpha Paper",
        authors="Huilian Qiu",
        topics="Science of Science",
        instruction=whole[2],
        selected="推广性语言",
        paragraphs="推广性语言与资助成功有关。",
        note="# Alpha\n\n" + ("背景。" * 5000),
    )
    assert "和全文" in whole[1] or "整篇笔记" in whole[2]
    assert "归类作者：Huilian Qiu" in prompt
    assert "划取的文字" in prompt
    assert "笔记后部已省略" in prompt
