from __future__ import annotations

import json
import re

from autopaper.errors import LLMError

RANK_SYSTEM = """你是文献筛选助手。只依据给定的标题和摘要判断论文与研究主题的相关性。
不要因为论文看起来著名就给高分，也不要编造摘要里没有的内容。
相关分从 0 到 100。输出 JSON 数组，不要使用 Markdown。
数组元素包含 key、score、reason、keep。reason 用一句简体中文，keep 为 true 或 false。"""


def rank_user_prompt(topic, blocks: str) -> str:
    keywords = "、".join(topic.keywords) or "（无，按检索式判断）"
    tags = "、".join(topic.domain_tags) or "（无）"
    return (
        f"研究主题：{topic.name}\n"
        f"检索式：{topic.query}\n"
        f"主题词：{keywords}\n"
        f"领域标签：{tags}\n\n"
        "请为下面每一篇论文打分，key 必须原样复制方括号中的值。\n\n"
        f"{blocks}"
    )


READING_SYSTEM = """你是严谨的科研阅读助手。只依据给定论文文本作答，不要编造实验数字、数据集、对比方法或结论。
文本没有写明的内容，写“论文文本未说明”。使用简体中文。
先按论文自身的论证顺序理解，再回答阅读问题。主张和证据必须能在文本中找到依据。
只输出一个 JSON 对象，不要使用 Markdown 围栏。"""


def reading_user_prompt(meta: str, text: str, only_abstract: bool) -> str:
    limit_note = ""
    if only_abstract:
        limit_note = (
            "当前只有摘要，没有全文。方法、实验、结果和复现步骤如果摘要没有写明，"
            "请回答“论文文本未说明”。\n\n"
        )
    return f"""当我把一篇论文发给你时，请完成这些事：概括该论文所在领域，介绍该领域目前的发展状况，定位该论文在知识网络中的坐标。该论文基于什么背景，提出了什么问题，进行了什么实验，做出了什么结果，能不能解决前面的问题，又提出了哪些新的问题？该论文在方法、方向、理论上有什么创新点或者可能的不足？复现思路是什么？

另外请把论文拆成可核对的结构：一句话概括、关键术语、主要主张及对应证据、复现所需的数据和步骤。文本出现了比元数据更完整的标题时，用 title 返回它。

{limit_note}论文元数据：
{meta}

论文文本：
{text}

用下面这些 JSON 键作答：
title, one_sentence, field, field_status, position, background, problem, method, experiments, results, solves_problem, new_questions, innovations, limitations, reproduction, claims, terms, reader_checks
约束：
- title、one_sentence、field、field_status、position、background、problem、method、experiments、results、solves_problem 使用字符串。
- new_questions、reader_checks 使用字符串数组。
- innovations 和 limitations 使用对象，键为 method、direction、theory，值是字符串数组。分别对应方法、方向、理论。
- reproduction 使用对象，键为 data、steps、risks。data 和 risks 是字符串，steps 是字符串数组。
- claims 是对象数组，每个对象包含 claim、evidence、where。where 写章节或页码线索，没有就写“论文文本未说明”。
- terms 是对象数组，每个对象包含 term、meaning。"""


SYNTHESIS_SYSTEM = """你是科研笔记综合助手。只根据已经生成的单篇笔记写作，不要引入笔记里没有的论文、数字或结论。
使用简体中文。只输出一个 JSON 对象，不要使用 Markdown 围栏。"""


def synthesis_user_prompt(author_label: str, topic_name: str, notes: str) -> str:
    return f"""作者：{author_label}
研究主题：{topic_name}

下面是这组已归类论文的结构化笔记。请说明这些论文放在一起时，作者在该主题上的研究位置：共同问题、每篇论文扮演的角色、方法/方向/理论上的线索与分歧、仍未回答的问题，以及下一步值得核对或补做的事。

{notes}

用下面这些 JSON 键作答：
shared_problem, papers, method_thread, direction_thread, theory_thread, tensions, open_questions, next_steps
papers 是对象数组，每个对象包含 title 和 role。tensions、open_questions、next_steps 是字符串数组。其余使用字符串。"""


AUTHOR_PROFILE_SYSTEM = """你是科研笔记助手。只根据给定的本地论文笔记写一位作者的研究笔记。
不要补充笔记里没有的任职、合作者、引用或论文。笔记没有写明的关系，写“笔记未说明”。
使用简体中文。只输出一个 JSON 对象，不要使用 Markdown 围栏。"""


def author_profile_user_prompt(author_label: str, aliases: str, missing: str, notes: str) -> str:
    return f"""作者：{author_label}
别名：{aliases or "无"}
尚未生成笔记、不能当作已读结论的论文：{missing or "无"}

下面是该作者已归类论文的结构化笔记。请据此写作者笔记，至少说清研究方向、在知识网络中的定位，以及学术关系。学术关系要分开写：这些论文彼此是递进、并列、方法复用还是主题切换；笔记里点名的合作者、被对比或被对话的学者是什么关系，依据来自哪一篇；若论文分属多个主题，每个主题处在其研究中的什么位置。

{notes}

用下面这些 JSON 键作答：
direction, positioning, relationships, works, open_questions, evidence_limits
约束：
- direction、positioning、evidence_limits 是字符串。direction 写研究方向，positioning 写定位。
- relationships 是对象，键为 among_papers、scholars、topics。among_papers 是字符串。scholars 是对象数组，每个对象包含 name、relation、evidence。topics 是对象数组，每个对象包含 topic、relation。
- works 是对象数组，每个对象包含 title、role。
- open_questions 是字符串数组。"""


TOPIC_PROFILE_SYSTEM = """你是科研笔记助手。只根据给定的本地论文笔记写一个研究主题的笔记。
不要补充笔记里没有的作者、合作关系或论文。笔记没有写明的内容，写“笔记未说明”。
使用简体中文。只输出一个 JSON 对象，不要使用 Markdown 围栏。"""


def topic_profile_user_prompt(topic_name: str, missing: str, notes: str) -> str:
    return f"""研究主题：{topic_name}
尚未生成笔记、不能当作已读结论的论文：{missing or "无"}

下面是这个主题下已归类论文的结构化笔记。请写主题笔记，并按作者说明每个人的研究方向、在该主题中的定位，以及学术关系。学术关系包括：该作者自己的论文如何相连，笔记里出现的其他作者或学者和这个人是什么关系。再说明不同论文之间的关系、分歧，以及这批笔记还回答不了的问题。

每位作者的 direction、positioning、relationships 必须从该作者名下论文笔记归纳，不能因为笔记没有单独的作者主页就写成“笔记未说明”。direction 写这些论文反复处理的问题、方法和理论。positioning 写这些工作在本主题里是工具、测量、干预还是关联证据。relationships 写该作者自己的论文如何相连，以及笔记点名的学者是引用、对照、方法来源还是合作，并指出依据来自哪一篇。只有笔记里完全没有线索时才写“笔记未说明”。

{notes}

用下面这些 JSON 键作答：
overview, authors, paper_relations, tensions, open_questions
约束：
- overview 是字符串。
- authors 是对象数组，每个对象包含 name、direction、positioning、relationships。四个值都是字符串。direction 是研究方向，positioning 是定位，relationships 是学术关系。
- paper_relations 是对象数组，每个对象包含 papers、relation。papers 用论文标题，多篇用“；”连接。
- tensions、open_questions 是字符串数组。"""


REPORT_SYSTEM = """你是科研综述助手。只根据给定的论文题录和摘要写作，不要编造未提供的论文、数字或结论。
使用简体中文。提到论文时使用其标题。输出 Markdown，包含三个小节：领域现状、主要路线、不足与开放问题。"""


def report_user_prompt(topic, evidence: str) -> str:
    return (
        f"研究主题：{topic.name}\n"
        f"检索式：{topic.query}\n\n"
        "下面是本地知识库中相关性较高的论文。请据此写一份简短研究报告。\n\n"
        f"{evidence}"
    )


RESEARCHER_SYSTEM = """你是科研人员画像助手。只根据给定的机构、主题、论文题录和合作者统计来写。
不要编造论文、任职或合作关系。使用简体中文。
只输出一个 JSON 对象，键为 summary、directions、relations_note。
summary 是一段话，directions 是字符串数组，relations_note 说明该研究者和合作者、同方向研究者的关系。"""


def researcher_user_prompt(profile_text: str, papers_text: str, relations_text: str) -> str:
    return (
        f"研究者资料：\n{profile_text}\n\n"
        f"论文样本：\n{papers_text}\n\n"
        f"关系统计：\n{relations_text}"
    )


def parse_json_payload(text: str):
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start_object = cleaned.find("{")
    start_array = cleaned.find("[")
    starts = [index for index in (start_object, start_array) if index >= 0]
    if not starts:
        raise LLMError("模型没有返回 JSON")
    start = min(starts)
    end = max(cleaned.rfind("}"), cleaned.rfind("]"))
    if end <= start:
        raise LLMError("模型没有返回 JSON")
    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMError("模型返回的 JSON 无法解析") from exc


def coerce_array(data):
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("papers", "results", "items", "scores", "data"):
            value = data.get(key)
            if isinstance(value, list):
                return value
    raise LLMError("模型没有返回 JSON 数组")
