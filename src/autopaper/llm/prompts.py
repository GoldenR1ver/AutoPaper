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


FOLLOWUP_SYSTEM = """你是科研阅读助手。只根据给定的本地笔记和划取段落回答。
不要补充笔记里没有的论文、数字、任职、合作关系或引用。笔记没有写明的内容，写“笔记未说明”。
使用简体中文，只写两到四句，不要分节，不要使用 Markdown。
只输出一个 JSON 对象，键为 answer。"""


FOLLOWUP_QUESTIONS = (
    ("meaning", "这段在说什么", "用两到四句说明这段实际在主张什么，点明它谈论的对象、做法或结果。"),
    ("whole", "和全文的关系", "说明这段在整篇笔记中的角色：背景、问题、方法、结果、限度还是新问题，以及它如何支撑或限制全文的结论。"),
    ("author", "和作者的关系", "说明这段怎样体现归类作者的研究方向、定位或惯用方法。只使用笔记里能看到的论文，不要写成在线作者主页。"),
    ("topic", "和主题的关系", "说明这段把当前主题推进了哪一步。它是这个主题的直接证据、方法工具，还是只是相邻材料。"),
    ("method", "方法能支持什么", "说明这段使用或描述的方法是什么，它能支持哪类判断，不能支持哪类判断。"),
    ("evidence", "主张和证据", "分开写出这段里的主张、支撑它的证据，以及尚未证明的部分。"),
    ("corpus", "和其他论文的关系", "若笔记提到其他论文或学者，说明这段与它们是递进、并列、方法复用、对照还是分歧，并指出依据来自哪里。没有提到就写笔记未说明。"),
    ("gap", "还缺什么", "指出读完这段后，若要继续这项研究，笔记还回答不了的一个关键问题。"),
    ("term", "术语", "解释这段里最需要核对的一个术语或指标，并说明笔记给它的定义。笔记没定义就写笔记未说明。"),
    ("reproduce", "能否沿用", "说明若要把这段的做法用到自己的研究里，笔记已经给出了什么，还缺什么材料或步骤。"),
)


def followup_question(question_id: str) -> tuple[str, str] | None:
    for ident, label, instruction in FOLLOWUP_QUESTIONS:
        if ident == question_id:
            return label, instruction
    return None


def followup_user_prompt(
    *,
    scope_name: str,
    label: str,
    authors: str,
    topics: str,
    instruction: str,
    selected: str,
    paragraphs: str,
    note: str,
) -> str:
    clipped = _clip_note(note)
    return f"""笔记类型：{scope_name}
笔记名称：{label}
归类作者：{authors or "笔记未说明"}
归类主题：{topics or "笔记未说明"}

要回答的问题：{instruction}

划取的文字：
{selected or "无"}

划取所在的段落：
{paragraphs}

整篇笔记：
{clipped}"""


def _clip_note(note: str, limit: int = 12000) -> str:
    text = (note or "").strip()
    if len(text) <= limit:
        return text or "笔记未说明"
    head = int(limit * 0.75)
    return text[:head] + "\n\n……（笔记后部已省略）……\n\n" + text[-(limit - head) :]


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


ANNOTATE_SYSTEM = """你是严谨的论文批注助手。只标出给定论文文本里真实出现的原句或术语。
quote 必须从论文文本连续抄出，不要改写、不要用省略号、不要拼接不相邻的句子。
文本里没有的内容不要编造条目。论文没写的数字、系数和阈值不要写成论文的结果。
通用知识必须标明：范围用“通用标准：”，计算用“通用定义：”，指标的一般作用用“通用作用：”，对数值的一般解读用“通常意味着：”。
使用简体中文写说明。
只输出一个 JSON 对象，不要使用 Markdown 围栏。"""


def annotate_user_prompt(text: str) -> str:
    return f"""请按精读习惯，为下面的论文原文挑选需要高亮的片段。每条的 quote 必须能在原文中逐字找到。不要输出 metric，指标另一次处理。

种类 kind 只能是：
- keyword：概念名、构念、方法专名、效度或统计量的名称。只标术语本身，大约两到六个词，不要整句。
- definition：论文自己的定义、操作化定义，以及量表怎么划分、选项、目标人群、计分规则。quote 要能看出在定义哪一个概念，不要只标 optional 这种单词。note 用一句话复述它界定了什么。
- related：被引用的前人、理论、指南、推荐阈值或推荐做法。note 只写原文里它和本论文的关系；原文没写关系就写“论文文本未说明”。
- method：具体做了什么，包括谁来做、多少人、哪一步、如何分组、如何计分。不要把结果句标成方法。
- conclusion：研究目的、结果、对结果的解释、贡献。包括表明、支持、证实、预测了、贡献是。不要把背景句标成结论。
- design_reason：为什么这样定义、分组、旋转、措辞或设置选项。note 复述这个原因。
- limitation：论文承认的边界、没做的检验、样本或措辞带来的限制。note 用一句话写清限制的是什么。

同一段文字只使用一种 kind。关键词和它的定义句要分开标。
数量上限：keyword 20，definition 12，related 10，method 10，conclusion 12，design_reason 8，limitation 8。短文少标，不要为了凑数去标无关句。

论文文本：
{text}

输出 JSON 对象，键为 items。items 是对象数组，每个对象包含 kind、quote、note。note 在 keyword、method、conclusion 上可以是空字符串。"""


def annotate_metric_prompt(text: str) -> str:
    return f"""请只批注这篇论文在分析、实验结果和表格里实际使用的指标。每条 kind 必须是 metric，quote 必须能在原文中逐字找到。

要覆盖的内容：拟合指数、信度、因子载荷、相关、区分效度、取样充分性、显著性检验，以及其他被论文用来支持结论的统计量。普通单词不要标。
同一指标多次出现时，quote 选带有论文数值或论文解释的那一处，优先选结果和表格，不要选文献综述里第一次出现的名字。
一段里有多个指标就拆成多条，每条 quote 尽量只包含一个指标和它旁边的数值。表格里的指标名和数字也要标。
最多 14 条。

每条包含：
- why：论文为什么用它。论文没写原因时，以“通用作用：”开头说明它在这类分析里做什么。
- range：取值范围和常用好坏标准。论文写了就按论文写。论文没写就以“通用标准：”开头。不要把通用阈值说成论文自己的标准。
- finding：先复述原文中的数值，以及原文对这个数值的解释。再以“通常意味着：”开头，用一句话说明按通用标准这个数值通常反映什么特征。原文没有数字就写“论文文本未说明”，不要编造数字。
- calculation：论文给了公式或算法就按论文写。没给就以“通用定义：”开头。不要编造系数。无法确定就写“论文文本未说明”。
- measures：这个指标一般用来衡量什么特征。

格式例子只说明字段怎么写，不要把例子里的数字写进其他论文：
quote 为 “CFI = .97” 时，why 可以是“论文用它检验模型拟合。”，range 可以是“通用标准：多在 0 到 1 之间，.95 及以上常视为拟合良好。”，finding 可以是“论文报告 CFI = .97。通常意味着：拟合达到常用的良好标准。”，calculation 可以是“通用定义：比较假设模型与基线模型的拟合，并对复杂度作惩罚。”，measures 可以是“模型与数据的吻合程度。”

论文文本：
{text}

输出 JSON 对象，键为 items。每个对象包含 kind、quote、why、range、finding、calculation、measures。kind 只能是 metric。"""


SUGGEST_SYSTEM = """你是文献归类助手。只从给定的已有作者和已有主题中选择，不要发明名单之外的条目。
每篇论文单独判断。作者只根据该篇的作者栏或文件名，主题只根据该篇的摘要和结论。
没有摘要和结论时，才可以看文首。不要根据其他论文，也不要根据未提供的正文。
没有合适项时对应数组为空。只输出一个 JSON 对象，不要使用 Markdown。"""


def suggest_user_prompt(authors: str, topics: str, text: str) -> str:
    return f"""已有作者（id 必须原样复制）：
{authors or "（无）"}

已有主题（id 必须原样复制）：
{topics or "（无）"}

{text}

输出 JSON 对象，键为 papers。papers 是数组，每项包含 name、authors、topics、reason。
name 必须与给出的文件名完全一致。authors 和 topics 是该篇自己的 id 数组。reason 用一句简体中文。"""


CLASSIFY_SYSTEM = """你是文献归类助手。只从给定的已有主题中选择，不要发明新主题。
一篇论文可以属于多个主题。只根据给出的标题、摘要和结论判断。没有这些栏目时，才看文首。
没有合适主题时 topics 为空数组。只输出一个 JSON 对象，不要使用 Markdown。"""


def classify_user_prompt(topics: str, text: str) -> str:
    return f"""已有主题（id 必须原样复制）：
{topics}

{text}

输出 JSON 对象，包含 topics 和 reason。topics 是 id 数组。reason 用一句简体中文。"""


def coerce_array(data):
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("papers", "results", "items", "scores", "data"):
            value = data.get(key)
            if isinstance(value, list):
                return value
    raise LLMError("模型没有返回 JSON 数组")
