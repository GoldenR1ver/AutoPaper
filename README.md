# AutoPaper

AutoPaper 是一个本地科研助手。它把研究主题保存下来，从 arXiv、Crossref、OpenAlex、PubMed 和 Semantic Scholar 收集题录，去重并筛选后写入 SQLite 知识库，再导出候选论文、引用列表和研究报告。你也可以把一篇论文交给它做结构化精读，或整理某位研究者的方向、代表作、合作者和同方向研究者。

没有模型密钥时，检索、去重、关键词过滤、启发式排序、导出和研究者题录仍然可用。配置一个兼容 OpenAI 的接口后，才会启用摘要打分、论文精读和综述段落。

## 安装

在项目目录打开 PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
copy config.example.yaml autopaper.yaml
copy .env.example .env
python -m autopaper init
python -m autopaper doctor
```

把 `autopaper.yaml` 里的 `mailto` 改成你的邮箱。OpenAlex 和 Crossref 会据此把请求放进礼貌池。模型密钥写在 `.env`：

```text
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.deepseek.com
AUTOPAPER_MODEL=deepseek-chat
```

`OPENAI_BASE_URL` 留空时使用配置文件里的地址。DeepSeek、OpenAI 以及其他兼容 Chat Completions 的服务都可以。可选的 `S2_API_KEY` 和 `NCBI_API_KEY` 能降低 Semantic Scholar 与 PubMed 被限速的概率，不填也能检索。

## 收集一个主题

下面这组命令对应一次完整收集：定义主题、拉取五个数据源、按 DOI 和标题去重、做主题词与领域标签过滤、打分、写入知识库，并导出 CSV、JSON、BibTeX 和候选名单。

```powershell
python -m autopaper topic add "graph neural networks for time series" `
  --keyword "graph neural network" `
  --keyword "time series" `
  --exclude survey `
  --tag "machine learning" `
  --since-year 2021 `
  --schedule weekly

python -m autopaper collect graph-neural-networks-for-time-series
python -m autopaper papers graph-neural-networks-for-time-series
python -m autopaper report graph-neural-networks-for-time-series
```

`topic add` 会打印主题 id。之后的命令可以用这个 id，也可以用完整名称。

主题词按字面匹配，标点和大小写会被忽略，所以 `time series` 可以命中 `time-series`。多个 `--keyword` 默认命中任意一个；`--keyword-mode all` 要求全部命中。字面对不上的同义词，如果配置了 embedding 接口，会按和主题的向量相似度补进候选，并参与后面的排序。没配向量接口时，这些论文仍然进不来，同义词需要分成多个主题词。对话用 DeepSeek、向量用别的服务时，把向量地址和密钥写在 `AUTOPAPER_EMBEDDING_BASE_URL` 和 `AUTOPAPER_EMBEDDING_API_KEY`。

领域标签会在分类、标题、摘要和期刊里做子串匹配。默认 `--domain-mode soft`：论文自己带了分类但一个标签都没命中时才剔除；没有分类的记录会保留。`--domain-mode strict` 则要求标签一定出现。标签建议用自然语言，例如 `machine learning`，不要只写 arXiv 分类号，否则其他数据源上的论文容易被滤掉。

`--min-score` 默认是 0。低于它的论文仍会入库，状态是 `held`，候选导出里默认不出现。模型不可用时，分数来自关键词覆盖、标题命中、被引和年份。配置了向量接口时，和主题更相近的论文会用相似度抬高分数，从而进入后面的模型重排。模型可用时，只对排序后的前 `llm_top_n` 篇重新打分，默认 20 篇，避免一次把整批摘要都送出去。

每次收集都会在 `data/exports/<主题 id>/latest/` 写下：

- `papers.csv`：用 Excel 可直接打开的 UTF-8 名单
- `papers.json`
- `papers.bib`：引用列表
- `candidates.md`：候选论文和摘要

带日期的一份副本留在同级目录，方便回看某一次运行。`--fresh` 会先清掉这个主题的旧关联再写入；论文本身仍留在知识库，因为别的主题可能也引用了它们。

`report` 读取已经入库的论文，写出年份、出处、作者分布、代表论文和引用列表。有模型密钥时会额外写“领域现状、主要路线、不足与开放问题”，并且只允许依据给定摘要，不新造论文。

## 上传自己的论文

把一个目录交给 `upload` 后，PDF、TXT 和 Markdown 会按文件名归类到 `data/library/<论文名>/`。文件名去掉扩展名后相同的论文视为同一篇，本批重复、知识库里已有、或本地已经归类过的都不会再次上传。

上传新论文时要指定作者和研究主题。可以填已有编号、主名、别名或新名称，多项用 `;` 隔开。新名称会直接建档。配置了模型密钥后，新论文会自动生成笔记，并更新同一作者、同一主题的综合笔记。也可以只针对一位作者或一个主题，根据已有单篇笔记生成研究方向、定位和学术关系：

```powershell
python -m autopaper profile --author "Huilian Qiu"
python -m autopaper profile --topic "Science of Science with semantics"
```

作者笔记写在 `data/notes/authors/<作者>/profile.md`。主题笔记写在 `data/notes/topics/<主题>/profile.md`，其中每位作者都有研究方向、定位和学术关系。图谱界面里选中一位作者或一个主题后，也可以生成或查看这份笔记。

```powershell
python -m autopaper upload D:\papers\qiu
python -m autopaper upload D:\papers\qiu --author "Huilian Sophie Qiu" --topic "time series"
python -m autopaper author alias "Huilian Sophie Qiu" --add "Huilian Qiu" --add "Qiu Huilian" --add "邱惠莲"
python -m autopaper author list
```

作者对外显示为主名加括号别名，例如 `Huilian Sophie Qiu（Huilian Qiu / Qiu Huilian / 邱惠莲）`。之后再用别名选择，会指向同一个作者。

上传时会从原文第一页识别年份、期刊或会议、DOI 和链接。识别只用 PDF 文本和文件名，不调用模型。模板 DOI、截断的 DOI 和参考文献里的 DOI 不会写入。识别结果会显示在上传结果里，也可以在论文页随时修改。已经归档的论文可以补上仍然空着的字段：

```powershell
python -m autopaper biblio
```

## 生成结构化笔记

`notes` 会阅读已经归类的本地论文。每篇笔记写在论文自己的目录里：`note.md` 和 `note.json`。同一作者、同一主题再写一份综合笔记，放在 `data/notes/<作者>/<主题>/`，里面有 `index.md`、`synthesis.md` 和 `synthesis.json`。

单篇笔记保留原来的阅读问题：领域、领域现状、知识网络坐标、背景、问题、实验、结果、是否解决问题、新问题，以及方法、方向、理论上的创新和不足、复现思路。另外增加一句话概括、主张与证据、关键术语和需要核对的地方，方便之后对照原文。

只配置了 `DEEPSEEK_API_KEY` 时，默认使用 `https://api.deepseek.com` 和 `deepseek-flash`。这个模型名当前对应 DeepSeek-V4.1-Flash。

```powershell
python -m autopaper notes --author "Huilian Qiu" --topic "Science of Science with semantics"
```

## 打开图谱界面

```powershell
python -m autopaper serve
```

浏览器里会打开本机页面。论文按笔记内容的相近程度连线，线越粗越接近；选中一篇后，其他论文按和它的相关度改变大小。左侧可以按作者、主题和相关度筛选。点开论文阅读笔记，点开作者可以增删别名。上传支持多文件或整个文件夹，同名论文会跳过，新论文会自动生成笔记，并给出从原文识别的年份、期刊或会议、DOI 和链接。打开论文后可以随时改这四项；「补全题录」和「从原文补全」只填写仍然空着的字段。

笔记标题旁会显示更新日期。阅读论文、作者、主题或综合笔记时，可以点「更新笔记」重新调用模型；已有的高亮和批注不会被删掉。在笔记里用鼠标左键划取文字后，可以选择黄、绿、蓝、粉、橙高亮，或写下批注。也可以点「追问」，让模型用两三句话说明这段在说什么、和全文、作者、主题、方法、证据及其他论文的关系，以及还缺什么、术语和能否沿用。回答出来后点「快速保存为笔记」，这段会高亮，回答写进同一条笔记。保存的是所在整段，并写入「我的笔记库」（`data/notes/highlights/library.md`）。

## 精读一篇论文

精读问题集中在 `src/autopaper/llm/prompts.py` 的 `reading_user_prompt`。当前要求是：概括领域和领域现状，给论文在知识网络中定位，说明背景、问题、实验、结果、是否解决问题、新问题，再写出方法、方向或理论上的创新与不足，并给出复现思路。文本里没有的内容会写成“论文文本未说明”。

```powershell
python -m autopaper read 10.1000/example
python -m autopaper read --file paper.pdf --title "论文标题" --topic graph-neural-networks-for-time-series
```

结果写入知识库，同时保存到 `data/reports/readings/`。只有摘要时也可以读，但实验和复现部分会明确说明依据不足。PDF 文本用 pypdf 提取，加密的 PDF 会直接提示无法读取。

## 研究者与参考文献

```powershell
python -m autopaper researcher "Yoshua Bengio"
python -m autopaper researcher "Jane Doe" --pick 2
python -m autopaper researcher --openalex-id A5023888391 "Yoshua Bengio"
python -m autopaper citations 10.1000/example
```

研究者画像使用 OpenAlex，能对上 Semantic Scholar 时补 h-index 和论文。报告包含方向、高被引论文、近年论文、合作者和同主题研究者，并附一份 Mermaid 关系图和 `*.graph.json`。合作次数来自本次采样，不是完整职业生涯。同名且机构接近时，命令会停下来列出序号，用 `--pick` 指定。

`citations` 需要论文已在库里，并且有 DOI、arXiv id 或 Semantic Scholar id。它把参考文献写成 BibTeX 和 Markdown，放在 `data/exports/citations/`。

本地查找用：

```powershell
python -m autopaper search "time series"
```

## 每天或每周自动更新

主题的 `--schedule` 可以是 `manual`、`daily` 或 `weekly`。手动主题不会自己到期。

```powershell
python -m autopaper schedule run --dry-run
python -m autopaper schedule run
python -m autopaper schedule run --all
```

`--all` 会忽略周期，把所有主题再跑一遍。全部数据源都失败时，不会更新“上次运行时间”，方便同一天重试。

Windows 计划任务可以调用仓库里的脚本。脚本会先进入项目目录，并优先使用 `.venv`：

```powershell
schtasks /Create /TN AutoPaper /SC WEEKLY /D MON /ST 09:00 /TR "powershell -NoProfile -ExecutionPolicy Bypass -File D:\LEARNING_RESOURCE\AutoPaper\scripts\run_schedule.ps1"
```

定时运行前如果希望省掉模型费用，在 `.env` 里设置 `AUTOPAPER_DISABLE_LLM=1`。这样仍会收集、去重、过滤并用启发式分数排序。

## 代码怎么改

| 想改的事情 | 位置 |
| --- | --- |
| 阅读、打分、综述、研究者提示词 | `src/autopaper/llm/prompts.py` |
| 新增或调整一个数据源 | `src/autopaper/sources/`，并在 `sources/__init__.py` 注册 |
| 去重阈值 | `autopaper.yaml` 的 `similarity_threshold`；分桶和规则在 `pipeline/dedup.py` |
| 关键词和领域标签 | `pipeline/filter.py` |
| 语义召回 | `pipeline/recall.py`；阈值在 `autopaper.yaml` 的 `recall` |
| 启发式分数 | `pipeline/rank.py` |
| 表结构 | `storage/db.py`。新改动追加迁移版本，不要改已经发布的建表语句 |
| 导出字段 | `storage/export.py` |
| 命令行 | `cli.py` |

数据流是一条直线：数据源把结果收成同一种 `Paper`，去重和过滤之后打分，`KnowledgeBase` 负责合并与保存，导出和报告只读库。数据源失败不会让整次运行中断，结果记在该主题的运行记录里。

SQLite 文件在 `data/autopaper.sqlite`。同一 DOI，或同一规范化标题且标识不冲突的记录，会合成一条，所以重复收集不会把一篇论文存成两条。合并后的主键按 DOI、arXiv、PubMed、OpenAlex、Semantic Scholar、标题这个固定顺序生成，两次收集的到达顺序不同，导出的 key 也相同。两个不同 DOI 即使标题一样也会分开，避免把短标题如 “Introduction” 误并。

## 参考项目里借用了什么

实现是独立写的，没有拷贝参考仓库的代码。文献记录收成统一结构，这个用法来自 litstudy 的多源文档模型。主题检索再进入阅读，来自 auto_research 的组织方式。本地 SQLite 作为可重复运行的缓存，来自 SPAR。研究者、主题和论文分开存储，避免把原始接口响应整包塞进库里，这个边界参考了 paperseek 的知识库分层。SurveyX 和 DeepResearch 更偏向长篇综述和网络代理，这里没有把那条链路放进来；报告停在可核对的题录、摘要和引用上。

## 使用时的边界

- 五个数据源都不需要付费密钥，但会限速。arXiv 默认每次请求间隔 3 秒。
- Semantic Scholar 的检索按相关度取样，起始年份仍然生效。`date` 排序作用于 arXiv、Crossref、OpenAlex 和 PubMed。
- Crossref 经常没有摘要，这些论文主要靠标题参与过滤和排序。
- 模型看到的 PDF 文本最长约 4 万字符，超长论文会保留开头和结尾。
- 精读和综述只根据手头文本生成。它不是实验复现器，也不会代替你核对原文中的数字。

开发检查：

```powershell
pytest
```
