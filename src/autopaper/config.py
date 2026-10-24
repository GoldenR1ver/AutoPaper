from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from autopaper import __version__
from autopaper.errors import ConfigError

EXAMPLE_YAML = """\
# AutoPaper 本地配置。复制为 autopaper.yaml 后按需修改。
# API 密钥放在 .env 或环境变量里，不要写进这个文件。

data_dir: data

# 填真实邮箱后，OpenAlex 和 Crossref 会把请求放进礼貌池。
mailto: you@example.com

# 标题去重相似度，0-100。越高越不容易把两篇论文并成一篇。
similarity_threshold: 92

llm:
  # 任何兼容 OpenAI Chat Completions 的服务都可以。
  # 只设置 DEEPSEEK_API_KEY 时，会使用 https://api.deepseek.com 和 deepseek-flash（DeepSeek-V4.1-Flash）。
  base_url: https://api.openai.com/v1
  model: gpt-4o-mini
  temperature: 0.2
  # 向量召回。对话走 DeepSeek 时，这里填一个提供 embedding 的地址；留空则沿用上面的 base_url。
  # DeepSeek 目前没有 embedding，留空会跳过语义召回。
  embedding_base_url:
  embedding_model: text-embedding-3-small

# 字面主题词没命中时，用向量把语义相近的论文补回候选。
recall:
  similarity: 0.42
  limit: 30
  embed_limit: 400

sources:
  timeout_seconds: 40
  arxiv_delay_seconds: 3
  ncbi_delay_seconds: 0.4
  default_delay_seconds: 0.3
  s2_delay_seconds: 1

disable_llm: false
"""


@dataclass
class Settings:
    data_dir: Path
    mailto: str = ""
    similarity_threshold: float = 92
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_provider: str = ""
    llm_temperature: float = 0.2
    s2_api_key: str = ""
    ncbi_api_key: str = ""
    ncbi_email: str = ""
    timeout_seconds: float = 40
    arxiv_delay_seconds: float = 3
    ncbi_delay_seconds: float = 0.4
    default_delay_seconds: float = 0.3
    s2_delay_seconds: float = 1
    disable_llm: bool = False
    disable_embedding: bool = False
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    embedding_model: str = ""
    recall_similarity: float = 0.42
    recall_limit: int = 30
    recall_embed_limit: int = 400

    @property
    def database_path(self) -> Path:
        return self.data_dir / "autopaper.sqlite"

    @property
    def user_agent(self) -> str:
        email = self.mailto or "anonymous@autopaper.local"
        return f"AutoPaper/{__version__} (mailto:{email})"

    def http_delays(self) -> dict[str, float]:
        return {
            "arxiv": self.arxiv_delay_seconds,
            "pubmed": self.ncbi_delay_seconds,
            "semantic_scholar": self.s2_delay_seconds,
            "openalex": self.default_delay_seconds,
            "crossref": self.default_delay_seconds,
            "default": self.default_delay_seconds,
        }

    @property
    def polite_email(self) -> str:
        return self.ncbi_email or self.mailto


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _read_yaml(path: Path) -> dict:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"无法读取配置文件 {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"配置文件格式不正确 {path}: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"配置文件的顶层必须是映射: {path}")
    return loaded


def load_settings() -> Settings:
    load_dotenv(Path.cwd() / ".env")
    data: dict = {}
    config_env = os.environ.get("AUTOPAPER_CONFIG")
    if config_env:
        path = Path(config_env)
        if path.exists():
            data.update(_read_yaml(path))
    else:
        local = Path.cwd() / "autopaper.yaml"
        if local.exists():
            data.update(_read_yaml(local))

    llm = data.get("llm") or {}
    sources = data.get("sources") or {}
    recall = data.get("recall") or {}
    if not isinstance(llm, dict) or not isinstance(sources, dict) or not isinstance(recall, dict):
        raise ConfigError("llm、sources 和 recall 必须是映射")

    data_dir = Path(
        os.environ.get("AUTOPAPER_DATA_DIR")
        or data.get("data_dir")
        or "data"
    )
    data_dir = data_dir.expanduser()
    if not data_dir.is_absolute():
        data_dir = (Path.cwd() / data_dir).resolve()

    mailto = os.environ.get("AUTOPAPER_MAILTO") or str(data.get("mailto") or "")
    if mailto == "you@example.com":
        mailto = ""

    try:
        threshold = float(data.get("similarity_threshold", 92))
    except (TypeError, ValueError) as exc:
        raise ConfigError("similarity_threshold 必须是数字") from exc
    threshold = min(100.0, max(0.0, threshold))

    disable = bool(data.get("disable_llm", False))
    if "AUTOPAPER_DISABLE_LLM" in os.environ:
        disable = _as_bool(os.environ["AUTOPAPER_DISABLE_LLM"])

    try:
        temperature = float(os.environ.get("AUTOPAPER_TEMPERATURE") or llm.get("temperature", 0.2))
        timeout = float(sources.get("timeout_seconds", 40))
    except (TypeError, ValueError) as exc:
        raise ConfigError("temperature 和 timeout_seconds 必须是数字") from exc

    openai_present = "OPENAI_API_KEY" in os.environ
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if openai_present:
        api_key = openai_key
        use_deepseek = False
    elif deepseek_key:
        api_key = deepseek_key
        use_deepseek = True
    else:
        api_key = str(llm.get("api_key") or "")
        use_deepseek = False

    base_url = os.environ.get("OPENAI_BASE_URL") or str(llm.get("base_url") or "")
    model = os.environ.get("AUTOPAPER_MODEL") or str(llm.get("model") or "")
    if use_deepseek:
        if not os.environ.get("OPENAI_BASE_URL") and (not base_url or "api.openai.com" in base_url):
            base_url = "https://api.deepseek.com"
        if not os.environ.get("AUTOPAPER_MODEL") and (not model or model == "gpt-4o-mini"):
            model = "deepseek-flash"
    base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
    model = model or "gpt-4o-mini"

    disable_embedding = bool(data.get("disable_embedding", False))
    if "AUTOPAPER_DISABLE_EMBEDDING" in os.environ:
        disable_embedding = _as_bool(os.environ["AUTOPAPER_DISABLE_EMBEDDING"])
    embedding_model = os.environ.get("AUTOPAPER_EMBEDDING_MODEL") or str(llm.get("embedding_model") or "")
    explicit_embedding_base = os.environ.get("AUTOPAPER_EMBEDDING_BASE_URL") or str(
        llm.get("embedding_base_url") or ""
    )
    embedding_key = os.environ.get("AUTOPAPER_EMBEDDING_API_KEY") or ""
    if disable_embedding:
        embedding_base = ""
        embedding_key = ""
        embedding_model = ""
    else:
        embedding_key = embedding_key or api_key
        if explicit_embedding_base.strip():
            embedding_base = explicit_embedding_base.strip().rstrip("/")
        elif "api.deepseek.com" in base_url:
            embedding_base = ""
        else:
            embedding_base = base_url
        if not embedding_base:
            embedding_key = ""
            embedding_model = ""
        elif not embedding_model:
            embedding_model = "text-embedding-3-small"

    try:
        recall_similarity = float(recall.get("similarity", 0.42))
        recall_limit = int(recall.get("limit", 30))
        recall_embed_limit = int(recall.get("embed_limit", 400))
    except (TypeError, ValueError) as exc:
        raise ConfigError("recall.similarity、recall.limit 和 recall.embed_limit 必须是数字") from exc
    recall_similarity = min(1.0, max(0.0, recall_similarity))
    recall_limit = max(0, recall_limit)
    recall_embed_limit = max(0, recall_embed_limit)

    return Settings(
        data_dir=data_dir,
        mailto=mailto.strip(),
        similarity_threshold=threshold,
        llm_base_url=base_url,
        llm_api_key=api_key,
        llm_model=model,
        llm_provider="deepseek" if use_deepseek else "",
        llm_temperature=temperature,
        s2_api_key=os.environ.get("S2_API_KEY") or "",
        ncbi_api_key=os.environ.get("NCBI_API_KEY") or "",
        ncbi_email=(os.environ.get("NCBI_EMAIL") or mailto).strip(),
        timeout_seconds=timeout,
        arxiv_delay_seconds=float(sources.get("arxiv_delay_seconds", 3)),
        ncbi_delay_seconds=float(sources.get("ncbi_delay_seconds", 0.4)),
        default_delay_seconds=float(sources.get("default_delay_seconds", 0.3)),
        s2_delay_seconds=float(sources.get("s2_delay_seconds", 1)),
        disable_llm=disable,
        disable_embedding=disable_embedding,
        embedding_base_url=embedding_base,
        embedding_api_key=embedding_key,
        embedding_model=embedding_model,
        recall_similarity=recall_similarity,
        recall_limit=recall_limit,
        recall_embed_limit=recall_embed_limit,
    )
