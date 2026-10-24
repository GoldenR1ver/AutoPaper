from __future__ import annotations

import logging

import httpx

from autopaper.config import Settings
from autopaper.errors import LLMError
from autopaper.llm.prompts import coerce_array, parse_json_payload

log = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, settings: Settings):
        if not settings.llm_api_key:
            raise LLMError("未配置 OPENAI_API_KEY 或 DEEPSEEK_API_KEY")
        self.settings = settings
        self._client = httpx.Client(
            timeout=httpx.Timeout(300.0, connect=15.0),
            headers={
                "Authorization": f"Bearer {settings.llm_api_key}",
                "Content-Type": "application/json",
            },
            follow_redirects=True,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> LLMClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def complete(self, system: str, user: str, *, max_tokens: int = 2048, json_mode: bool = False) -> str:
        url = self.settings.llm_base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.settings.llm_model,
            "temperature": self.settings.llm_temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        payload["thinking"] = {"type": "disabled"}
        response = self._post(url, payload)
        for _ in range(3):
            if response.status_code != 400:
                break
            detail = response.text
            if "thinking" in payload and "thinking" in detail:
                payload.pop("thinking", None)
            elif "response_format" in payload and ("response_format" in detail or "json_object" in detail):
                payload.pop("response_format", None)
            elif "max_tokens" in payload and "max_tokens" in detail:
                payload.pop("max_tokens", None)
            else:
                break
            response = self._post(url, payload)
        if response.status_code >= 400:
            detail = response.text[:300].replace("\n", " ")
            raise LLMError(f"模型请求失败 HTTP {response.status_code}: {detail}")
        try:
            data = response.json()
            message = data["choices"][0]["message"]
            content = message.get("content")
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError("模型响应缺少文本") from exc
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part) for part in content
            )
        if not content or not str(content).strip():
            content = message.get("reasoning_content") if isinstance(message, dict) else ""
        if not content or not str(content).strip():
            raise LLMError("模型返回了空文本")
        return str(content)

    def complete_json(self, system: str, user: str, *, expect: str = "object", max_tokens: int = 2048):
        text = self.complete(system, user, max_tokens=max_tokens, json_mode=True)
        try:
            return self._coerce(text, expect)
        except LLMError:
            log.info("模型 JSON 解析失败，重试一次")
            text = self.complete(
                system,
                user + "\n\n只输出 JSON，不要解释。",
                max_tokens=max_tokens,
                json_mode=True,
            )
            return self._coerce(text, expect)

    def _coerce(self, text: str, expect: str):
        data = parse_json_payload(text)
        if expect == "array":
            return coerce_array(data)
        if expect == "object" and not isinstance(data, dict):
            raise LLMError("模型没有返回 JSON 对象")
        return data

    def _post(self, url: str, payload: dict) -> httpx.Response:
        try:
            return self._client.post(url, json=payload)
        except httpx.HTTPError as exc:
            raise LLMError(f"模型请求失败：{exc}") from exc


class EmbeddingClient:
    """OpenAI 兼容的 /embeddings。对话模型和向量模型可以不是同一家。"""

    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None):
        if not settings.embedding_api_key or not settings.embedding_base_url or not settings.embedding_model:
            raise LLMError("未配置 embedding 接口")
        self.model = settings.embedding_model
        self.base_url = settings.embedding_base_url.rstrip("/")
        self._client = httpx.Client(
            timeout=httpx.Timeout(120.0, connect=15.0),
            headers={
                "Authorization": f"Bearer {settings.embedding_api_key}",
                "Content-Type": "application/json",
            },
            transport=transport,
            follow_redirects=True,
        )

    def close(self) -> None:
        self._client.close()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        cleaned = [text.strip() or " " for text in texts]
        vectors: list[list[float] | None] = [None] * len(cleaned)
        url = self.base_url + "/embeddings"
        batch_size = 64
        for start in range(0, len(cleaned), batch_size):
            batch = cleaned[start : start + batch_size]
            try:
                response = self._client.post(url, json={"model": self.model, "input": batch})
            except httpx.HTTPError as exc:
                raise LLMError(f"embedding 请求失败：{exc}") from exc
            if response.status_code >= 400:
                detail = response.text[:300].replace("\n", " ")
                raise LLMError(f"embedding 请求失败 HTTP {response.status_code}: {detail}")
            try:
                payload = response.json()["data"]
            except (KeyError, TypeError, ValueError) as exc:
                raise LLMError("embedding 响应缺少向量") from exc
            if not isinstance(payload, list):
                raise LLMError("embedding 响应缺少向量")
            for item in payload:
                if not isinstance(item, dict):
                    continue
                index = item.get("index")
                embedding = item.get("embedding")
                if not isinstance(index, int) or not isinstance(embedding, list):
                    continue
                absolute = start + index
                if 0 <= absolute < len(vectors):
                    vectors[absolute] = [float(value) for value in embedding]
            if any(vectors[start + offset] is None for offset in range(len(batch))):
                raise LLMError("embedding 响应缺少部分向量")
        return [vector for vector in vectors if vector is not None]


def maybe_embedder(settings) -> EmbeddingClient | None:
    if getattr(settings, "disable_llm", False) or getattr(settings, "disable_embedding", False):
        return None
    key = getattr(settings, "embedding_api_key", "") or ""
    base = getattr(settings, "embedding_base_url", "") or ""
    model = getattr(settings, "embedding_model", "") or ""
    if not key or not base or not model:
        return None
    return EmbeddingClient(settings)
