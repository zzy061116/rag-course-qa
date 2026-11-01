"""LLM 层：OpenAI 兼容协议调用 DeepSeek，支持流式输出。

DeepSeek 与 OpenAI 接口兼容，因此直接用官方 SDK 换 base_url 即可，
代码里不写死厂商，换成任意 OpenAI 兼容模型只改环境变量。
"""

from __future__ import annotations

from collections.abc import Iterator

from openai import OpenAI

from app.config import Settings, get_settings


class LLMClient:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()
        self.model = self.s.llm_model
        self._client: OpenAI | None = None

    @property
    def enabled(self) -> bool:
        return self.s.llm_enabled

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self.s.llm_api_key, base_url=self.s.llm_base_url)
        return self._client

    def chat(self, messages: list[dict]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.s.temperature,
            max_tokens=self.s.max_tokens,
        )
        return resp.choices[0].message.content or ""

    def stream(self, messages: list[dict]) -> Iterator[str]:
        """逐段产出文本增量，供 SSE 推送给前端实现打字机效果。"""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.s.temperature,
            max_tokens=self.s.max_tokens,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


def offline_answer(question: str, hits: list[dict]) -> str:
    """没有配置 LLM 密钥时的降级回答：只做检索结果摘要，明确标注未生成。"""
    if not hits:
        return "未配置大模型密钥（DEEPSEEK_API_KEY），且本次没有检索到相关资料。"
    lines = ["[离线模式] 未配置大模型密钥，以下为检索到的原文片段，未经过模型生成：", ""]
    for i, hit in enumerate(hits, start=1):
        meta = hit.get("metadata", {})
        lines.append(
            f"{i}. {meta.get('source', '未知文档')} 第 {meta.get('page', '?')} 页 "
            f"(相似度 {hit.get('score', 0):.2f})\n{hit.get('text', '')[:300]}"
        )
    return "\n".join(lines)
