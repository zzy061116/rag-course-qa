"""Embedding（向量化）层：统一封装多家云端接口，屏蔽厂商差异。

关键背景：DeepSeek 官方只提供对话补全接口，不提供 Embedding 接口，
因此本项目把「生成」和「向量化」解耦成两个可独立替换的组件。

支持：
- dashscope：通义 text-embedding-v3（支持指定维度，1024 维性价比高）
- zhipu：智谱 embedding-3（密钥是 {id}.{secret}，需自行签发 JWT）
- openai：任意 OpenAI 兼容的 /embeddings 端点
- local_hash：离线降级，确定性哈希向量（无语义，仅用于无密钥时跑通整条链路）
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import time

import httpx
import numpy as np

from app.config import Settings, get_settings


def _zhipu_token(api_key: str) -> str:
    """智谱密钥形如 `id.secret`，需要用 HS256 自签一个短期 JWT。"""

    def b64(obj: dict) -> str:
        raw = json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    kid, _, secret = api_key.partition(".")
    if not secret:
        raise ValueError("ZHIPU_API_KEY 格式应为 {id}.{secret}")
    now_ms = int(time.time() * 1000)
    header = b64({"alg": "HS256", "sign_type": "SIGN", "typ": "JWT"})
    payload = b64({"api_key": kid, "exp": now_ms + 3_600_000, "timestamp": now_ms})
    signature = hmac.new(secret.encode(), f"{header}.{payload}".encode(), hashlib.sha256).digest()
    return f"{header}.{payload}.{base64.urlsafe_b64encode(signature).decode().rstrip('=')}"


class EmbeddingError(RuntimeError):
    pass


class EmbeddingClient:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()
        self.provider = self.s.embedding_provider
        self.model = self.s.embedding_model
        self.dim = self.s.embedding_dim
        self._client = httpx.Client(timeout=30.0)

    # ---------------- 对外接口 ----------------
    def embed_documents(self, texts: list[str], batch_size: int = 10) -> list[list[float]]:
        if not texts:
            return []
        if self.provider == "local_hash":
            return [self._hash_vector(t) for t in texts]
        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            out.extend(self._call_api(texts[i : i + batch_size]))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    # ---------------- 各厂商实现 ----------------
    def _call_api(self, batch: list[str]) -> list[list[float]]:
        if self.provider == "dashscope":
            return self._dashscope(batch)
        if self.provider == "zhipu":
            return self._zhipu(batch)
        if self.provider == "openai":
            return self._openai(batch)
        raise EmbeddingError(f"未知的 embedding_provider: {self.provider}")

    def _dashscope(self, batch: list[str]) -> list[list[float]]:
        resp = self._client.post(
            "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding",
            headers={
                "Authorization": f"Bearer {self.s.dashscope_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "input": {"texts": batch},
                "parameters": {"dimension": self.dim},
            },
        )
        data = self._json(resp)
        if resp.status_code >= 400:
            raise EmbeddingError(f"通义 embedding 调用失败 {resp.status_code}: {data}")
        items = data.get("output", {}).get("embeddings", [])
        return [item["embedding"] for item in items]

    def _zhipu(self, batch: list[str]) -> list[list[float]]:
        resp = self._client.post(
            "https://open.bigmodel.cn/api/paas/v4/embeddings",
            headers={
                "Authorization": f"Bearer {_zhipu_token(self.s.zhipu_api_key)}",
                "Content-Type": "application/json",
            },
            json={"model": self.model or "embedding-3", "input": batch},
        )
        data = self._json(resp)
        if resp.status_code >= 400:
            raise EmbeddingError(f"智谱 embedding 调用失败 {resp.status_code}: {data}")
        return [item["embedding"] for item in data.get("data", [])]

    def _openai(self, batch: list[str]) -> list[list[float]]:
        resp = self._client.post(
            f"{self.s.openai_base_url.rstrip('/')}/embeddings",
            headers={
                "Authorization": f"Bearer {self.s.openai_api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self.model, "input": batch},
        )
        data = self._json(resp)
        if resp.status_code >= 400:
            raise EmbeddingError(f"OpenAI 兼容 embedding 调用失败 {resp.status_code}: {data}")
        return [item["embedding"] for item in data.get("data", [])]

    def _json(self, resp: httpx.Response) -> dict:
        try:
            return resp.json()
        except Exception:  # 非 JSON 响应（网关错误页等）
            return {"raw": resp.text[:500]}

    # ---------------- 离线降级 ----------------
    def _hash_vector(self, text: str) -> list[float]:
        """词袋哈希向量：无语义但可复现，用于无密钥时验证链路。"""
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens: list[str] = []
        # 中文没有空格，按 2-gram 切分近似分词（单字噪声大，整句又无法匹配）
        for seg in re.findall(r"[\u4e00-\u9fff]+", text.lower()):
            tokens.extend(seg[i : i + 2] for i in range(max(len(seg) - 1, 1)))
        tokens.extend(re.findall(r"[A-Za-z]+|\d+", text.lower()))
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            value = int(digest, 16)
            vec[value % self.dim] += 1.0 if (value >> 8) % 2 else -1.0
        norm = float(np.linalg.norm(vec))
        return (vec / norm).tolist() if norm > 0 else vec.tolist()
