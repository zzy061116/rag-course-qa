"""全局配置：统一从环境变量 / .env 读取。

设计要点：
- LLM 与 Embedding 解耦，可独立替换服务商（DeepSeek 官方没有 Embedding 接口）。
- 所有云服务商密钥都不进代码库，只从 .env 注入。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------------- LLM（对话生成）----------------
    llm_provider: str = "deepseek"  # deepseek | openai
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"

    # ---------------- Embedding（向量化）----------------
    embedding_provider: str = "dashscope"  # dashscope | zhipu | openai | local_hash
    embedding_model: str = "text-embedding-v3"
    embedding_dim: int = 1024
    dashscope_api_key: str = ""
    zhipu_api_key: str = ""

    # ---------------- 向量库 ----------------
    chroma_dir: str = "./data/chroma"
    collection: str = "course_kb"

    # ---------------- 分块与检索 ----------------
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 4

    # ---------------- 文件存储 ----------------
    upload_dir: str = "./data/uploads"
    max_upload_mb: int = 50

    # ---------------- 生成参数 ----------------
    temperature: float = 0.2
    max_tokens: int = 1024

    # ---------------- 派生属性 ----------------
    @property
    def llm_api_key(self) -> str:
        return self.deepseek_api_key if self.llm_provider == "deepseek" else self.openai_api_key

    @property
    def llm_base_url(self) -> str:
        return self.deepseek_base_url if self.llm_provider == "deepseek" else self.openai_base_url

    @property
    def llm_model(self) -> str:
        return self.deepseek_model if self.llm_provider == "deepseek" else self.openai_model

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_api_key)

    @property
    def embedding_enabled(self) -> bool:
        """local_hash 为离线降级模式，不需要任何密钥。"""
        if self.embedding_provider == "local_hash":
            return True
        return bool(self.embedding_api_key)

    @property
    def embedding_api_key(self) -> str:
        return {
            "dashscope": self.dashscope_api_key,
            "zhipu": self.zhipu_api_key,
            "openai": self.openai_api_key,
        }.get(self.embedding_provider, "")

    def abs_path(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else BASE_DIR / p


@lru_cache
def get_settings() -> Settings:
    return Settings()
