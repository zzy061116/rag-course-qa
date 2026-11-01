"""向量库层：Chroma 持久化存储（带内存降级后端）。

对外只暴露统一的 add / query / delete 接口，
上层业务不感知底层是 Chroma 还是内存实现，方便本地调试与云端部署切换。
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from app.config import Settings, get_settings


class Hit(dict):
    """一条检索结果：text / metadata / score（越大越相关）。"""


class BaseStore:
    def add(self, ids: list[str], texts: list[str], metadatas: list[dict], embeddings: list[list[float]]) -> None:
        raise NotImplementedError

    def query(self, embedding: list[float], top_k: int, where: dict | None = None) -> list[Hit]:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def delete(self, where: dict) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


class ChromaStore(BaseStore):
    def __init__(self, settings: Settings, suffix: str = ""):
        import chromadb

        self.settings = settings
        self.path = settings.abs_path(settings.chroma_dir)
        self.path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.path))
        self.name = f"{settings.collection}{suffix}"
        self.collection = self.client.get_or_create_collection(
            name=self.name, metadata={"hnsw:space": "cosine"}
        )

    def add(self, ids, texts, metadatas, embeddings):
        self.collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)

    def query(self, embedding, top_k, where=None):
        res = self.collection.query(
            query_embeddings=[embedding], n_results=top_k, where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[Hit] = []
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            hits.append(Hit(text=doc, metadata=meta or {}, score=float(1 - dist)))
        return hits

    def count(self) -> int:
        return self.collection.count()

    def delete(self, where: dict) -> None:
        self.collection.delete(where=where)

    def reset(self) -> None:
        self.client.delete_collection(self.name)
        self.collection = self.client.get_or_create_collection(
            name=self.name, metadata={"hnsw:space": "cosine"}
        )


class MemoryStore(BaseStore):
    """纯 numpy 的内存向量库（带 JSON 持久化），用于无 Chroma 环境下的降级。"""

    def __init__(self, settings: Settings, suffix: str = ""):
        self.path: Path = settings.abs_path(settings.chroma_dir) / f"memory{suffix}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict[str, Any]] = []
        if self.path.exists():
            self.rows = json.loads(self.path.read_text(encoding="utf-8"))

    def add(self, ids, texts, metadatas, embeddings):
        for _id, text, meta, emb in zip(ids, texts, metadatas, embeddings):
            self.rows.append({"id": _id, "text": text, "metadata": meta, "embedding": emb})
        self._flush()

    def query(self, embedding, top_k, where=None):
        if not self.rows:
            return []
        matrix = np.array([r["embedding"] for r in self.rows], dtype=np.float32)
        query = np.array(embedding, dtype=np.float32)
        scores = matrix @ query / (
            np.linalg.norm(matrix, axis=1) * (np.linalg.norm(query) + 1e-9) + 1e-9
        )
        order = np.argsort(-scores)[: top_k * 3]
        hits = []
        for idx in order:
            row = self.rows[int(idx)]
            if where and any(row["metadata"].get(k) != v for k, v in where.items()):
                continue
            hits.append(Hit(text=row["text"], metadata=row["metadata"], score=float(scores[int(idx)])))
            if len(hits) >= top_k:
                break
        return hits

    def count(self) -> int:
        return len(self.rows)

    def delete(self, where: dict) -> None:
        self.rows = [
            r for r in self.rows
            if any(r["metadata"].get(k) != v for k, v in where.items())
        ]
        self._flush()

    def reset(self) -> None:
        self.rows = []
        self._flush()

    def _flush(self) -> None:
        self.path.write_text(
            json.dumps(self.rows, ensure_ascii=False), encoding="utf-8"
        )


def get_store(settings: Settings | None = None, suffix: str = "", backend: str = "auto") -> BaseStore:
    settings = settings or get_settings()
    if backend == "memory":
        return MemoryStore(settings, suffix)
    try:
        return ChromaStore(settings, suffix)
    except Exception:  # Chroma 不可用时自动降级，保证服务可用
        return MemoryStore(settings, suffix)


def new_id() -> str:
    return uuid.uuid4().hex
