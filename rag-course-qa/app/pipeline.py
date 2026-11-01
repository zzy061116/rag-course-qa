"""RAG 主流程编排：解析 -> 分块 -> 向量化 -> 入库 / 检索 -> 生成。

这一层刻意做成与 Web 框架无关的纯 Python 类，
命令行脚本（消融实验）和 HTTP 接口复用同一份逻辑，保证实验结果与线上行为一致。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.config import Settings, get_settings
from app.generation.llm import LLMClient, offline_answer
from app.generation.prompts import build_messages
from app.ingestion.parsers import parse_file
from app.ingestion.splitter import build_chunks
from app.retrieval.embeddings import EmbeddingClient
from app.retrieval.vectorstore import get_store, new_id


@dataclass
class DocumentMeta:
    doc_id: str
    filename: str
    chunk_size: int
    chunk_overlap: int
    num_pages: int
    num_chunks: int
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "num_pages": self.num_pages,
            "num_chunks": self.num_chunks,
            "created_at": self.created_at,
        }


class DocumentRegistry:
    """轻量文档清单（json 文件），避免为元数据再引入一个数据库。"""

    def __init__(self, settings: Settings):
        self.path: Path = settings.abs_path(settings.upload_dir).parent / "documents.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, items: list[dict]) -> None:
        self.path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

    def add(self, meta: DocumentMeta) -> None:
        items = self._load()
        items.append(meta.to_dict())
        self._save(items)

    def list(self) -> list[dict]:
        return self._load()

    def get(self, doc_id: str) -> dict | None:
        return next((i for i in self._load() if i["doc_id"] == doc_id), None)

    def remove(self, doc_id: str) -> dict | None:
        items = self._load()
        target = next((i for i in items if i["doc_id"] == doc_id), None)
        if target:
            self._save([i for i in items if i["doc_id"] != doc_id])
        return target


class RAGPipeline:
    def __init__(self, settings: Settings | None = None, collection_suffix: str = ""):
        self.s = settings or get_settings()
        self.embedder = EmbeddingClient(self.s)
        self.store = get_store(self.s, collection_suffix)
        self.llm = LLMClient(self.s)
        self.registry = DocumentRegistry(self.s)
        self.upload_dir: Path = self.s.abs_path(self.s.upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- 入库 ----------------
    def ingest(
        self,
        filename: str,
        data: bytes,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> DocumentMeta:
        suffix = Path(filename).suffix.lower()
        chunk_size = chunk_size or self.s.chunk_size
        chunk_overlap = chunk_overlap or self.s.chunk_overlap

        pages = parse_file(suffix, data)
        if not pages:
            raise ValueError("未能从文件中解析出文本内容（可能是扫描版 PDF 或空文件）")

        chunks = build_chunks(pages, source=filename, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        doc_id = new_id()[:12]

        # 分批向量化，避免单次请求过大
        batch_size = 16
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = self.embedder.embed_documents([c.text for c in batch])
            self.store.add(
                ids=[c.chunk_id for c in batch],
                texts=[c.text for c in batch],
                metadatas=[
                    {"doc_id": doc_id, "source": filename, "page": c.page, "chunk_index": c.chunk_index}
                    for c in batch
                ],
                embeddings=vectors,
            )

        # 原始文件落盘，便于后续溯源与重建索引
        (self.upload_dir / f"{doc_id}_{filename}").write_bytes(data)

        meta = DocumentMeta(
            doc_id=doc_id,
            filename=filename,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            num_pages=len(pages),
            num_chunks=len(chunks),
        )
        self.registry.add(meta)
        return meta

    # ---------------- 检索 ----------------
    def retrieve(self, question: str, top_k: int | None = None, doc_ids: list[str] | None = None) -> list[dict]:
        top_k = top_k or self.s.top_k
        embedding = self.embedder.embed_query(question)
        where = {"doc_id": doc_ids[0]} if doc_ids and len(doc_ids) == 1 else None
        hits = self.store.query(embedding, top_k=top_k, where=where)
        # 多文档过滤：Chroma 的 where 不擅长 in 查询，这里在内存侧过滤
        if doc_ids and len(doc_ids) > 1:
            hits = [h for h in hits if h["metadata"].get("doc_id") in doc_ids]
        return [dict(h) for h in hits]

    def delete_document(self, doc_id: str) -> bool:
        meta = self.registry.remove(doc_id)
        if meta:
            self.store.delete({"doc_id": doc_id})
            for file in self.upload_dir.glob(f"{doc_id}_*"):
                file.unlink(missing_ok=True)
        return bool(meta)

    # ---------------- 生成 ----------------
    def answer(self, question: str, history: list[dict] | None = None, top_k: int | None = None,
               doc_ids: list[str] | None = None) -> dict:
        started = time.time()
        hits = self.retrieve(question, top_k=top_k, doc_ids=doc_ids)
        if self.llm.enabled:
            try:
                content = self.llm.chat(build_messages(question, hits, history))
                mode = "llm"
            except Exception as exc:  # 密钥无效 / 限流 / 超时：降级为检索结果，保证接口可用
                content = f"{offline_answer(question, hits)}\n\n[生成失败，已降级为检索结果] {exc}"
                mode = "llm_error"
        else:
            content = offline_answer(question, hits)
            mode = "offline"
        return {
            "answer": content,
            "citations": self._citations(hits),
            "mode": mode,
            "confidence": self._confidence(hits),
            "latency_ms": int((time.time() - started) * 1000),
        }

    def answer_stream(self, question: str, history: list[dict] | None = None,
                      top_k: int | None = None, doc_ids: list[str] | None = None):
        """产出事件字典，由 API 层转成 SSE。"""
        started = time.time()
        try:
            hits = self.retrieve(question, top_k=top_k, doc_ids=doc_ids)
        except Exception as exc:  # 检索阶段失败（密钥错误 / 向量库异常）
            yield {"type": "error", "message": f"检索失败：{exc}"}
            return

        citations = self._citations(hits)
        yield {"type": "meta", "citations": citations, "confidence": self._confidence(hits)}

        if not self.llm.enabled:
            yield {"type": "delta", "content": offline_answer(question, hits)}
            yield {"type": "done", "answer": offline_answer(question, hits), "citations": citations,
                   "mode": "offline", "latency_ms": int((time.time() - started) * 1000)}
            return

        buffer: list[str] = []
        error_msg = ""
        try:
            for piece in self.llm.stream(build_messages(question, hits, history)):
                buffer.append(piece)
                yield {"type": "delta", "content": piece}
        except Exception as exc:
            error_msg = f"生成失败：{exc}"
            yield {"type": "error", "message": error_msg}

        if error_msg:
            # 生成失败也把检索到的原文返回，避免前端只看到报错
            fallback = offline_answer(question, hits)
            yield {"type": "delta", "content": fallback}
            yield {"type": "done", "answer": fallback, "citations": citations,
                   "mode": "llm_error", "error": error_msg,
                   "latency_ms": int((time.time() - started) * 1000)}
            return

        yield {
            "type": "done",
            "answer": "".join(buffer),
            "citations": citations,
            "mode": "llm",
            "latency_ms": int((time.time() - started) * 1000),
        }

    # ---------------- 工具 ----------------
    @staticmethod
    def _citations(hits: list[dict]) -> list[dict]:
        out = []
        for hit in hits:
            meta = hit.get("metadata", {})
            out.append(
                {
                    "source": meta.get("source", "未知"),
                    "page": meta.get("page", "?"),
                    "score": round(float(hit.get("score", 0.0)), 4),
                    "snippet": (hit.get("text", "") or "")[:200],
                }
            )
        return out

    @staticmethod
    def _confidence(hits: list[dict]) -> str:
        if not hits:
            return "低"
        top = float(hits[0].get("score", 0.0))
        if top >= 0.75:
            return "高"
        if top >= 0.45:
            return "中"
        return "低"
