"""HTTP 接口层：文档管理 + 问答（SSE 流式 / 同步）。"""

from __future__ import annotations

import json
from typing import AsyncIterator

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from app.ingestion.parsers import SUPPORTED_SUFFIX
from app.pipeline import RAGPipeline

router = APIRouter(prefix="/api")

_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(get_settings())
    return _pipeline


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    top_k: int | None = None
    doc_ids: list[str] | None = None
    history: list[dict] | None = None


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.get("/health")
def health():
    s = get_settings()
    return {
        "status": "ok",
        "llm": {"provider": s.llm_provider, "model": s.llm_model, "enabled": s.llm_enabled},
        "embedding": {"provider": s.embedding_provider, "model": s.embedding_model, "enabled": s.embedding_enabled},
        "chunk": {"chunk_size": s.chunk_size, "chunk_overlap": s.chunk_overlap, "top_k": s.top_k},
        "chunks_in_store": get_pipeline().store.count(),
    }


@router.post("/documents")
async def upload_document(file: UploadFile = File(...), chunk_size: int | None = None,
                          chunk_overlap: int | None = None):
    s = get_settings()
    suffix = "." + (file.filename or "").rsplit(".", 1)[-1].lower() if "." in (file.filename or "") else ""
    if suffix not in SUPPORTED_SUFFIX:
        raise HTTPException(status_code=400, detail=f"仅支持 {sorted(SUPPORTED_SUFFIX)} 格式的文档")

    data = await file.read()
    if len(data) > s.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"文件超过 {s.max_upload_mb}MB 限制")

    try:
        meta = get_pipeline().ingest(file.filename or "unnamed", data, chunk_size, chunk_overlap)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"入库失败：{exc}")
    return meta.to_dict()


@router.get("/documents")
def list_documents():
    return {"items": get_pipeline().registry.list(), "chunks_in_store": get_pipeline().store.count()}


@router.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    if not get_pipeline().delete_document(doc_id):
        raise HTTPException(status_code=404, detail="文档不存在")
    return {"deleted": doc_id}


@router.post("/chat")
async def chat_stream(req: ChatRequest):
    """SSE 流式问答：先推引用来源，再逐段推送生成内容。"""

    async def gen() -> AsyncIterator[bytes]:
        for event in get_pipeline().answer_stream(
            req.question, req.history, req.top_k, req.doc_ids
        ):
            yield _sse(event.pop("type"), event).encode("utf-8")

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/chat/sync")
def chat_sync(req: ChatRequest):
    """同步问答，便于脚本调用、ablation 评测与接口调试。"""
    try:
        return get_pipeline().answer(req.question, req.history, req.top_k, req.doc_ids)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"问答失败：{exc}")
