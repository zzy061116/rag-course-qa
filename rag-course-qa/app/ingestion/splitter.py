"""文本分块层：递归字符切分 + 页码元数据保留。

为什么要自己写而不是直接用 LangChain：
1. 需要把"页码"作为一等公民传给每个 chunk，默认切分器不保留跨切分的页面信息；
2. 分块消融实验要求切分逻辑可参数化、可复现、可单独调用。

切分策略（自顶向下递归）：
先按段落切 -> 段落过长按句号切 -> 再长按逗号切 -> 最后按字硬切；
切完再做"贪心合并"，把小碎片拼到接近 chunk_size，并用 chunk_overlap 保留上下文重叠。
"""

from __future__ import annotations

from dataclasses import dataclass

from app.ingestion.parsers import Page

SEPARATORS = ["\n\n", "\n", "。", "；", "，", "、", " ", ""]


@dataclass
class Chunk:
    text: str
    source: str
    page: int
    chunk_index: int

    @property
    def chunk_id(self) -> str:
        return f"{self.source}::p{self.page}::c{self.chunk_index}"


def _split_by(text: str, sep: str) -> list[str]:
    """按分隔符切分并保留分隔符，避免丢标点。"""
    if sep == "":
        return list(text)
    parts = text.split(sep)
    return [p + sep for p in parts[:-1]] + [parts[-1]]


def _recursive_split(text: str, seps: list[str], chunk_size: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size or not seps:
        return [text]
    sep, rest = seps[0], seps[1:]
    out: list[str] = []
    for part in _split_by(text, sep):
        part = part.strip()
        if not part:
            continue
        if len(part) <= chunk_size:
            out.append(part)
        else:
            out.extend(_recursive_split(part, rest, chunk_size))
    return out


def _merge(pieces: list[str], chunk_size: int, overlap: int) -> list[str]:
    """贪心合并碎片，块间保留 overlap 长度的重叠，缓解切断语义的问题。"""
    merged: list[str] = []
    current = ""
    for piece in pieces:
        if not current:
            current = piece
            continue
        if len(current) + len(piece) <= chunk_size:
            current += piece
        else:
            merged.append(current)
            tail = current[-overlap:] if overlap > 0 else ""
            current = (tail + piece) if tail else piece
    if current.strip():
        merged.append(current)
    return [m.strip() for m in merged if m.strip()]


def split_page_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    overlap = max(0, min(overlap, chunk_size // 2))
    pieces = _recursive_split(text, SEPARATORS, chunk_size)
    return _merge(pieces, chunk_size, overlap)


def build_chunks(pages: list[Page], source: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """按页切分：每个 chunk 只归属一个页码，引用溯源才准确。"""
    chunks: list[Chunk] = []
    for page in pages:
        for piece in split_page_text(page.text, chunk_size, chunk_overlap):
            chunks.append(
                Chunk(text=piece, source=source, page=page.page, chunk_index=len(chunks))
            )
    return chunks
