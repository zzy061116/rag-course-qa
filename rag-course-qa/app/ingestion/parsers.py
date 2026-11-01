"""文档解析层：PDF / Word / 纯文本 -> 带页码的页面列表。

页码是本项目引用溯源的基础，因此 PDF 按物理页解析；
docx / txt 没有稳定的物理页概念，用"虚拟页"近似（每 PAGE_CHARS 字记为一页）。
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

SUPPORTED_SUFFIX = {".pdf", ".docx", ".txt", ".md"}
PAGE_CHARS = 1200  # docx / txt 虚拟页大小


@dataclass
class Page:
    page: int
    text: str


def parse_pdf(data: bytes | str | Path) -> list[Page]:
    reader = PdfReader(data if isinstance(data, (str, Path)) else io.BytesIO(data))
    pages: list[Page] = []
    for idx, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:  # 加密页 / 损坏页跳过
            text = ""
        text = _clean(text)
        if text:
            pages.append(Page(page=idx, text=text))
    return pages


def parse_docx(data: bytes | str | Path) -> list[Page]:
    import docx  # 延迟导入，减少启动开销

    document = docx.Document(data if isinstance(data, (str, Path)) else io.BytesIO(data))
    buffer, pages, page_no = [], [], 1
    for para in document.paragraphs:
        line = para.text.strip()
        if line:
            buffer.append(line)
        if sum(len(x) for x in buffer) >= PAGE_CHARS:
            pages.append(Page(page=page_no, text="\n".join(buffer)))
            page_no += 1
            buffer = []
    if buffer:
        pages.append(Page(page=page_no, text="\n".join(buffer)))
    return pages


def parse_text(data: bytes | str | Path) -> list[Page]:
    if isinstance(data, bytes):
        raw = data.decode("utf-8", errors="ignore")
    elif isinstance(data, (str, Path)) and Path(data).exists():
        raw = Path(data).read_text(encoding="utf-8", errors="ignore")
    else:
        raw = str(data)
    raw = _clean(raw)
    pages, page_no = [], 1
    for start in range(0, len(raw), PAGE_CHARS):
        chunk = raw[start : start + PAGE_CHARS].strip()
        if chunk:
            pages.append(Page(page=page_no, text=chunk))
            page_no += 1
    return pages


def _clean(text: str) -> str:
    """去掉多余空白，保留段落结构，避免切分时全是噪声。"""
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def parse_file(suffix: str, data: bytes | str | Path) -> list[Page]:
    suffix = suffix.lower()
    if suffix not in SUPPORTED_SUFFIX:
        raise ValueError(f"不支持的文件类型：{suffix}，仅支持 {'/'.join(sorted(SUPPORTED_SUFFIX))}")
    if suffix == ".pdf":
        return parse_pdf(data)
    if suffix == ".docx":
        return parse_docx(data)
    return parse_text(data)
