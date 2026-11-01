"""分块消融实验：量化 chunk_size 对检索质量的影响。

为什么做这个实验：
chunk 太小 -> 语义被切断，检索命中但信息不全；
chunk 太大 -> 噪声多，向量被稀释，召回不精准。
只有量化对比，才能给"为什么选 512"一个站得住脚的理由。

用法：
    python scripts/ablation.py --file data/uploads/xxx.pdf \
        --sizes 256 512 1024 --top-k 4 --questions scripts/eval_questions.json [--llm-judge]

输出：终端表格 + reports/ablation.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.ingestion.parsers import parse_file  # noqa: E402
from app.ingestion.splitter import build_chunks  # noqa: E402
from app.pipeline import RAGPipeline  # noqa: E402

DEFAULT_QUESTIONS = [
    {"question": "梯度下降法的更新公式是什么？", "keywords": ["梯度", "下降", "学习率"]},
    {"question": "过拟合有哪些常见解决办法？", "keywords": ["过拟合", "正则", "dropout"]},
]


def load_questions(path: str | None) -> list[dict]:
    if not path:
        return DEFAULT_QUESTIONS
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"评测问题文件不存在：{p}")
    return json.loads(p.read_text(encoding="utf-8"))


def evaluate(pipeline: RAGPipeline, questions: list[dict], top_k: int, llm_judge: bool) -> dict:
    keyword_recall, hit_rates, sims, judged = [], [], [], []

    for item in questions:
        hits = pipeline.retrieve(item["question"], top_k=top_k)
        if hits:
            sims.append(statistics.mean(h["score"] for h in hits))
        else:
            sims.append(0.0)

        keywords = item.get("keywords", [])
        if keywords:
            matched = 0
            for kw in keywords:
                if any(kw in (h.get("text") or "") for h in hits):
                    matched += 1
            keyword_recall.append(matched / len(keywords))
            hit_rates.append(1.0 if matched > 0 else 0.0)

        if llm_judge and pipeline.llm.enabled and hits:
            judged.append(llm_score(pipeline, item["question"], hits))

    return {
        "keyword_recall": statistics.mean(keyword_recall) if keyword_recall else 0.0,
        "hit_rate": statistics.mean(hit_rates) if hit_rates else 0.0,
        "mean_similarity": statistics.mean(sims) if sims else 0.0,
        "llm_score": statistics.mean(judged) if judged else None,
    }


def llm_score(pipeline: RAGPipeline, question: str, hits: list[dict]) -> float:
    """让模型给"检索片段对回答该问题的有用程度"打 1-5 分。"""
    context = "\n".join(f"[p.{h['metadata'].get('page')}] {h['text'][:300]}" for h in hits)
    prompt = (
        "你是检索质量评审员。请判断下列检索片段对回答问题的有用程度，"
        "只输出 1 到 5 的整数（1=完全无关，5=完全包含答案），不要输出其他内容。\n\n"
        f"问题：{question}\n\n检索片段：\n{context}"
    )
    try:
        raw = pipeline.llm.chat([{"role": "user", "content": prompt}]).strip()
        return float("".join(ch for ch in raw if ch.isdigit())[:1] or 0)
    except Exception:
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 分块消融实验")
    parser.add_argument("--file", required=True, help="待评测的教材/课件文件")
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--overlap-ratio", type=float, default=0.125, help="重叠长度 = chunk_size * ratio")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--questions", help="评测问题 json 路径")
    parser.add_argument("--llm-judge", action="store_true", help="额外用大模型对召回质量打分")
    args = parser.parse_args()

    src = Path(args.file)
    if not src.exists():
        raise SystemExit(f"文件不存在：{src}")
    questions = load_questions(args.questions)
    settings = get_settings()

    results = []
    for size in args.sizes:
        overlap = int(size * args.overlap_ratio)
        pipeline = RAGPipeline(settings, collection_suffix=f"_cs{size}")
        pipeline.store.reset()

        started = time.time()
        meta = pipeline.ingest(src.name, src.read_bytes(), chunk_size=size, chunk_overlap=overlap)
        ingest_sec = time.time() - started

        pages = parse_file(src.suffix, src.read_bytes())
        chunks = build_chunks(pages, src.name, size, overlap)
        avg_chunk_chars = round(sum(len(c.text) for c in chunks) / max(len(chunks), 1), 1)
        metrics = evaluate(pipeline, questions, args.top_k, args.llm_judge)

        results.append(
            {
                "chunk_size": size,
                "chunk_overlap": overlap,
                "num_chunks": meta.num_chunks,
                "num_pages": meta.num_pages,
                "avg_chunk_chars": avg_chunk_chars,
                "ingest_sec": round(ingest_sec, 2),
                **metrics,
            }
        )
        print(f"[chunk_size={size}] 片段数={meta.num_chunks} 命中率={metrics['hit_rate']:.2%} "
              f"关键词召回={metrics['keyword_recall']:.2%} 平均相似度={metrics['mean_similarity']:.3f}")

    # 输出报告
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    lines = [
        "# 分块策略消融实验报告",
        "",
        f"- 文档：`{src.name}`（{results[0]['num_pages']} 页）",
        f"- 检索参数：top_k={args.top_k}，重叠比例={args.overlap_ratio}",
        f"- 评测问题数：{len(questions)}",
        "",
        "| chunk_size | overlap | 片段数 | 平均片段长度(字) | 命中率 | 关键词召回率 | 平均相似度 | LLM 评分 | 入库耗时(s) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        lines.append(
            f"| {r['chunk_size']} | {r['chunk_overlap']} | {r['num_chunks']} | {r['avg_chunk_chars']} | "
            f"{r['hit_rate']:.2%} | {r['keyword_recall']:.2%} | {r['mean_similarity']:.3f} | "
            f"{r['llm_score'] if r['llm_score'] is not None else '-'} | {r['ingest_sec']} |"
        )
    best = max(results, key=lambda r: (r["keyword_recall"], r["hit_rate"]))
    lines += [
        "",
        f"结论：当前文档下 chunk_size={best['chunk_size']} 表现最好"
        f"（关键词召回率 {best['keyword_recall']:.2%}）。",
        "注：结论随文档类型变化，教材类长句文档通常偏向 512～1024，课件类短句文档偏向 256～512。",
    ]
    (reports / "ablation.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告已写入：{reports / 'ablation.md'}")


if __name__ == "__main__":
    main()
