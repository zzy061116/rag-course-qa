"""端到端冒烟测试：不需要任何云 API 密钥即可跑通整条链路。

会临时把 EMBEDDING_PROVIDER 切成 local_hash（确定性哈希向量），
验证：解析 -> 分块 -> 向量化 -> 入库 -> 检索 -> 回答，全部环节无异常。

用法：
    python scripts/smoke_test.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("EMBEDDING_PROVIDER", "local_hash")

from app.config import Settings  # noqa: E402
from app.pipeline import RAGPipeline  # noqa: E402

SAMPLE = """机器学习基础（示例教材片段）

第 1 节 梯度下降
梯度下降是一种一阶优化算法，通过沿损失函数负梯度方向迭代更新参数来最小化目标函数。
更新公式为：theta = theta - alpha * grad(J(theta))，其中 alpha 是学习率。
学习率过大可能导致震荡不收敛，过小则收敛速度慢。

第 2 节 过拟合与正则化
过拟合指模型在训练集上表现好、在测试集上表现差的现象。
常见缓解手段包括：L1/L2 正则化、Dropout、数据增强、早停（early stopping）。

第 3 节 评估指标
分类任务常用准确率、精确率、召回率和 F1 分数作为评估指标。
"""


def main() -> None:
    settings = Settings(embedding_provider="local_hash", chroma_dir="./data/chroma_smoke")
    pipeline = RAGPipeline(settings, collection_suffix="_smoke")
    pipeline.store.reset()

    meta = pipeline.ingest("smoke_sample.txt", SAMPLE.encode("utf-8"), chunk_size=256, chunk_overlap=32)
    print(f"入库完成：{meta.num_pages} 页 → {meta.num_chunks} 个片段")

    question = "梯度下降的更新公式是什么？"
    hits = pipeline.retrieve(question, top_k=2)
    print(f"检索命中 {len(hits)} 条，最高相关度 {hits[0]['score'] if hits else 0:.3f}")
    for hit in hits:
        print(f"  - 第 {hit['metadata']['page']} 页：{hit['text'][:60]}…")

    result = pipeline.answer(question)
    print(f"\n回答模式：{result['mode']}，置信度：{result['confidence']}，耗时 {result['latency_ms']}ms")
    print(result["answer"][:300])

    assert meta.num_chunks > 0 and hits, "冒烟测试失败：链路未跑通"
    print("\n冒烟测试通过：整条 RAG 链路可用")


if __name__ == "__main__":
    main()
