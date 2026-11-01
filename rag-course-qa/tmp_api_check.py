"""临时接口验证脚本（验证后删除）。"""

import json
import pathlib

import httpx

BASE = "http://127.0.0.1:8000"
SAMPLE = """机器学习基础（示例教材片段）

第 1 节 梯度下降
梯度下降是一种一阶优化算法，通过沿损失函数负梯度方向迭代更新参数来最小化目标函数。
更新公式为：theta = theta - alpha * grad(J(theta))，其中 alpha 是学习率。

第 2 节 过拟合与正则化
过拟合指模型在训练集上表现好、在测试集上表现差的现象。
常见缓解手段包括：L1/L2 正则化、Dropout、数据增强、早停（early stopping）。

第 3 节 评估指标
分类任务常用准确率、精确率、召回率和 F1 分数作为评估指标。
"""

p = pathlib.Path("tmp_demo.txt")
p.write_text(SAMPLE, encoding="utf-8")

r = httpx.post(
    f"{BASE}/api/documents",
    files={"file": ("tmp_demo.txt", p.read_bytes(), "text/plain")},
    timeout=120,
)
print("UPLOAD", r.status_code, r.json())

print("DOCS", httpx.get(f"{BASE}/api/documents", timeout=30).json()["chunks_in_store"])

with httpx.stream(
    "POST",
    f"{BASE}/api/chat",
    json={"question": "梯度下降的更新公式是什么？", "top_k": 2},
    timeout=120,
) as resp:
    for line in resp.iter_lines():
        if line.startswith("event:"):
            print("EVENT:", line.strip())
        elif line.startswith("data:"):
            data = json.loads(line[5:])
            print("DATA:", repr(json.dumps(data, ensure_ascii=False))[:260])

sync = httpx.post(
    f"{BASE}/api/chat/sync",
    json={"question": "过拟合怎么缓解？", "top_k": 2},
    timeout=120,
).json()
print("SYNC mode:", sync["mode"], "| confidence:", sync["confidence"],
      "| citations:", [(c["source"], c["page"], c["score"]) for c in sync["citations"]])

p.unlink(missing_ok=True)
