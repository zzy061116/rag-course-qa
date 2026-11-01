# 课程教材 RAG 知识库问答系统

面向**课程教材 / 课件**的私有文档问答系统。上传 PDF、Word 教材后自动完成解析、分块、向量化与入库；提问时先检索原文片段，再交由大模型生成回答，并**强制给出原文页码引用**，用于缓解专业教材场景下的幻觉问题。

## 核心特性

- **完整 RAG 链路**：解析 → 分块 → 向量化 → 检索 → 生成 → 溯源，全部自研可控
- **页码级引用**：每个 chunk 只归属一个物理页，回答可精确到 `第 X 页`
- **抗幻觉 Prompt**：禁止使用外部知识，资料不足时明确回答"文档中未提及"，并输出置信度
- **SSE 流式输出**：先返回引用来源，再逐段流式返回答案（打字机效果）
- **分块消融实验**：脚本化对比 256 / 512 / 1024 chunk_size，量化召回质量
- **组件可替换**：LLM（DeepSeek / OpenAI 兼容）与 Embedding（通义 / 智谱 / OpenAI / 离线哈希）解耦

## 架构

```
       ┌──── 入库链路 ────────────────────────────────┐
PDF/Word ──▶ 文档解析(带页码) ──▶ 递归分块 ──▶ Embedding ──▶ Chroma 向量库
                                                                  │
       ┌──── 问答链路 ────────────────────────────────┐            │
用户提问 ──▶ Query 向量化 ──▶ Top-K 检索 ◀────────────────────────┘
                                 │
                                 ▼
                    带引用约束的 Prompt ──▶ DeepSeek ──▶ SSE 流式回答（含页码引用）
```

## 目录结构

```
rag-course-qa/
├── app/
│   ├── config.py               # 统一配置（环境变量注入）
│   ├── pipeline.py             # RAG 主流程编排（入库/检索/生成）
│   ├── ingestion/              # 解析层（PDF/Word/TXT，保留页码）+ 分块层
│   ├── retrieval/              # Embedding 客户端 + 向量库（Chroma/内存降级）
│   ├── generation/             # Prompt 模板 + LLM 流式调用
│   └── api/routes.py           # HTTP 接口（上传 / 问答 / SSE）
├── web/index.html              # 简易前端（上传、流式对话、引用展示）
├── scripts/
│   ├── smoke_test.py           # 无密钥端到端冒烟测试
│   ├── ablation.py             # 分块消融实验
│   └── eval_questions.example.json
├── Dockerfile / docker-compose.yml
└── requirements.txt
```

## 快速开始

```bash
# 1. 创建虚拟环境（建议 Python 3.11 / 3.12）
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置密钥
cp .env.example .env        # Windows: copy .env.example .env
#    填写 DEEPSEEK_API_KEY 与 DASHSCOPE_API_KEY

# 4. 无密钥也能先跑通链路（离线哈希向量 + 检索摘要）
python scripts/smoke_test.py

# 5. 启动服务
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

打开 <http://localhost:8000> 使用页面，接口文档在 <http://localhost:8000/docs>。

## 环境变量

| 变量 | 说明 | 默认值 |
| --- | --- | --- |
| `LLM_PROVIDER` | `deepseek` 或 `openai` | `deepseek` |
| `DEEPSEEK_API_KEY` | DeepSeek 对话密钥（必填，否则走离线检索模式） | - |
| `DEEPSEEK_MODEL` | 对话模型 | `deepseek-chat` |
| `EMBEDDING_PROVIDER` | `dashscope` / `zhipu` / `openai` / `local_hash` | `dashscope` |
| `DASHSCOPE_API_KEY` | 通义 text-embedding-v3 密钥 | - |
| `EMBEDDING_DIM` | 向量维度（通义 v3 支持 1024/768/512） | `1024` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 分块大小与重叠 | `512` / `64` |
| `TOP_K` | 每次召回片段数 | `4` |
| `CHROMA_DIR` / `UPLOAD_DIR` | 向量库与原始文件目录 | `./data/...` |

> **注意**：DeepSeek 官方只提供对话接口，**不提供 Embedding 接口**，所以向量化单独接了通义/智谱等服务商；两者在代码里完全解耦，可任意替换。

## HTTP 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/health` | 服务与模型配置状态、当前片段数 |
| POST | `/api/documents` | 上传文档（multipart），自动解析入库 |
| GET | `/api/documents` | 文档列表 |
| DELETE | `/api/documents/{doc_id}` | 删除文档及其向量 |
| POST | `/api/chat` | **SSE 流式问答**，事件：`meta` / `delta` / `done` / `error` |
| POST | `/api/chat/sync` | 同步问答（脚本与评测用） |

## 分块消融实验

```bash
python scripts/ablation.py \
  --file data/uploads/机器学习.pdf \
  --sizes 256 512 1024 \
  --top-k 4 \
  --questions scripts/eval_questions.example.json \
  --llm-judge            # 可选：额外让大模型给召回质量打 1-5 分
```

输出终端表格并生成 `reports/ablation.md`，指标包括：命中率、关键词召回率、平均相似度、平均片段长度、入库耗时、LLM 评分。

## Docker 部署

```bash
cp .env.example .env && vi .env     # 填入密钥
docker compose up -d --build
docker compose logs -f
curl http://localhost:8000/api/health
```

数据（向量库 + 上传文件）持久化在宿主机 `./data` 目录。

## 常见问题

| 问题 | 原因与处理 |
| --- | --- |
| 回答出现"离线模式" | 未配置 `DEEPSEEK_API_KEY`，系统只返回检索片段 |
| 上传后片段数为 0 | 扫描版 PDF 没有文本层，需要先用 OCR 处理 |
| Embedding 报 401 | 检查 `EMBEDDING_PROVIDER` 与对应密钥是否匹配 |
| Chroma 安装失败 | 会自动降级为内存向量库（功能可用，重启后需重新入库） |

## 后续迭代

- [ ] 接入 BGE-Rerank 重排，提升 Top-K 精准度
- [ ] 原文溯源高亮（命中片段在原文中的定位）
- [ ] Function Calling：课表 / 成绩查询工具调用
- [ ] Vue3 + TypeScript 前端（多轮会话、会话本地存储）
