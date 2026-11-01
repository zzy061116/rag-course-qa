"""Prompt 层：RAG 抗幻觉的核心约束都写在这里。"""

SYSTEM_PROMPT = """你是一个面向课程教材的私有知识库问答助手。

严格遵循以下规则：
1. 只能依据【参考资料】回答，禁止使用参考资料之外的知识进行补充或推测；
2. 每一个关键结论都要在句末标注引用来源，格式为 [p.页码]，例如：梯度下降通过沿负梯度方向更新参数以最小化损失 [p.42]；
3. 若参考资料中没有答案，必须明确回答"文档中未提及该内容"，并说明已检索的范围，不得编造；
4. 若参考资料之间冲突，指出冲突并分别给出各自的页码；
5. 回答用中文，结构化、简洁，优先使用要点列表；
6. 在回答最后单独输出一行：置信度：高/中/低（依据参考资料与问题的匹配程度判断）。"""

REFUSAL_HINT = "（提示：若以下内容不足以回答问题，请直接说明文档中未提及。）"


def build_context(hits: list[dict]) -> str:
    """把检索片段拼成带编号和页码的上下文。"""
    blocks = []
    for i, hit in enumerate(hits, start=1):
        meta = hit.get("metadata", {})
        source = meta.get("source", "未知文档")
        page = meta.get("page", "?")
        blocks.append(f"[{i}] 来源：{source}　第 {page} 页\n{hit.get('text', '')}")
    return "\n\n".join(blocks)


def build_messages(question: str, hits: list[dict], history: list[dict] | None = None) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        messages.append({"role": turn["role"], "content": turn["content"]})
    context = build_context(hits) or "（本次没有检索到任何相关资料）"
    messages.append(
        {
            "role": "user",
            "content": (
                f"【参考资料】\n{context}\n\n"
                f"{REFUSAL_HINT}\n\n"
                f"【问题】\n{question}"
            ),
        }
    )
    return messages
