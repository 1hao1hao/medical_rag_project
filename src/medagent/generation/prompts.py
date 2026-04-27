from typing import Any, List


SYSTEM_PROMPT = """你是一个专业的医疗AI助手。请严格根据用户提供的【参考资料】回答问题。
要求：
1. 回答必须基于参考资料，语言专业、准确、清晰。
2. 如果参考资料不足以支持结论，请直接回答“根据提供的资料，我无法回答此问题”。
3. 严禁使用未在参考资料中出现的先验知识进行猜测或编造。"""


def build_system_prompt() -> str:
    return SYSTEM_PROMPT


def _doc_content(doc: Any) -> str:
    if hasattr(doc, "content"):
        return str(getattr(doc, "content", ""))
    if hasattr(doc, "page_content"):
        return str(getattr(doc, "page_content", ""))
    if isinstance(doc, dict) and "content" in doc:
        return str(doc.get("content", ""))
    return str(doc)


def build_context_text(retrieved_docs: List[Any]) -> str:
    lines = []
    for idx, doc in enumerate(retrieved_docs, 1):
        lines.append("[参考片段 {0}] {1}".format(idx, _doc_content(doc)))
    return "\n".join(lines)


def build_user_prompt(query: str, retrieved_docs: List[Any]) -> str:
    context = build_context_text(retrieved_docs)
    return "【参考资料】\n{0}\n\n【用户问题】\n{1}\n\n请给出回答：".format(context, query)


def build_messages(query: str, retrieved_docs: List[Any]) -> List[dict]:
    return [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": build_user_prompt(query, retrieved_docs)},
    ]
