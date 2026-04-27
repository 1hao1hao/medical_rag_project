from typing import Any, List

from .llm_client import BaseLLMClient, create_llm_client
from .prompts import build_messages


class AnswerGenerator:
    """统一的答案生成入口。"""

    def __init__(self, config: Any, llm_client: BaseLLMClient = None) -> None:
        self.config = config
        self.llm_client = llm_client or create_llm_client(config)

    def generate_answer(self, query: str, retrieved_docs: List[Any]) -> str:
        messages = build_messages(query=query, retrieved_docs=retrieved_docs)
        return self.llm_client.generate(messages)


def generate_answer(config: Any, query: str, retrieved_docs: List[Any]) -> str:
    generator = AnswerGenerator(config=config)
    return generator.generate_answer(query=query, retrieved_docs=retrieved_docs)
