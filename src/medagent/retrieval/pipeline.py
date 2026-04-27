from typing import Any, List, Tuple

from .dense_retriever import DenseRetriever
from .reranker import CrossEncoderReranker
from .types import ScoredDocument


class RetrievalPipeline:
    """检索主流程：先召回，再按需重排。"""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.retriever = DenseRetriever(config)
        self.reranker = CrossEncoderReranker(config)

    def run(self, query: str) -> Tuple[List[ScoredDocument], List[ScoredDocument]]:
        recall_docs = self.retriever.retrieve(
            query=query,
            top_k=self.config.top_k_recall,
        )
        reranked_docs = self.reranker.rerank(
            query=query,
            documents=recall_docs,
            top_k=self.config.top_k_rerank,
        )
        return recall_docs, reranked_docs
