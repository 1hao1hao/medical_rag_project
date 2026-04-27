from typing import Any, List, Optional

from .types import ScoredDocument


class CrossEncoderReranker:
    """基于 CrossEncoder 的重排器。"""

    def __init__(self, config: Any) -> None:
        self.config = config
        self._model = None

    def _get_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "缺少重排依赖。请安装 sentence-transformers 后重试。"
            )
        self._model = CrossEncoder(
            self.config.reranker_model,
            device=self.config.retrieval_device,
        )
        return self._model

    def rerank(
        self,
        query: str,
        documents: List[ScoredDocument],
        top_k: Optional[int] = None,
    ) -> List[ScoredDocument]:
        if not documents:
            return []

        k = self.config.top_k_rerank if top_k is None else int(top_k)
        if k <= 0:
            raise ValueError("top_k_rerank 必须大于 0")

        if not self.config.use_reranker:
            # 跳过重排时，直接截断召回结果。
            return documents[:k]

        model = self._get_model()
        cross_input = [[query, doc.content] for doc in documents]
        scores = model.predict(cross_input)

        reranked = []
        for doc, score in zip(documents, scores):
            reranked.append(
                ScoredDocument(
                    content=doc.content,
                    metadata=doc.metadata,
                    score=float(score),
                    source=doc.source,
                )
            )

        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked[:k]
