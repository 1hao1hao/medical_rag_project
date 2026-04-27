from typing import Any, Dict, List, Optional, Sequence, Tuple

from .types import ScoredDocument


def _extract_source(metadata: Dict[str, Any]) -> str:
    if not isinstance(metadata, dict):
        return "unknown"
    for key in ("source", "file_path", "path"):
        value = metadata.get(key)
        if value:
            return str(value)
    return "unknown"


class DenseRetriever:
    """基于 Chroma 的稠密检索器。"""

    def __init__(self, config: Any) -> None:
        self.config = config
        self._vector_db = None

    def _get_vector_db(self):
        if self._vector_db is not None:
            return self._vector_db

        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_community.vectorstores import Chroma
        except ImportError:
            raise ImportError(
                "缺少检索依赖。请安装 langchain-community、transformers 与 chromadb。"
            )

        embeddings = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model,
            model_kwargs={"device": self.config.retrieval_device},
            encode_kwargs={"normalize_embeddings": True},
        )
        self._vector_db = Chroma(
            persist_directory=self.config.db_dir,
            embedding_function=embeddings,
        )
        return self._vector_db

    def _convert_with_scores(
        self, docs_and_scores: Sequence[Tuple[Any, Any]]
    ) -> List[ScoredDocument]:
        converted = []
        for doc, score in docs_and_scores:
            metadata = getattr(doc, "metadata", {}) or {}
            converted.append(
                ScoredDocument(
                    content=str(getattr(doc, "page_content", "")),
                    metadata=metadata,
                    score=float(score),
                    source=_extract_source(metadata),
                )
            )
        return converted

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[ScoredDocument]:
        k = self.config.top_k_recall if top_k is None else int(top_k)
        if k <= 0:
            raise ValueError("top_k_recall 必须大于 0")

        vector_db = self._get_vector_db()

        docs_with_scores = None
        if hasattr(vector_db, "similarity_search_with_relevance_scores"):
            try:
                docs_with_scores = vector_db.similarity_search_with_relevance_scores(
                    query,
                    k=k,
                )
            except Exception:
                docs_with_scores = None

        if docs_with_scores is None and hasattr(vector_db, "similarity_search_with_score"):
            try:
                docs_with_scores = vector_db.similarity_search_with_score(query, k=k)
            except Exception:
                docs_with_scores = None

        if docs_with_scores:
            return self._convert_with_scores(docs_with_scores)

        docs = vector_db.similarity_search(query, k=k)
        fallback = []
        for doc in docs:
            metadata = getattr(doc, "metadata", {}) or {}
            fallback.append(
                ScoredDocument(
                    content=str(getattr(doc, "page_content", "")),
                    metadata=metadata,
                    score=0.0,
                    source=_extract_source(metadata),
                )
            )
        return fallback
