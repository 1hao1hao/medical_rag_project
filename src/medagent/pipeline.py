import os
import time
from typing import Any, Dict, List

from .generation.answer_generator import AnswerGenerator
from .retrieval.pipeline import RetrievalPipeline


def _has_any_file(path: str) -> bool:
    for _root, _dirs, files in os.walk(path):
        if files:
            return True
    return False


def ensure_vector_db_ready(db_dir: str) -> None:
    """检查向量库是否存在且可用。"""

    if not os.path.exists(db_dir):
        raise ValueError(
            "未找到向量库目录: {0}。请先执行索引构建脚本，例如 "
            "`python scripts/build_index.py --config configs/dev_cpu.yaml`".format(
                db_dir
            )
        )
    if not os.path.isdir(db_dir):
        raise ValueError("向量库路径不是目录: {0}".format(db_dir))
    if not _has_any_file(db_dir):
        raise ValueError(
            "向量库目录为空: {0}。请先构建向量库。".format(db_dir)
        )


class BaselineRAGPipeline:
    """Baseline RAG 主流程：检索 + 生成。"""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.retrieval_pipeline = RetrievalPipeline(config)
        self.answer_generator = AnswerGenerator(config)

    def run(self, question: str) -> Dict[str, Any]:
        query = str(question).strip()
        if not query:
            raise ValueError("用户问题不能为空。")

        ensure_vector_db_ready(self.config.db_dir)

        start = time.time()
        recall_docs, top_docs = self.retrieval_pipeline.run(query)
        answer = self.answer_generator.generate_answer(query, top_docs)
        latency_seconds = time.time() - start

        return {
            "question": query,
            "answer": answer,
            "retrieved_contexts": [doc.content for doc in top_docs],
            "scores": [doc.score for doc in top_docs],
            "latency": {
                "seconds": round(latency_seconds, 4),
                "milliseconds": int(latency_seconds * 1000),
            },
            "recall_count": len(recall_docs),
            "returned_count": len(top_docs),
        }


def run_baseline_rag(config: Any, question: str) -> Dict[str, Any]:
    pipeline = BaselineRAGPipeline(config)
    return pipeline.run(question)
