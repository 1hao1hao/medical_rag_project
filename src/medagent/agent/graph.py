from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from medagent.generation.answer_generator import AnswerGenerator
from medagent.retrieval.evidence_evaluator import EvidenceEvaluator
from medagent.retrieval.pipeline import RetrievalPipeline
from medagent.retrieval.types import ScoredDocument

from .query_analyzer import QueryAnalyzer


class AgentState:
    """轻量 Agentic RAG 状态。"""

    __slots__ = (
        "question",
        "current_query",
        "trace",
        "analysis",
        "decision",
        "answer",
        "retrieved_docs",
    )

    def __init__(self, question: str) -> None:
        self.question = question
        self.current_query = question
        self.trace = []
        self.analysis = {}
        self.decision = {}
        self.answer = ""
        self.retrieved_docs = []

    def add_trace(self, step: str, payload: Dict[str, Any]) -> None:
        self.trace.append({"step": step, "payload": payload})


class AgenticRAGGraph:
    """轻量状态机：analyze -> retrieve -> evaluate -> generate/refuse。"""

    def __init__(
        self,
        config: Any,
        query_analyzer: Any = None,
        retrieval_pipeline: Any = None,
        evidence_evaluator: Any = None,
        answer_generator: Any = None,
        retrieval_fn: Optional[Callable[[str, int], Tuple[List[Any], List[Any]]]] = None,
    ) -> None:
        self.config = config
        self.query_analyzer = query_analyzer or QueryAnalyzer()
        self.retrieval_pipeline = retrieval_pipeline or RetrievalPipeline(config)
        self.evidence_evaluator = evidence_evaluator or EvidenceEvaluator(config)
        self.answer_generator = answer_generator or AnswerGenerator(config)
        self.retrieval_fn = retrieval_fn
        self.max_retrieval_attempts = int(getattr(config, "max_retrieval_attempts", 2))
        if self.max_retrieval_attempts <= 0:
            self.max_retrieval_attempts = 2

    def _analysis_to_dict(self, analysis: Any) -> Dict[str, Any]:
        if hasattr(analysis, "to_dict"):
            return analysis.to_dict()
        if isinstance(analysis, dict):
            return analysis
        raise ValueError("analyze_query 返回结果格式不支持")

    def _run_retrieval(self, query: str, recall_k: int) -> Tuple[List[Any], List[Any]]:
        if self.retrieval_fn is not None:
            return self.retrieval_fn(query, recall_k)

        # 默认复用已实现的 retriever + reranker，支持动态 recall_k。
        retriever = getattr(self.retrieval_pipeline, "retriever", None)
        reranker = getattr(self.retrieval_pipeline, "reranker", None)
        if retriever is not None and reranker is not None:
            recall_docs = retriever.retrieve(query=query, top_k=recall_k)
            top_docs = reranker.rerank(
                query=query,
                documents=recall_docs,
                top_k=int(getattr(self.config, "top_k_rerank", 3)),
            )
            return recall_docs, top_docs

        # 兜底：调用 pipeline.run（无法覆盖 recall_k 时使用配置默认值）。
        if hasattr(self.retrieval_pipeline, "run"):
            return self.retrieval_pipeline.run(query)
        raise RuntimeError("retrieval_pipeline 不可用")

    def _evaluate(self, question: str, docs: Sequence[Any]) -> Dict[str, Any]:
        evaluator = self.evidence_evaluator
        if hasattr(evaluator, "evaluate"):
            return evaluator.evaluate(question, docs)
        if callable(evaluator):
            return evaluator(question, docs)
        raise RuntimeError("evidence_evaluator 不可用")

    def _generate_answer(self, question: str, docs: Sequence[Any]) -> str:
        generator = self.answer_generator
        if hasattr(generator, "generate_answer"):
            return generator.generate_answer(question, list(docs))
        if callable(generator):
            return generator(question, list(docs))
        raise RuntimeError("answer_generator 不可用")

    def _build_refusal_answer(self, analysis: Dict[str, Any], reason: str) -> str:
        if str(analysis.get("risk_level", "")).lower() == "high":
            return (
                "根据当前信息，我无法安全给出具体用药建议。"
                "你描述了高风险症状，请立即线下就医或呼叫急救。"
                "原因：{0}".format(reason)
            )
        return "根据当前检索证据不足，我暂时无法回答此问题。原因：{0}".format(reason)

    def _docs_to_contexts(self, docs: Sequence[Any]) -> List[Dict[str, Any]]:
        contexts = []
        for doc in docs:
            if isinstance(doc, ScoredDocument):
                contexts.append(doc.to_dict())
            else:
                contexts.append(
                    {
                        "content": str(getattr(doc, "content", getattr(doc, "page_content", ""))),
                        "metadata": dict(getattr(doc, "metadata", {}) or {}),
                        "score": float(getattr(doc, "score", 0.0) or 0.0),
                        "source": str(getattr(doc, "source", "unknown")),
                    }
                )
        return contexts

    def run(self, question: str) -> Dict[str, Any]:
        text = str(question or "").strip()
        if not text:
            raise ValueError("question 不能为空")

        state = AgentState(question=text)

        analysis_obj = self.query_analyzer.analyze(text)
        analysis = self._analysis_to_dict(analysis_obj)
        state.analysis = analysis
        state.add_trace("analyze_query", analysis)

        if not bool(analysis.get("need_retrieval", True)):
            answer = self._generate_answer(text, [])
            state.answer = answer
            state.decision = {
                "status": "sufficient",
                "action": "generate",
                "reason": "分析结果显示无需检索，直接生成回答。",
                "attempts": 0,
                "final_query": state.current_query,
                "risk_level": analysis.get("risk_level", "low"),
                "query_type": analysis.get("query_type", "general"),
            }
            state.add_trace("finalize", state.decision)
            return {
                "answer": state.answer,
                "trace": state.trace,
                "contexts": [],
                "decision": state.decision,
            }

        base_recall_k = int(getattr(self.config, "top_k_recall", 15))
        rewrite_used = False
        expand_used = False
        attempts = 0

        while attempts < self.max_retrieval_attempts:
            attempts += 1
            recall_k = base_recall_k * (2 if expand_used else 1)
            state.add_trace(
                "retrieve",
                {
                    "attempt": attempts,
                    "query": state.current_query,
                    "top_k_recall": recall_k,
                },
            )

            try:
                recall_docs, top_docs = self._run_retrieval(state.current_query, recall_k)
            except Exception as exc:
                state.add_trace(
                    "retrieve_error",
                    {"attempt": attempts, "error": str(exc)},
                )
                recall_docs, top_docs = [], []

            state.retrieved_docs = list(top_docs)
            state.add_trace(
                "retrieve_result",
                {
                    "attempt": attempts,
                    "recall_count": len(recall_docs),
                    "top_count": len(top_docs),
                },
            )

            decision = self._evaluate(state.current_query, top_docs)
            state.decision = dict(decision)
            state.add_trace("evaluate_evidence", state.decision)

            action = str(decision.get("action", "refuse"))
            if action == "generate":
                state.answer = self._generate_answer(state.current_query, top_docs)
                break

            if action == "refuse":
                state.answer = self._build_refusal_answer(
                    analysis=analysis,
                    reason=str(decision.get("reason", "证据不足")),
                )
                break

            if action == "rewrite_and_retrieve":
                if rewrite_used or attempts >= self.max_retrieval_attempts:
                    state.decision = {
                        "status": "insufficient",
                        "action": "refuse",
                        "reason": "改写重试次数已达上限。",
                    }
                    state.answer = self._build_refusal_answer(
                        analysis=analysis,
                        reason=state.decision["reason"],
                    )
                    break
                rewritten = analysis.get("rewritten_query")
                if not rewritten:
                    rewritten = state.current_query + "（请明确疾病背景、症状与目标）"
                state.current_query = str(rewritten)
                rewrite_used = True
                state.add_trace(
                    "rewrite_query",
                    {"new_query": state.current_query, "attempt": attempts},
                )
                continue

            if action == "expand_retrieval":
                if expand_used or attempts >= self.max_retrieval_attempts:
                    state.decision = {
                        "status": "insufficient",
                        "action": "refuse",
                        "reason": "扩召回重试次数已达上限。",
                    }
                    state.answer = self._build_refusal_answer(
                        analysis=analysis,
                        reason=state.decision["reason"],
                    )
                    break
                expand_used = True
                state.add_trace(
                    "expand_retrieval",
                    {"new_top_k_recall": base_recall_k * 2, "attempt": attempts},
                )
                continue

            # 未知动作：保护性拒答
            state.decision = {
                "status": "insufficient",
                "action": "refuse",
                "reason": "未知决策动作: {0}".format(action),
            }
            state.answer = self._build_refusal_answer(
                analysis=analysis,
                reason=state.decision["reason"],
            )
            break

        if not state.answer:
            state.decision = {
                "status": "insufficient",
                "action": "refuse",
                "reason": "已达到最大检索次数，仍无法生成可靠答案。",
            }
            state.answer = self._build_refusal_answer(
                analysis=analysis,
                reason=state.decision["reason"],
            )

        state.decision.update(
            {
                "attempts": attempts,
                "final_query": state.current_query,
                "risk_level": analysis.get("risk_level", "low"),
                "query_type": analysis.get("query_type", "general"),
            }
        )
        state.add_trace("finalize", state.decision)

        return {
            "answer": state.answer,
            "trace": state.trace,
            "contexts": self._docs_to_contexts(state.retrieved_docs),
            "decision": state.decision,
        }


def run_agentic_rag(config: Any, question: str) -> Dict[str, Any]:
    graph = AgenticRAGGraph(config=config)
    return graph.run(question)
