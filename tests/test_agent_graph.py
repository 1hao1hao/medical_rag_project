from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medagent.agent.graph import AgenticRAGGraph
from medagent.config import Settings
from medagent.retrieval.types import ScoredDocument


def _base_config() -> Settings:
    return Settings(
        data_dir="./data",
        db_dir="./chroma_db",
        output_dir="./outputs/dev_cpu",
        embedding_model="BAAI/bge-m3",
        reranker_model="BAAI/bge-reranker-large",
        llm_provider="mock",
        llm_model="mock-medagent",
        retrieval_device="cpu",
        generation_device="cpu",
        top_k_recall=15,
        top_k_rerank=3,
        use_reranker=False,
    )


class _Analyzer(object):
    def __init__(self, payload):
        self.payload = payload

    def analyze(self, _question):
        return self.payload


class _SeqEvaluator(object):
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def evaluate(self, _question, _docs):
        idx = self.calls
        self.calls += 1
        if idx >= len(self.outputs):
            return self.outputs[-1]
        return self.outputs[idx]


class _Gen(object):
    def __init__(self, text):
        self.text = text

    def generate_answer(self, _question, _docs):
        return self.text


def _doc(content: str, score: float) -> ScoredDocument:
    return ScoredDocument(
        content=content,
        metadata={"source": "fake"},
        score=score,
        source="fake",
    )


def test_agent_graph_greeting_no_retrieval() -> None:
    graph = AgenticRAGGraph(config=_base_config())
    result = graph.run("你好")

    assert isinstance(result["answer"], str)
    assert result["decision"]["action"] == "generate"
    assert result["decision"]["attempts"] == 0
    assert len(result["contexts"]) == 0
    assert any(item["step"] == "analyze_query" for item in result["trace"])


def test_agent_graph_rewrite_then_generate_once() -> None:
    cfg = _base_config()
    calls = []

    def retrieval_fn(query, recall_k):
        calls.append((query, recall_k))
        return [_doc("证据A", 0.2)], [_doc("证据A", 0.2)]

    analyzer = _Analyzer(
        {
            "need_retrieval": True,
            "query_type": "medical_fact",
            "risk_level": "low",
            "need_query_rewrite": True,
            "rewritten_query": "高血压患者每日食盐摄入上限是多少",
            "reason": "问题可检索",
        }
    )
    evaluator = _SeqEvaluator(
        [
            {
                "status": "insufficient",
                "confidence": 0.2,
                "action": "rewrite_and_retrieve",
                "reason": "首轮证据不足",
            },
            {
                "status": "sufficient",
                "confidence": 0.8,
                "action": "generate",
                "reason": "证据已足够",
            },
        ]
    )
    graph = AgenticRAGGraph(
        config=cfg,
        query_analyzer=analyzer,
        evidence_evaluator=evaluator,
        answer_generator=_Gen("生成成功"),
        retrieval_fn=retrieval_fn,
    )

    result = graph.run("高血压患者每天吃盐不能超过多少克")

    assert result["answer"] == "生成成功"
    assert result["decision"]["attempts"] == 2
    assert result["decision"]["final_query"] == "高血压患者每日食盐摄入上限是多少"
    assert len(calls) == 2
    assert calls[0][0] == "高血压患者每天吃盐不能超过多少克"
    assert calls[1][0] == "高血压患者每日食盐摄入上限是多少"
    assert any(item["step"] == "rewrite_query" for item in result["trace"])


def test_agent_graph_expand_then_generate_once() -> None:
    cfg = _base_config()
    calls = []

    def retrieval_fn(query, recall_k):
        calls.append((query, recall_k))
        return [_doc("证据B", 0.4)], [_doc("证据B", 0.4)]

    analyzer = _Analyzer(
        {
            "need_retrieval": True,
            "query_type": "medical_fact",
            "risk_level": "low",
            "need_query_rewrite": False,
            "rewritten_query": None,
            "reason": "问题可检索",
        }
    )
    evaluator = _SeqEvaluator(
        [
            {
                "status": "insufficient",
                "confidence": 0.3,
                "action": "expand_retrieval",
                "reason": "关键词覆盖不足",
            },
            {
                "status": "sufficient",
                "confidence": 0.9,
                "action": "generate",
                "reason": "扩召回后证据充足",
            },
        ]
    )
    graph = AgenticRAGGraph(
        config=cfg,
        query_analyzer=analyzer,
        evidence_evaluator=evaluator,
        answer_generator=_Gen("扩召回后生成成功"),
        retrieval_fn=retrieval_fn,
    )

    result = graph.run("高血压患者每天吃盐不能超过多少克")

    assert result["answer"] == "扩召回后生成成功"
    assert result["decision"]["attempts"] == 2
    assert len(calls) == 2
    assert calls[0][1] == 15
    assert calls[1][1] == 30
    assert any(item["step"] == "expand_retrieval" for item in result["trace"])


def test_agent_graph_refuse_when_retrieve_fails() -> None:
    cfg = _base_config()

    def retrieval_fn(_query, _recall_k):
        raise RuntimeError("检索后端不可用")

    analyzer = _Analyzer(
        {
            "need_retrieval": True,
            "query_type": "medical_fact",
            "risk_level": "low",
            "need_query_rewrite": False,
            "rewritten_query": None,
            "reason": "问题可检索",
        }
    )
    evaluator = _SeqEvaluator(
        [
            {
                "status": "insufficient",
                "confidence": 0.0,
                "action": "refuse",
                "reason": "无可用证据",
            }
        ]
    )
    graph = AgenticRAGGraph(
        config=cfg,
        query_analyzer=analyzer,
        evidence_evaluator=evaluator,
        answer_generator=_Gen("不会用到"),
        retrieval_fn=retrieval_fn,
    )

    result = graph.run("高血压患者每天吃盐不能超过多少克")

    assert result["decision"]["action"] == "refuse"
    assert "无法回答" in result["answer"]
    assert any(item["step"] == "retrieve_error" for item in result["trace"])
