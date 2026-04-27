from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medagent.retrieval.evidence_evaluator import EvidenceEvaluator
from medagent.retrieval.types import ScoredDocument


class _DummyConfig(object):
    evidence_min_top_score = 0.4
    evidence_min_keyword_overlap = 0.3
    evidence_conflict_min_score = 0.5


def _doc(content: str, score: float) -> ScoredDocument:
    return ScoredDocument(
        content=content,
        metadata={"source": "fake"},
        score=score,
        source="fake",
    )


def test_evidence_evaluator_no_docs() -> None:
    evaluator = EvidenceEvaluator(_DummyConfig())
    result = evaluator.evaluate("高血压患者每天吃盐不能超过多少克", [])

    assert result["status"] == "insufficient"
    assert result["action"] == "refuse"


def test_evidence_evaluator_low_top_score() -> None:
    evaluator = EvidenceEvaluator(_DummyConfig())
    docs = [_doc("高血压患者每天吃盐不能超过5克。", 0.2)]
    result = evaluator.evaluate("高血压患者每天吃盐不能超过多少克", docs)

    assert result["status"] == "insufficient"
    assert result["action"] == "rewrite_and_retrieve"


def test_evidence_evaluator_low_keyword_overlap() -> None:
    evaluator = EvidenceEvaluator(_DummyConfig())
    docs = [_doc("糖尿病患者建议减少精制碳水摄入。", 0.9)]
    result = evaluator.evaluate("高血压患者每天吃盐不能超过多少克", docs)

    assert result["status"] == "insufficient"
    assert result["action"] == "expand_retrieval"


def test_evidence_evaluator_sufficient() -> None:
    evaluator = EvidenceEvaluator(_DummyConfig())
    docs = [_doc("高血压患者每天吃盐不宜超过5克，建议低盐饮食。", 0.9)]
    result = evaluator.evaluate("高血压患者每天吃盐不能超过多少克", docs)

    assert result["status"] == "sufficient"
    assert result["action"] == "generate"
    assert result["confidence"] >= 0.3


def test_evidence_evaluator_conflicting() -> None:
    evaluator = EvidenceEvaluator(_DummyConfig())
    docs = [
        _doc("高血压患者可以吃咸菜。", 0.9),
        _doc("高血压患者不可以吃咸菜。", 0.85),
    ]
    result = evaluator.evaluate("高血压患者能不能吃咸菜", docs)

    assert result["status"] == "conflicting"
    assert result["action"] == "expand_retrieval"
