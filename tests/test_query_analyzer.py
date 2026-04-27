from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medagent.agent import QueryAnalyzer


def test_query_analyzer_greeting() -> None:
    analyzer = QueryAnalyzer()
    result = analyzer.analyze("你好")
    data = result.to_dict()

    assert data["need_retrieval"] is False
    assert data["query_type"] == "greeting"
    assert data["risk_level"] == "low"
    assert data["need_query_rewrite"] is False
    assert data["rewritten_query"] is None


def test_query_analyzer_medical_fact() -> None:
    analyzer = QueryAnalyzer()
    result = analyzer.analyze("高血压患者每天吃盐不能超过多少克")
    data = result.to_dict()

    assert data["need_retrieval"] is True
    assert data["query_type"] == "medical_fact"
    assert data["risk_level"] == "low"
    assert data["need_query_rewrite"] is False


def test_query_analyzer_high_risk() -> None:
    analyzer = QueryAnalyzer()
    result = analyzer.analyze("我胸痛血压很高要不要自己加药")
    data = result.to_dict()

    assert data["need_retrieval"] is True
    assert data["query_type"] == "high_risk_medical"
    assert data["risk_level"] == "high"


def test_query_analyzer_need_rewrite() -> None:
    analyzer = QueryAnalyzer()
    result = analyzer.analyze("这个能吃吗")
    data = result.to_dict()

    assert data["need_retrieval"] is True
    assert data["need_query_rewrite"] is True
    assert isinstance(data["rewritten_query"], str)
    assert data["rewritten_query"].strip() != ""
