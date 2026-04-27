from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medagent.generation.citation_guard import CitationGuard


def test_citation_guard_sufficient_with_citation_accept() -> None:
    guard = CitationGuard()
    answer = "高血压患者每日食盐建议不超过5克。[来源1]"
    result = guard.evaluate(answer=answer, evidence_status="sufficient")

    assert result["passed"] is True
    assert result["action"] == "accept"
    assert result["has_citation"] is True


def test_citation_guard_sufficient_without_citation_rewrite() -> None:
    guard = CitationGuard()
    messages = [
        {"role": "system", "content": "你是医疗助手"},
        {"role": "user", "content": "高血压患者每天吃盐不能超过多少克"},
    ]
    answer = "高血压患者每日食盐建议不超过5克。"
    result = guard.evaluate(
        answer=answer,
        evidence_status="sufficient",
        messages=messages,
    )

    assert result["passed"] is False
    assert result["action"] == "rewrite_prompt"
    rewritten = result["rewritten_messages"]
    assert isinstance(rewritten, list)
    assert "来源标记" in rewritten[0]["content"]


def test_citation_guard_insufficient_without_refusal_refuse() -> None:
    guard = CitationGuard()
    answer = "建议自行加大药量。"
    result = guard.evaluate(answer=answer, evidence_status="insufficient")

    assert result["passed"] is False
    assert result["action"] == "refuse"
    assert "无法回答" in result["final_answer"]


def test_citation_guard_insufficient_with_refusal_accept() -> None:
    guard = CitationGuard()
    answer = "根据提供的资料无法回答此问题。"
    result = guard.evaluate(answer=answer, evidence_status="insufficient")

    assert result["passed"] is True
    assert result["action"] == "accept"
    assert result["has_refusal_phrase"] is True


def test_citation_guard_sufficient_with_refusal_inconsistent() -> None:
    guard = CitationGuard()
    answer = "根据提供的资料无法回答此问题。"
    result = guard.evaluate(answer=answer, evidence_status="sufficient")

    assert result["passed"] is False
    assert result["action"] == "rewrite_prompt"
    assert result["consistent"] is False
