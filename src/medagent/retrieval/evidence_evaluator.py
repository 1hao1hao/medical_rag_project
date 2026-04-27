import re
from typing import Any, Dict, List, Sequence, Tuple


STATUS_SUFFICIENT = "sufficient"
STATUS_INSUFFICIENT = "insufficient"
STATUS_CONFLICTING = "conflicting"

ACTION_GENERATE = "generate"
ACTION_REWRITE = "rewrite_and_retrieve"
ACTION_EXPAND = "expand_retrieval"
ACTION_REFUSE = "refuse"

_CONFLICT_PAIRS = [
    ("可以", "不可以"),
    ("能", "不能"),
    ("建议", "不建议"),
    ("应该", "不应该"),
    ("需要", "不需要"),
    ("宜", "不宜"),
]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def _extract_keywords(question: str) -> List[str]:
    text = _normalize(question)
    if not text:
        return []

    terms = []
    # 英文/数字词
    terms.extend(re.findall(r"[A-Za-z0-9]{2,}", text))

    # 中文连续片段 -> 2-gram 关键词，提升中文场景重叠鲁棒性
    chinese_chunks = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    for chunk in chinese_chunks:
        if len(chunk) == 2:
            terms.append(chunk)
            continue
        for idx in range(0, len(chunk) - 1):
            terms.append(chunk[idx : idx + 2])

    # 去重并保持顺序
    seen = set()
    deduped = []
    for term in terms:
        if term not in seen:
            seen.add(term)
            deduped.append(term)
    return deduped


def _top_score(retrieved_docs: Sequence[Any]) -> float:
    if not retrieved_docs:
        return 0.0
    return float(getattr(retrieved_docs[0], "score", 0.0) or 0.0)


def _keyword_overlap_ratio(question: str, retrieved_docs: Sequence[Any]) -> float:
    keywords = _extract_keywords(question)
    if not keywords:
        return 0.0

    joined_text = _normalize(
        " ".join([str(getattr(doc, "content", "")) for doc in retrieved_docs])
    )
    matched = 0
    for kw in keywords:
        if kw in joined_text:
            matched += 1
    return float(matched) / float(len(keywords))


def _detect_conflict(
    retrieved_docs: Sequence[Any], conflict_min_score: float
) -> Tuple[bool, str]:
    if len(retrieved_docs) < 2:
        return False, ""

    candidates = [
        _normalize(str(getattr(doc, "content", "")))
        for doc in retrieved_docs
        if float(getattr(doc, "score", 0.0) or 0.0) >= conflict_min_score
    ]
    if len(candidates) < 2:
        return False, ""

    text_all = "||".join(candidates)
    for pos, neg in _CONFLICT_PAIRS:
        if pos in text_all and neg in text_all:
            return True, "检测到潜在冲突证据（{0} / {1}）".format(pos, neg)
    return False, ""


class EvidenceEvaluator:
    """证据充分性评估器（启发式规则版）。"""

    def __init__(self, config: Any) -> None:
        self.config = config

    def evaluate(self, question: str, retrieved_docs: Sequence[Any]) -> Dict[str, Any]:
        docs = list(retrieved_docs or [])
        if not docs:
            return {
                "status": STATUS_INSUFFICIENT,
                "confidence": 0.0,
                "action": ACTION_REFUSE,
                "reason": "未检索到任何证据文档。",
            }

        min_top_score = float(self.config.evidence_min_top_score)
        min_overlap = float(self.config.evidence_min_keyword_overlap)
        conflict_min_score = float(self.config.evidence_conflict_min_score)

        top_score = _top_score(docs)
        if top_score < min_top_score:
            confidence = max(0.0, min(1.0, top_score))
            return {
                "status": STATUS_INSUFFICIENT,
                "confidence": confidence,
                "action": ACTION_REWRITE,
                "reason": "最高检索分数过低，建议改写问题后重新检索。",
            }

        overlap_ratio = _keyword_overlap_ratio(question, docs)
        if overlap_ratio < min_overlap:
            confidence = max(0.0, min(1.0, overlap_ratio))
            return {
                "status": STATUS_INSUFFICIENT,
                "confidence": confidence,
                "action": ACTION_EXPAND,
                "reason": "证据与问题关键词重叠不足，建议扩大召回范围。",
            }

        conflict, conflict_reason = _detect_conflict(docs, conflict_min_score)
        if conflict:
            confidence = max(0.0, min(1.0, top_score))
            return {
                "status": STATUS_CONFLICTING,
                "confidence": confidence,
                "action": ACTION_EXPAND,
                "reason": conflict_reason,
            }

        confidence = max(0.0, min(1.0, 0.5 * top_score + 0.5 * overlap_ratio))
        return {
            "status": STATUS_SUFFICIENT,
            "confidence": confidence,
            "action": ACTION_GENERATE,
            "reason": "证据分数与关键词覆盖达标，可进入生成阶段。",
        }
