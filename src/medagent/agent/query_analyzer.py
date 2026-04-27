import re
from typing import Any, List, Optional

from .state import QueryAnalysisState

# 规则区：后续可替换为 LLM classifier，仅需保留相同输出结构。
GREETING_PATTERNS = [
    re.compile(r"^\s*(你好|您好|嗨|hi|hello|早上好|晚上好)[！!。,.，?？]*\s*$", re.IGNORECASE),
]

HIGH_RISK_KEYWORDS = [
    "胸痛",
    "急救",
    "严重",
    "呼吸困难",
    "昏迷",
    "抽搐",
    "大出血",
    "意识不清",
    "自行加药",
    "自己加药",
    "加药",
]

MEDICAL_FACT_HINTS = [
    "高血压",
    "糖尿病",
    "指南",
    "每天",
    "不能超过",
    "多少克",
    "怎么做",
    "应该",
    "是否",
    "能不能",
    "可以吗",
    "预防",
    "治疗",
]

AMBIGUOUS_REFERENCES = [
    "这个",
    "那个",
    "它",
    "这药",
    "那个药",
    "这个药",
]

MEDICAL_ENTITIES = [
    "高血压",
    "糖尿病",
    "冠心病",
    "药",
    "盐",
    "食物",
    "剂量",
    "血压",
]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _is_greeting(question: str) -> bool:
    for pattern in GREETING_PATTERNS:
        if pattern.match(question):
            return True
    return False


def _contains_any(question: str, keywords: List[str]) -> bool:
    return any(word in question for word in keywords)


def _is_short_query(question: str) -> bool:
    pure = _normalize_text(question)
    return len(pure) <= 6


def _is_ambiguous_query(question: str) -> bool:
    has_ref = _contains_any(question, AMBIGUOUS_REFERENCES)
    has_entity = _contains_any(question, MEDICAL_ENTITIES)
    return has_ref and not has_entity


def _suggest_rewrite(question: str, short_query: bool, ambiguous_query: bool) -> Optional[str]:
    if ambiguous_query:
        return "请明确“这个”具体指什么（药物/食物/行为），并补充相关疾病背景。"
    if short_query:
        return "请补充具体人群、疾病背景和你想确认的指标或处理方案。"
    return None


class QueryAnalyzer:
    """Query Analyzer：默认规则判定，可选 LLM 增强。"""

    def __init__(self, llm_client: Any = None, enable_llm: bool = False) -> None:
        self.llm_client = llm_client
        self.enable_llm = bool(enable_llm and llm_client is not None)

    def _maybe_refine_with_llm(
        self, question: str, state: QueryAnalysisState
    ) -> QueryAnalysisState:
        if not self.enable_llm:
            return state
        if not state.need_query_rewrite:
            return state

        try:
            messages = [
                {
                    "role": "system",
                    "content": "你是查询改写助手。请在一行内给出更明确的医疗问题，不要附加解释。",
                },
                {"role": "user", "content": question},
            ]
            rewritten = str(self.llm_client.generate(messages)).strip()
            if rewritten:
                state.rewritten_query = rewritten
                state.reason = state.reason + "；已使用 LLM 生成改写建议"
        except Exception:
            # LLM 增强失败时回退规则结果，不影响主流程稳定性。
            pass

        return state

    def analyze(self, question: str) -> QueryAnalysisState:
        text = str(question or "").strip()
        if not text:
            raise ValueError("question 不能为空。")

        if _is_greeting(text):
            state = QueryAnalysisState(
                need_retrieval=False,
                query_type="greeting",
                risk_level="low",
                need_query_rewrite=False,
                rewritten_query=None,
                reason="识别为问候语，无需进入检索流程。",
            )
            return self._maybe_refine_with_llm(text, state)

        high_risk = _contains_any(text, HIGH_RISK_KEYWORDS)
        short_query = _is_short_query(text)
        ambiguous_query = _is_ambiguous_query(text)
        need_rewrite = short_query or ambiguous_query

        looks_medical_fact = _contains_any(text, MEDICAL_FACT_HINTS)

        if high_risk:
            query_type = "high_risk_medical"
            risk_level = "high"
            reason = "检测到胸痛/急救/严重症状或自行加药等高风险关键词。"
        elif looks_medical_fact:
            query_type = "medical_fact"
            risk_level = "low"
            reason = "识别为医疗指南或事实类问题，建议检索后回答。"
        elif need_rewrite:
            query_type = "ambiguous_query"
            risk_level = "medium"
            reason = "问题过短或存在指代不明，建议先改写再检索。"
        else:
            query_type = "general_medical"
            risk_level = "low"
            reason = "识别为一般医疗问题，建议先检索后回答。"

        state = QueryAnalysisState(
            need_retrieval=True,
            query_type=query_type,
            risk_level=risk_level,
            need_query_rewrite=need_rewrite,
            rewritten_query=_suggest_rewrite(
                question=text,
                short_query=short_query,
                ambiguous_query=ambiguous_query,
            ),
            reason=reason,
        )
        return self._maybe_refine_with_llm(text, state)
