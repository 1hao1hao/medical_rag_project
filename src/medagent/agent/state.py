from typing import Any, Dict, Optional


class QueryAnalysisState:
    """Query Analyzer 输出状态。"""

    __slots__ = (
        "need_retrieval",
        "query_type",
        "risk_level",
        "need_query_rewrite",
        "rewritten_query",
        "reason",
    )

    def __init__(
        self,
        need_retrieval: bool,
        query_type: str,
        risk_level: str,
        need_query_rewrite: bool,
        rewritten_query: Optional[str],
        reason: str,
    ) -> None:
        self.need_retrieval = bool(need_retrieval)
        self.query_type = str(query_type)
        self.risk_level = str(risk_level)
        self.need_query_rewrite = bool(need_query_rewrite)
        self.rewritten_query = rewritten_query
        self.reason = str(reason)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "need_retrieval": self.need_retrieval,
            "query_type": self.query_type,
            "risk_level": self.risk_level,
            "need_query_rewrite": self.need_query_rewrite,
            "rewritten_query": self.rewritten_query,
            "reason": self.reason,
        }

    def __repr__(self) -> str:
        return "QueryAnalysisState({0!r})".format(self.to_dict())
