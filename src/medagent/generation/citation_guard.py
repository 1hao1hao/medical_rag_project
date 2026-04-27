from typing import Any, Dict, List, Optional


class CitationGuard:
    """基于规则的引用守卫。"""

    CITATION_MARKER = "[来源"
    REFUSAL_PHRASE = "根据提供的资料无法回答"
    CITATION_REQUIREMENT = (
        "请确保回答中的关键结论都带来源标记，例如 [来源1]、[来源2]。"
        "若证据不足，请直接回答“根据提供的资料无法回答此问题”。"
    )
    DEFAULT_REFUSAL_ANSWER = "根据提供的资料无法回答此问题。"

    def has_citation_marker(self, answer: str) -> bool:
        return self.CITATION_MARKER in str(answer or "")

    def has_refusal_phrase(self, answer: str) -> bool:
        return self.REFUSAL_PHRASE in str(answer or "")

    def _requires_refusal(self, evidence_status: str) -> bool:
        return str(evidence_status).lower() in {"insufficient", "conflicting"}

    def _build_rewrite_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        rewritten = [dict(msg) for msg in messages]
        system_idx = -1
        for idx, msg in enumerate(rewritten):
            if str(msg.get("role", "")).strip() == "system":
                system_idx = idx
                break

        if system_idx >= 0:
            original = str(rewritten[system_idx].get("content", "")).strip()
            rewritten[system_idx]["content"] = (
                original + "\n" + self.CITATION_REQUIREMENT
            ).strip()
        else:
            rewritten.insert(
                0,
                {"role": "system", "content": self.CITATION_REQUIREMENT},
            )
        return rewritten

    def evaluate(
        self,
        answer: str,
        evidence_status: str,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """评估答案是否满足引用要求与证据一致性。"""

        normalized_status = str(evidence_status or "").lower()
        has_citation = self.has_citation_marker(answer)
        has_refusal = self.has_refusal_phrase(answer)
        requires_refusal = self._requires_refusal(normalized_status)

        if requires_refusal and not has_refusal:
            return {
                "passed": False,
                "action": "refuse",
                "reason": "证据不足或冲突时，答案必须拒答。",
                "consistent": False,
                "has_citation": has_citation,
                "has_refusal_phrase": has_refusal,
                "final_answer": self.DEFAULT_REFUSAL_ANSWER,
                "rewritten_messages": None,
            }

        if requires_refusal and has_refusal:
            return {
                "passed": True,
                "action": "accept",
                "reason": "证据不足且答案已正确拒答。",
                "consistent": True,
                "has_citation": has_citation,
                "has_refusal_phrase": has_refusal,
                "final_answer": answer,
                "rewritten_messages": None,
            }

        # sufficient 场景：不应拒答，且应带引用标记
        if normalized_status == "sufficient" and has_refusal:
            rewritten_messages = (
                self._build_rewrite_messages(messages) if messages is not None else None
            )
            return {
                "passed": False,
                "action": "rewrite_prompt",
                "reason": "证据充分时不应直接拒答，需重写提示词重新生成。",
                "consistent": False,
                "has_citation": has_citation,
                "has_refusal_phrase": has_refusal,
                "final_answer": answer,
                "rewritten_messages": rewritten_messages,
            }

        if normalized_status == "sufficient" and not has_citation:
            rewritten_messages = (
                self._build_rewrite_messages(messages) if messages is not None else None
            )
            return {
                "passed": False,
                "action": "rewrite_prompt",
                "reason": "证据充分但答案缺少来源标记，需重写提示词并重试。",
                "consistent": True,
                "has_citation": has_citation,
                "has_refusal_phrase": has_refusal,
                "final_answer": answer,
                "rewritten_messages": rewritten_messages,
            }

        if normalized_status == "sufficient" and has_citation:
            return {
                "passed": True,
                "action": "accept",
                "reason": "证据充分且答案包含来源标记。",
                "consistent": True,
                "has_citation": has_citation,
                "has_refusal_phrase": has_refusal,
                "final_answer": answer,
                "rewritten_messages": None,
            }

        # 未知状态，保护性拒答。
        return {
            "passed": False,
            "action": "refuse",
            "reason": "未知 evidence_status，进入保护性拒答。",
            "consistent": False,
            "has_citation": has_citation,
            "has_refusal_phrase": has_refusal,
            "final_answer": self.DEFAULT_REFUSAL_ANSWER,
            "rewritten_messages": None,
        }
