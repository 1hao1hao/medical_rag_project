from typing import Any, Dict


class ScoredDocument:
    """带分数的检索文档。"""

    __slots__ = ("content", "metadata", "score", "source")

    def __init__(
        self,
        content: str,
        metadata: Dict[str, Any],
        score: float,
        source: str,
    ) -> None:
        self.content = content
        self.metadata = metadata
        self.score = float(score)
        self.source = source

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "metadata": self.metadata,
            "score": self.score,
            "source": self.source,
        }

    def __repr__(self) -> str:
        return "ScoredDocument(source={0!r}, score={1:.4f})".format(
            self.source,
            self.score,
        )
