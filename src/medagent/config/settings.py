from pathlib import Path
from typing import Any, Dict

_VALID_LLM_PROVIDERS = {"mock", "tongyi", "local"}
_DEFAULT_CHUNK_SIZE = 500
_DEFAULT_CHUNK_OVERLAP = 100


class Settings:
    """项目运行配置。"""

    __slots__ = (
        "data_dir",
        "db_dir",
        "output_dir",
        "embedding_model",
        "reranker_model",
        "llm_provider",
        "llm_model",
        "retrieval_device",
        "generation_device",
        "top_k_recall",
        "top_k_rerank",
        "use_reranker",
        "chunk_size",
        "chunk_overlap",
    )

    def __init__(
        self,
        data_dir: str,
        db_dir: str,
        output_dir: str,
        embedding_model: str,
        reranker_model: str,
        llm_provider: str,
        llm_model: str,
        retrieval_device: str,
        generation_device: str,
        top_k_recall: int,
        top_k_rerank: int,
        use_reranker: bool,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        self.data_dir = data_dir
        self.db_dir = db_dir
        self.output_dir = output_dir
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.retrieval_device = retrieval_device
        self.generation_device = generation_device
        self.top_k_recall = top_k_recall
        self.top_k_rerank = top_k_rerank
        self.use_reranker = use_reranker
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._validate()

    def _validate(self) -> None:
        if self.llm_provider not in _VALID_LLM_PROVIDERS:
            raise ValueError(
                f"llm_provider 非法: {self.llm_provider}，可选值: {sorted(_VALID_LLM_PROVIDERS)}"
            )
        if self.top_k_recall <= 0:
            raise ValueError("top_k_recall 必须大于 0")
        if self.top_k_rerank <= 0:
            raise ValueError("top_k_rerank 必须大于 0")
        if self.top_k_rerank > self.top_k_recall:
            raise ValueError("top_k_rerank 不能大于 top_k_recall")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap 不能小于 0")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")

    def to_dict(self) -> Dict[str, Any]:
        return {field: getattr(self, field) for field in self.__slots__}

    def __repr__(self) -> str:
        return f"Settings({self.to_dict()!r})"


_REQUIRED_KEYS = {
    "data_dir",
    "db_dir",
    "output_dir",
    "embedding_model",
    "reranker_model",
    "llm_provider",
    "llm_model",
    "retrieval_device",
    "generation_device",
    "top_k_recall",
    "top_k_rerank",
    "use_reranker",
}

_OPTIONAL_KEYS = {
    "chunk_size",
    "chunk_overlap",
}

_KNOWN_KEYS = _REQUIRED_KEYS | _OPTIONAL_KEYS


def _coerce_bool(value: Any, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    raise ValueError(f"{key} 必须是布尔值")


def _coerce_int(value: Any, key: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{key} 必须是整数")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text and (text.isdigit() or (text[0] == "-" and text[1:].isdigit())):
            return int(text)
    raise ValueError(f"{key} 必须是整数")


def _strip_quotes(text: str) -> str:
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    return text


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    """兜底解析器：仅支持简单的 key: value 映射。"""

    parsed: Dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"无法解析配置行: {raw_line}")
        key, value = line.split(":", 1)
        key = key.strip()
        value = _strip_quotes(value.strip())
        lowered = value.lower()
        if lowered in {"true", "false"}:
            parsed[key] = lowered == "true"
        elif value and (value.isdigit() or (value[0] == "-" and value[1:].isdigit())):
            parsed[key] = int(value)
        else:
            parsed[key] = value
    return parsed


def _load_yaml(path: Path) -> Dict[str, Any]:
    raw_text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(raw_text)
        if loaded is None:
            return {}
        if not isinstance(loaded, dict):
            raise ValueError("配置文件根节点必须是映射对象")
        return loaded
    except ImportError:
        return _parse_simple_yaml(raw_text)


def load_config(path: str) -> Settings:
    """从 YAML 文件加载配置。"""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    raw = _load_yaml(config_path)
    missing = _REQUIRED_KEYS - raw.keys()
    if missing:
        raise ValueError(f"配置缺少字段: {sorted(missing)}")

    unknown = set(raw.keys()) - _KNOWN_KEYS
    if unknown:
        raise ValueError(f"配置包含未知字段: {sorted(unknown)}")

    return Settings(
        data_dir=str(raw["data_dir"]),
        db_dir=str(raw["db_dir"]),
        output_dir=str(raw["output_dir"]),
        embedding_model=str(raw["embedding_model"]),
        reranker_model=str(raw["reranker_model"]),
        llm_provider=str(raw["llm_provider"]),  # type: ignore[arg-type]
        llm_model=str(raw["llm_model"]),
        retrieval_device=str(raw["retrieval_device"]),
        generation_device=str(raw["generation_device"]),
        top_k_recall=_coerce_int(raw["top_k_recall"], "top_k_recall"),
        top_k_rerank=_coerce_int(raw["top_k_rerank"], "top_k_rerank"),
        use_reranker=_coerce_bool(raw["use_reranker"], "use_reranker"),
        chunk_size=_coerce_int(raw.get("chunk_size", _DEFAULT_CHUNK_SIZE), "chunk_size"),
        chunk_overlap=_coerce_int(
            raw.get("chunk_overlap", _DEFAULT_CHUNK_OVERLAP), "chunk_overlap"
        ),
    )
