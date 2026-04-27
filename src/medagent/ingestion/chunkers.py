from typing import Any, Dict, Iterable, List, Optional, Sequence

DEFAULT_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "，", " "]


class _SimpleDocument(object):
    def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.page_content = page_content
        self.metadata = metadata or {}


def _resolve_document_class():
    try:
        from langchain_core.documents import Document  # type: ignore

        return Document
    except ImportError:
        pass

    try:
        from langchain.schema import Document  # type: ignore

        return Document
    except ImportError:
        return _SimpleDocument


def _can_use_langchain_splitter(documents: Sequence[Any]) -> bool:
    if not documents:
        return True
    if not all(hasattr(doc, "page_content") for doc in documents):
        return False
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore

        return RecursiveCharacterTextSplitter is not None
    except ImportError:
        return False


def _split_text_simple(
    text: str, chunk_size: int, chunk_overlap: int, separators: Sequence[str]
) -> List[str]:
    if not text:
        return []

    pieces = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        split_end = end

        if end < text_length:
            for sep in separators:
                idx = text.rfind(sep, start, end)
                if idx != -1:
                    candidate = idx + len(sep)
                    if candidate > start:
                        split_end = max(split_end if split_end != end else 0, candidate)

            if split_end <= start or split_end > end:
                split_end = end

        chunk = text[start:split_end].strip()
        if chunk:
            pieces.append(chunk)

        if split_end >= text_length:
            break

        next_start = split_end - chunk_overlap
        if next_start <= start:
            next_start = split_end
        start = next_start

    return pieces


def _iter_text_and_metadata(documents: Iterable[Any]):
    for doc in documents:
        if hasattr(doc, "page_content"):
            text = getattr(doc, "page_content", "")
            metadata = getattr(doc, "metadata", {}) or {}
            yield str(text), dict(metadata)
        else:
            yield str(doc), {}


def split_documents(
    documents: Sequence[Any],
    chunk_size: int,
    chunk_overlap: int,
    separators: Optional[Sequence[str]] = None,
) -> List[Any]:
    """按配置进行分块，优先使用 LangChain，缺失依赖时自动回退。"""

    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap 不能小于 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size")

    active_separators = list(separators or DEFAULT_SEPARATORS)

    if _can_use_langchain_splitter(documents):
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=active_separators,
            )
            return splitter.split_documents(list(documents))
        except Exception:
            # 分块器运行异常时，自动回退到内置实现，保证流程不中断。
            pass

    document_class = _resolve_document_class()
    fallback_chunks = []
    for text, metadata in _iter_text_and_metadata(documents):
        for chunk_text in _split_text_simple(
            text=text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=active_separators,
        ):
            fallback_chunks.append(document_class(page_content=chunk_text, metadata=metadata))

    return fallback_chunks
