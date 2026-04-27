from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medagent.ingestion.chunkers import split_documents


def _texts(chunks):
    result = []
    for chunk in chunks:
        if hasattr(chunk, "page_content"):
            result.append(chunk.page_content)
        else:
            result.append(str(chunk))
    return result


def test_split_documents_chinese_period_no_error() -> None:
    docs = ["高血压患者每天应限制盐摄入。建议采用低盐饮食。"]
    chunks = split_documents(docs, chunk_size=12, chunk_overlap=2)
    texts = _texts(chunks)

    assert len(chunks) > 0
    assert all(text.strip() for text in texts)


def test_split_documents_chunk_params_take_effect() -> None:
    docs = ["甲" * 50]
    chunks_small = split_documents(docs, chunk_size=10, chunk_overlap=2)
    chunks_large = split_documents(docs, chunk_size=20, chunk_overlap=2)

    texts_small = _texts(chunks_small)
    assert len(chunks_small) >= len(chunks_large)
    assert max(len(text) for text in texts_small) <= 10
