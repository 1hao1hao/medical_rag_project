from typing import Any, Dict

from .chunkers import split_documents
from .loaders import load_documents


def build_vector_db(config: Any) -> Dict[str, Any]:
    """构建向量数据库并返回构建摘要。"""

    documents = load_documents(config.data_dir)
    chunks = split_documents(
        documents=documents,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )

    if not chunks:
        raise ValueError("文档切分后结果为空，请检查输入文档或分块参数配置。")

    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import Chroma
    except ImportError:
        raise ImportError(
            "缺少向量化依赖。请安装 langchain-community、transformers 与 chromadb 后重试。"
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=config.embedding_model,
        model_kwargs={"device": config.retrieval_device},
        encode_kwargs={"normalize_embeddings": True},
    )

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=config.db_dir,
    )

    return {
        "db_dir": config.db_dir,
        "documents": len(documents),
        "chunks": len(chunks),
        "embedding_model": config.embedding_model,
        "retrieval_device": config.retrieval_device,
    }
