"""MedAgent 数据摄取模块。"""

from .chunkers import split_documents
from .index_builder import build_vector_db
from .loaders import load_documents

__all__ = ["load_documents", "split_documents", "build_vector_db"]
