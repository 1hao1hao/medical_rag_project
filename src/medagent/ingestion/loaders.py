import os
from typing import Any, List


def load_documents(data_dir: str) -> List[Any]:
    """加载目录下的 PDF 文档（不递归）。"""

    if not os.path.exists(data_dir):
        raise ValueError("数据目录不存在: {0}".format(data_dir))
    if not os.path.isdir(data_dir):
        raise ValueError("数据目录不是文件夹: {0}".format(data_dir))

    pdf_files = [
        os.path.join(data_dir, name)
        for name in sorted(os.listdir(data_dir))
        if name.lower().endswith(".pdf")
    ]
    if not pdf_files:
        raise ValueError(
            "在目录 {0} 下未找到 PDF 文件，请先放入至少一个 .pdf 文档。".format(data_dir)
        )

    try:
        from langchain_community.document_loaders import PDFPlumberLoader
    except ImportError:
        raise ImportError(
            "缺少 PDF 加载依赖。请安装 langchain-community 与 pdfplumber 后重试。"
        )

    documents = []
    for pdf_path in pdf_files:
        loader = PDFPlumberLoader(pdf_path)
        loaded = loader.load()
        documents.extend(loaded)

    if not documents:
        raise ValueError(
            "PDF 已找到但未解析出有效内容，请检查 PDF 是否损坏或为纯图片扫描件。"
        )

    return documents
