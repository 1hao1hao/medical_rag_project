import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ==========================================
# 1. 核心配置区
# ==========================================
DATA_DIR = "./data"
DB_DIR = "./chroma_db"
# 面试考点：为什么选 BGE-m3？
# 答：BGE-m3 支持多语言，且最高支持 8192 长度的上下文，对垂直领域的长文本支持非常好。
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

def load_documents(data_dir):
    """
    加载目录下的所有 PDF 文档
    为什么不用 PyPDF2？因为 pdfplumber 对表格和段落的解析更精准，适合医疗/法律文档。
    """
    documents =[]
    for filename in os.listdir(data_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(data_dir, filename)
            loader = PDFPlumberLoader(file_path)
            documents.extend(loader.load())
            print(f"已加载文档: {filename}")
    return documents

def split_documents(documents):
    """
    文本切分（Chunking）
    面试考点：Chunk size 和 overlap 怎么定？
    答：医疗文档上下文依赖强，设定 chunk_size=500 保证包含完整语义，overlap=100 防止生硬截断病症描述。
    后续优化方案：可以升级为 Parent-Child 切分法。
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "！", "？", "，", " "]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"文档总计被切分为 {len(chunks)} 个数据块 (Chunks)。")
    return chunks

def build_vector_db():
    """
    主函数：构建向量数据库
    """
    # 1. 加载文档
    print("开始加载文档...")
    docs = load_documents(DATA_DIR)
    if not docs:
        print("未在 data 目录下找到 PDF 文档！")
        return

    # 2. 文本切分
    print("开始切分文档...")
    chunks = split_documents(docs)

    # 3. 初始化 Embedding 模型
    print("正在加载 BGE-m3 Embedding 模型 (初次运行需下载，请耐心等待)...")
    model_kwargs = {'device': 'cuda'} # 如果你有显卡，改为 'cuda'
    encode_kwargs = {'normalize_embeddings': True} # 必须为 True，计算余弦相似度才准确
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # 4. 存入 Chroma 向量数据库并持久化到本地
    print("正在将数据向量化并存入 ChromaDB...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    
    print(f"✅ 向量数据库构建完成！已持久化保存至 {DB_DIR} 目录。")

if __name__ == "__main__":
    build_vector_db()
    
    
    
