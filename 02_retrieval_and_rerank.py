import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import CrossEncoder

# ==========================================
# 1. 核心配置区
# ==========================================
DB_DIR = "./chroma_db"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
# 面试考点：为什么选 bge-reranker-large？
# 答：它是目前开源界较强的多语言重排模型之一，对垂直领域的长文本重排效果较好。
RERANK_MODEL_NAME = "BAAI/bge-reranker-large" 

def get_vector_store():
    """加载第一阶段构建好的本地向量数据库。"""
    print("正在加载本地向量数据库...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cuda'}, # 依然使用你的 A5000 显卡
        encode_kwargs={'normalize_embeddings': True}
    )
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    return vector_db

def advanced_retrieval(query, vector_db, reranker_model, top_k_recall=15, top_k_rerank=3):
    """
    核心算法：双阶段检索。
    """
    print(f"\n[{'-'*10} 阶段一：向量粗排（召回） {'-'*10}]")
    # 1. 粗排：从 883 个块中，快速捞出最相似的 15 个块
    # 这里使用相似度检索，因做了 L2 归一化，余弦相似度可视作点积比较。
    docs = vector_db.similarity_search(query, k=top_k_recall)
    
    print(f"粗排完成，已召回 {len(docs)} 个候选文档块。")
    
    print(f"\n[{'-'*10} 阶段二：交叉编码器精排（重排） {'-'*10}]")
    # 2. 准备重排输入格式：[(问题, 文档1), (问题, 文档2), ...]
    cross_input = [[query, doc.page_content] for doc in docs]
    
    # 3. 让重排模型给这 15 个组合打分
    # 分数越高，代表文档和问题越匹配
    scores = reranker_model.predict(cross_input)
    
    # 4. 将文档和分数绑定，并按分数从高到低排序
    scored_docs = zip(docs, scores)
    sorted_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
    
    # 5. 截取精排后的前 3 条
    final_top_docs = sorted_docs[:top_k_rerank]
    
    return final_top_docs, docs # 返回精排结果，以及粗排结果（用于对比）

if __name__ == "__main__":
    # 1. 初始化组件
    vector_db = get_vector_store()
    
    print("正在加载重排模型 (初次运行需下载约 2GB，请耐心等待)...")
    # 加载交叉编码器模型并放到显卡
    reranker = CrossEncoder(RERANK_MODEL_NAME, device='cuda')
    
    # 2. 模拟用户提问（你可以改成任何你想问的医疗问题）
    # 既然你加载了高血压和糖尿病的指南，我们就问一个相关的
    user_query = "高血压患者每天吃盐不能超过多少克？"
    print(f"\n用户提问: {user_query}")
    
    # 3. 执行高级检索
    final_docs, recall_docs = advanced_retrieval(
        query=user_query, 
        vector_db=vector_db, 
        reranker_model=reranker,
        top_k_recall=15,  # 粗排捞 15 个
        top_k_rerank=3    # 精排留 3 个
    )
    
    # 4. 打印结果，见证奇迹
    print(f"\n[{'-'*10} 最终精排前 3 结果展示 {'-'*10}]")
    for i, (doc, score) in enumerate(final_docs):
        # 找一下这个文档在原来粗排里排第几名
        original_rank = recall_docs.index(doc) + 1 
        print(f"\n🏆 第 {i+1} 名（重排得分: {score:.4f} | 原粗排名次: 第 {original_rank} 名）")
        print(f"内容: {doc.page_content[:150]}...") # 只打印前150个字方便查看
