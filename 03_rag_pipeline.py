import os
import torch
from transformers import AutoTokenizer, AutoModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import CrossEncoder

# ==========================================
# 1. 核心配置区
# ==========================================
DB_DIR = "./chroma_db"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
RERANK_MODEL_NAME = "BAAI/bge-reranker-large"
LLM_MODEL_NAME = "THUDM/chatglm3-6b"  # 清华开源的 6B 大模型

# ==========================================
# 2. 检索模块（复用第二阶段逻辑）
# ==========================================
def get_vector_store():
    """加载第一阶段构建好的本地向量数据库。"""
    print("正在加载本地向量数据库...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

def advanced_retrieval(query, vector_db, reranker_model, top_k_recall=15, top_k_rerank=3):
    """
    核心算法：双阶段检索。
    """
    docs = vector_db.similarity_search(query, k=top_k_recall)
    cross_input = [[query, doc.page_content] for doc in docs]
    scores = reranker_model.predict(cross_input)
    scored_docs = zip(docs, scores)
    sorted_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
    return sorted_docs[:top_k_rerank]

# ==========================================
# 3. 大模型生成模块（第三阶段核心）
# ==========================================
def build_prompt(query, retrieved_docs):
    """
    构建提示词：将检索到的文档拼接成上下文。
    面试考点：如何防止大模型产生幻觉？
    答：在提示词中加入强约束指令，如“如果参考资料中没有相关信息，请回答无法回答，绝对不要胡编乱造”。
    """
    context = "\n".join([f"[参考片段 {i+1}] {doc.page_content}" for i, (doc, _) in enumerate(retrieved_docs)])
    
    prompt = f"""你是一个专业的医疗AI助手。请严格根据以下提供的参考资料，回答用户的医疗问题。
如果参考资料中没有相关信息，请回答“根据提供的资料无法回答此问题”，绝对不要胡编乱造。

【参考资料】
{context}

【用户问题】
{query}

请给出专业、准确、易懂的回答："""

    return prompt

if __name__ == "__main__":
    print(">>> 正在初始化 RAG 系统组件...")
    
    # 1. 加载检索组件
    vector_db = get_vector_store()
    reranker = CrossEncoder(RERANK_MODEL_NAME, device='cuda')
    
    # 2. 加载大模型 (初次运行需下载约 12GB，请耐心等待)
    print(f">>> 正在加载大模型 {LLM_MODEL_NAME} 到 A5000 显卡...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, trust_remote_code=True)
    # .half() 表示使用 FP16 半精度加载，节省一半显存；.cuda() 表示放入显卡
    llm_model = AutoModel.from_pretrained(LLM_MODEL_NAME, trust_remote_code=True).half().cuda()
    llm_model = llm_model.eval() # 设置为推理模式
    
    # 3. 接收用户提问
    user_query = "高血压患者每天吃盐不能超过多少克？"
    print("\n=======================================================")
    print(f"\n👨‍⚕️ 用户提问: {user_query}")
    
    # 4. 执行检索
    print(">>> 正在知识库中检索并重排...")
    top_docs = advanced_retrieval(user_query, vector_db, reranker)
    
    # 5. 构建提示词并生成答案
    print(">>> 正在让大模型阅读资料并生成最终回答...\n")
    prompt = build_prompt(user_query, top_docs)
    print("\n=======================================================")
    print(f"\n👨‍⚕️ 封装后的用户提问: {prompt}")

    # 调用 ChatGLM3 的对话接口，history 设为空列表（单轮问答）
    response, history = llm_model.chat(tokenizer, prompt, history=[])
    
    print("\n=======================================================")
    print("🤖 AI 医生回答：")
    print(response)
    
    print("\n=======================================================")
    print("🎉 恭喜！你已经成功跑通了包含 检索+重排+生成 的完整 RAG 闭环！")
    print("=======================================================")
