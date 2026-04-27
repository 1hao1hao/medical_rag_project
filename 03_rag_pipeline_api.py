import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import CrossEncoder
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage, SystemMessage

# 加载 .env 文件中的 API 密钥
load_dotenv()

# ==========================================
# 1. 核心配置区
# ==========================================
DB_DIR = "./chroma_db"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
RERANK_MODEL_NAME = "BAAI/bge-reranker-large"
# 使用阿里云百炼的 Qwen-Max 模型
LLM_MODEL_NAME = "qwen3-max" 

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
# 3. 大模型生成模块（调用阿里云 API）
# ==========================================
def generate_answer(query, retrieved_docs):
    """
    使用 LangChain 的 ChatTongyi 调用云端大模型。
    """
    # 1. 拼接参考资料
    context = "\n".join([f"[参考片段 {i+1}] {doc.page_content}" for i, (doc, _) in enumerate(retrieved_docs)])
    
    # 2. 构建系统提示词（系统人设与约束）
    system_prompt = """你是一个专业的医疗AI助手。请严格根据用户提供的【参考资料】来回答问题。
要求：
1. 回答要专业、准确、条理清晰。
2. 如果参考资料中没有相关信息，请直接回答“根据提供的资料，我无法回答此问题”，绝对不要利用你的先验知识胡编乱造。"""

    # 3. 构建用户提示词（用户输入）
    user_prompt = f"【参考资料】\n{context}\n\n【用户问题】\n{query}\n\n请给出回答："

    # 4. 初始化大模型客户端
    chat_model = ChatTongyi(model=LLM_MODEL_NAME)
    
    # 5. 发送请求
    messages =[
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    print("\n=======================================================")
    print(f"\n👨‍⚕️ 封装后的用户提问: {user_prompt}")
    
    response = chat_model.invoke(messages)
    return response.content

if __name__ == "__main__":
    print(">>> 正在初始化 RAG 系统组件...")
    vector_db = get_vector_store()
    reranker = CrossEncoder(RERANK_MODEL_NAME, device='cuda')
    
    # 接收用户提问
    user_query = "预防高血压应该怎么做？"
    print(f"\n👨‍⚕️ 用户提问: {user_query}")
    
    # 执行检索
    print(">>> 正在知识库中检索并重排...")
    top_docs = advanced_retrieval(user_query, vector_db, reranker)
    
    # 执行生成
    print(f">>> 正在呼叫云端大模型 {LLM_MODEL_NAME} 总结答案...\n")
    final_answer = generate_answer(user_query, top_docs)
    
    print("="*40)
    print("🤖 AI 医生最终回答：\n")
    print(final_answer)
    print("="*40)
