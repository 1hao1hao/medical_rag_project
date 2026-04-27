import os
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset

# 导入 RAGAS 评估指标
from ragas import evaluate
from ragas.metrics import (
    faithfulness,        # 忠实度：回答是否完全基于检索到的上下文（防幻觉）
    answer_relevancy,    # 回答相关性：回答是否直接解决了用户的问题
    context_precision,   # 上下文精确度：检索到的有用信息是否排在前面
    context_recall       # 上下文召回率：检索到的信息是否足以回答问题
)

# 导入 RAGAS 的 LangChain 包装器
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# 导入 LangChain 组件
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import CrossEncoder
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# ==========================================
# 1. 核心配置区
# ==========================================
DB_DIR = "./chroma_db"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
RERANK_MODEL_NAME = "BAAI/bge-reranker-large"
LLM_MODEL_NAME = "qwen3-max" # 裁判模型和生成模型都用它

# ==========================================
# 2. 初始化系统组件
# ==========================================
print(">>> 正在初始化系统组件...")
# 1. 检索组件
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)
vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
reranker = CrossEncoder(RERANK_MODEL_NAME, device='cuda')

# 2. 大模型组件
chat_model = ChatTongyi(model=LLM_MODEL_NAME)

# 3. RAGAS 裁判组件（将我们的模型包装给 RAGAS 使用）
ragas_llm = LangchainLLMWrapper(chat_model)
ragas_emb = LangchainEmbeddingsWrapper(embeddings)

# ==========================================
# 3. 核心逻辑：运行 RAG 并收集数据
# ==========================================
def run_rag_pipeline(query):
    """运行第三阶段完整链路，并返回评估所需的数据。"""
    # 1. 检索与重排
    docs = vector_db.similarity_search(query, k=15)
    cross_input = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(cross_input)
    sorted_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:3]
    
    # 提取纯文本上下文
    contexts = [doc.page_content for doc, _ in sorted_docs]
    context_str = "\n".join([f"[片段 {i+1}] {ctx}" for i, ctx in enumerate(contexts)])
    
    # 2. 大模型生成
    system_prompt = "你是一个专业的医疗AI助手。请严格根据用户提供的【参考资料】来回答问题。"
    user_prompt = f"【参考资料】\n{context_str}\n\n【用户问题】\n{query}\n\n请给出回答："
    
    response = chat_model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]).content
    
    return contexts, response

# ==========================================
# 4. 构建测试集并执行评估
# ==========================================
if __name__ == "__main__":
    # 1. 准备测试问题和标准答案（标注真值）
    # 在真实企业里，这通常是医生/专家标注的 100 个问题。我们这里用 2 个做演示。
    eval_data =[
        {
            "question": "高血压患者每天吃盐不能超过多少克？",
            "ground_truth": "高血压患者每天的钠盐摄入量应限制在5克以下（<5g/d）。"
        },
        {
            "question": "预防高血压应该怎么做？",
            "ground_truth": "预防高血压应限制钠盐摄入、减轻体重、适量运动、戒烟限酒等。"
        }
    ]
    
    data_for_ragas = {
        "question": [],
        "answer": [],
        "contexts":[],
        "ground_truth":[]
    }
    
    print("\n>>> 开始批量运行 RAG 系统收集回答...")
    for item in eval_data:
        q = item["question"]
        print(f"正在处理问题: {q}")
        contexts, answer = run_rag_pipeline(q)
        
        data_for_ragas["question"].append(q)
        data_for_ragas["answer"].append(answer)
        data_for_ragas["contexts"].append(contexts)
        data_for_ragas["ground_truth"].append(item["ground_truth"])
        
    # 2. 转换为 HuggingFace 数据集格式
    dataset = Dataset.from_dict(data_for_ragas)
    
    # 3. 启动 RAGAS 评估
    print("\n>>> 正在呼叫 Qwen-Max 裁判进行 RAGAS 量化评估 (可能需要 1-2 分钟)...")
    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,      # 忠实度 (防幻觉)
            answer_relevancy,  # 回答相关性
            context_precision, # 上下文精确度
            context_recall     # 上下文召回率 
        ],
        llm=ragas_llm,
        embeddings=ragas_emb,
        raise_exceptions=False # 防止因为网络波动中断
    )
    
    # 4. 打印最终成绩单！
    print("\n" + "="*40)
    print("🏆 RAG 系统量化评估报告 🏆")
    print("="*40)
    df = result.to_pandas()
    
    # 直接打印 result 对象即可，最新版 RAGAS 内置了格式化输出
    print(f"综合得分:\n{result}")
        
    print("\n详细数据已保存至 rag_evaluation_results.csv")
    df.to_csv("rag_evaluation_results.csv", index=False, encoding='utf-8-sig')
