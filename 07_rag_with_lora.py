import os
import torch
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder

# ==========================================
# 0. 环境兼容补丁
# ==========================================
transformers.modeling_utils.PreTrainedModel.all_tied_weights_keys = {}

# ==========================================
# 1. 核心配置区
# ==========================================
DB_DIR = "./chroma_db"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
RERANK_MODEL_NAME = "BAAI/bge-reranker-large"
BASE_MODEL_NAME = "THUDM/chatglm3-6b"
LORA_DIR = "./chatglm3_lora_medical"  # 我们刚刚训练好的 LoRA 权重目录

# ==========================================
# 2. 检索模块
# ==========================================
def get_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

def advanced_retrieval(query, vector_db, reranker_model, top_k_recall=15, top_k_rerank=3):
    docs = vector_db.similarity_search(query, k=top_k_recall)
    cross_input = [[query, doc.page_content] for doc in docs]
    scores = reranker_model.predict(cross_input)
    scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return scored_docs[:top_k_rerank]

# ==========================================
# 3. 主程序：加载模型并测试
# ==========================================
if __name__ == "__main__":
    print(">>> 1. 正在初始化检索组件...")
    vector_db = get_vector_store()
    reranker = CrossEncoder(RERANK_MODEL_NAME, device='cuda')
    
    print(">>> 2. 正在加载基础大模型...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    config = AutoConfig.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    # 直接加载模型，不需要手动干预 config
    base_model = AutoModel.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True).half().cuda()
    
    print(">>> 3. 正在挂载 LoRA 微调权重...")
    # 核心代码：将 LoRA 适配器加载到基础模型上
    model = PeftModel.from_pretrained(base_model, LORA_DIR)
    model = model.eval() # 切换到推理模式
    
    # 4. 执行检索
    user_query = "高血压患者每天吃盐不能超过多少克？"
    print(f"\n👨‍⚕️ 用户提问: {user_query}")
    print(">>> 正在检索知识库...")
    top_docs = advanced_retrieval(user_query, vector_db, reranker)
    
    # 5. 构建与训练时完全一致的 Prompt
    context = "\n".join([f"[片段 {i+1}] {doc.page_content}" for i, (doc, _) in enumerate(top_docs)])
    
    # 注意：这里的 instruction 必须和 05_build_lora_dataset.py 中完全一致，才能触发 LoRA 的记忆
    instruction = "你是一个专业的医疗AI助手。请严格根据【参考资料】回答问题。必须以“您好，我是AI医生。”开头。"
    input_text = f"【参考资料】\n{context}\n\n【用户问题】\n{user_query}"
    
    # 按照 ChatGLM3 的底层格式手动拼接（与训练时保持一致）
    prompt = f"<|system|>\n{instruction}\n<|user|>\n{input_text}\n<|assistant|>\n"
    
    print(">>> 正在生成回答...\n")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # 生成参数配置
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=100, 
            temperature=0.1, # 降低随机性，保证输出稳定
            do_sample=False
        )
    
    # 截取新生成的部分
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    print("="*40)
    print("🤖 经过 LoRA 微调后的 AI 医生回答：\n")
    print(response)
    print("="*40)