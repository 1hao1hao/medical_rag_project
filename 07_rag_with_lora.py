import os

import torch
import transformers
from peft import PeftModel
from sentence_transformers import CrossEncoder
from transformers import AutoConfig, AutoModel, AutoTokenizer

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

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
LORA_DIR = "./chatglm3_lora_medical"

# 为了降低 24GB 显卡上的 OOM 风险，默认将检索放在 CPU，
# 生成阶段优先使用 GPU。你也可以通过环境变量手动覆盖：
# RETRIEVAL_DEVICE=cuda GENERATION_DEVICE=cuda python 07_rag_with_lora.py
RETRIEVAL_DEVICE = os.getenv("RETRIEVAL_DEVICE", "cpu")
GENERATION_DEVICE = os.getenv(
    "GENERATION_DEVICE",
    "cuda" if torch.cuda.is_available() else "cpu",
)


# ==========================================
# 2. 检索模块
# ==========================================
def get_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": RETRIEVAL_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)


def advanced_retrieval(query, vector_db, reranker_model, top_k_recall=15, top_k_rerank=3):
    docs = vector_db.similarity_search(query, k=top_k_recall)
    cross_input = [[query, doc.page_content] for doc in docs]
    scores = reranker_model.predict(cross_input)
    scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return scored_docs[:top_k_rerank]


def load_generation_model():
    print(f">>> 2. 正在加载基础大模型... ({GENERATION_DEVICE})")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    config = AutoConfig.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)

    if not hasattr(config, "max_length"):
        config.max_length = getattr(config, "seq_length", 8192)

    base_model = AutoModel.from_pretrained(
        BASE_MODEL_NAME,
        config=config,
        trust_remote_code=True,
    )
    if GENERATION_DEVICE.startswith("cuda"):
        base_model = base_model.half()
    base_model = base_model.to(GENERATION_DEVICE)

    print(">>> 3. 正在挂载 LoRA 微调权重...")
    model = PeftModel.from_pretrained(base_model, LORA_DIR)
    model = model.eval()
    return tokenizer, model


# ==========================================
# 3. 主程序
# ==========================================
if __name__ == "__main__":
    print(f">>> 1. 正在初始化检索组件... ({RETRIEVAL_DEVICE})")
    vector_db = get_vector_store()
    reranker = CrossEncoder(RERANK_MODEL_NAME, device=RETRIEVAL_DEVICE)

    user_query = "高血压患者每天吃盐不能超过多少克？"
    print(f"\n用户提问: {user_query}")
    print(">>> 正在检索知识库...")
    top_docs = advanced_retrieval(user_query, vector_db, reranker)

    context = "\n".join(
        [f"[片段 {i+1}] {doc.page_content}" for i, (doc, _) in enumerate(top_docs)]
    )

    instruction = (
        "你是一个专业的医疗AI助手。请严格根据【参考资料】回答问题。"
        "必须以“您好，我是AI医生。”开头。"
    )
    input_text = f"【参考资料】\n{context}\n\n【用户问题】\n{user_query}"
    prompt = f"<|system|>\n{instruction}\n<|user|>\n{input_text}\n<|assistant|>\n"

    # 在加载 ChatGLM3 之前，先释放检索阶段占用的内存和显存
    del reranker
    del vector_db
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tokenizer, model = load_generation_model()

    print(">>> 正在生成回答...\n")
    inputs = tokenizer(prompt, return_tensors="pt").to(GENERATION_DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.1,
            do_sample=False,
        )

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )

    print("=" * 40)
    print("经过 LoRA 微调后的 AI 医生回答：\n")
    print(response)
    print("=" * 40)
