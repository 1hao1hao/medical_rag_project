import torch
import json
from datasets import Dataset
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType

# ==========================================
# 🚀 终极环境兼容补丁 (Monkey Patch)
# 强行注入 transformers 4.36.2 缺失的属性，绕过报错
# ==========================================
transformers.modeling_utils.PreTrainedModel.all_tied_weights_keys = property(lambda self: {})

# ==========================================
# 1. 核心配置区
# ==========================================
MODEL_NAME = "THUDM/chatglm3-6b"
DATASET_PATH = "medical_lora_dataset.json"
OUTPUT_DIR = "./chatglm3_lora_medical"

print(">>> 1. 正在加载 Tokenizer 和 基础大模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# 加载配置并注入 max_length
config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
if not hasattr(config, "max_length"):
    config.max_length = 8192

# 加载模型
model = AutoModel.from_pretrained(MODEL_NAME, config=config, trust_remote_code=True).half().cuda()

# ==========================================
# 2. 注入 LoRA 适配器
# ==========================================
print(">>> 2. 正在配置并注入 LoRA 适配器...")
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query_key_value", "dense_h_to_4h", "dense_4h_to_h"]
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters() 

# ==========================================
# 3. 数据预处理
# ==========================================
print(">>> 3. 正在处理训练数据...")
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)                         # 数据集由 05_build_lora_dataset.py 生成

def process_data(example):
    text = f"<|system|>\n{example['instruction']}\n<|user|>\n{example['input']}\n<|assistant|>\n{example['output']}"
    tokenized = tokenizer(text, max_length=512, truncation=True, padding="max_length")
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

dataset = Dataset.from_list(raw_data)
tokenized_dataset = dataset.map(process_data, remove_columns=dataset.column_names)

# ==========================================
# 4. 配置训练参数并启动训练
# ==========================================
print(">>> 4. 开始进行 LoRA 微调 (过拟合测试)...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1, 
    learning_rate=2e-4,            
    num_train_epochs=100,           
    logging_steps=1,               
    save_strategy="no",            
    fp16=True                      
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()

print(f">>> 5. 训练完成！正在保存 LoRA 权重至 {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
print("✅ 恭喜！纯手写 LoRA 微调链路已彻底跑通！")