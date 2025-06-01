import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("开始加载模型（无量化）...")

# ✅ 模型路径（确认为完整本地路径）
MODEL_PATH = "/app/LLM_Models/DeepSeek_R1_Distill_Qwen_1_5B"

# ✅ 加载模型（无量化，使用 float16 推理）
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",  # 自动分配到 GPU（如果可用）
)

print("模型加载完成，当前设备：", next(model.parameters()).device)

# ✅ 加载 tokenizer（使用与模型匹配的目录）
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

# ✅ 构造输入
inputs = tokenizer("你好吗？", return_tensors="pt").to(model.device)

# ✅ 推理生成
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=20)

print("输出结果：")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
