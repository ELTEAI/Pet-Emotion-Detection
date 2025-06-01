import os
import re
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain

# —— 强制使用离线模式 ——
os.environ["TRANSFORMERS_OFFLINE"] = "1"

MODEL_PATH = "/app/LLM_Models/Lora_Add_DeepSeek_R1_Distill_Qwen_1_5B"
device = "cuda" if torch.cuda.is_available() else "cpu"

# —— 加载 tokenizer ——
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True,
    use_fast=True
)

# —— 使用 4bit 量化加速加载 ——
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True,
    quantization_config=bnb_config,
    device_map="auto"
).eval()

# —— 构建共享 pipeline，扩大生成长度 —— 
text_gen = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    max_new_tokens=1024,       # 将新生成的最大 token 数扩大到 512
    max_length=2048,          # 整体最大长度（包括输入）
    do_sample=True,
    temperature=0.4,
    top_p=0.9,
)

llm = HuggingFacePipeline(pipeline=text_gen)

# —— 从外部文件加载 PromptTemplate ——
PROMPT_PATH = Path("app/prompts/emotion_prompt_en.txt")
with PROMPT_PATH.open("r", encoding="utf-8") as f:
    prompt_str = f.read()

prompt_template = PromptTemplate(
    input_variables=["breed", "emotion"],
    template=prompt_str
)

# —— 后处理函数：去掉思考内容，只保留最终回答 —— 
def strip_thoughts(text: str) -> str:
    # 删除 <think>…</think> 区块
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    # 删除以“思考”开头的行
    text = re.sub(r"^思考[^\n]*\n?", "", text, flags=re.M)
    return text.strip()

# ✅ 生成狗狗情绪建议（含后处理）
def generate_emotion_advice(breed: str, emotion: str) -> str:
    chain = LLMChain(llm=llm, prompt=prompt_template)
    raw_output = chain.run(breed=breed, emotion=emotion)
    clean_output = strip_thoughts(raw_output)
    return clean_output

# ✅ 提供给 LangChain QA 用的 LLM 接口
def get_local_llm() -> HuggingFacePipeline:
    return llm




"""
import os
# —— 强制离线模式，防止误连 Hugging Face —— 
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "/app/LLM_Models/Lora+_DeepSeek_R1_Distill_Qwen_1_5B"

print("开始加载模型...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, trust_remote_code=True, local_files_only=True
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
).eval()

# 情绪映射：英文->中文
emotion_map = {
    "happy": "开心",
    "sad": "难过",
    "angry": "愤怒",
    "anxious": "焦虑",
    "relaxed": "放松",
}

def normalize_breed(breed: str) -> str:
    return " ".join(w.capitalize() for w in breed.split("_"))

# PromptTemplate：只生成建议正文，不负责检测提示
template = PromptTemplate.from_template(
"""
#System: 你是一位专业的宠物心理专家，只输出最终建议，**绝不**输出任何思考、分析过程或过渡短语，禁止使用“好的”、“我现在需要…”、“接下来”、“然后”、“首先”等。

#User:
#请根据犬种“{breed_name_en}”和当前情绪“{emotion_cn}”，用中文写一段 200–300 字、流畅自然的宠物心理建议，内容涵盖可能原因、安抚方法和预防策略，可适当使用😊、🐾等表情增添亲和力，但不要过度。请直接给出建议正文，不要编号或分点，也不要任何前言。

#Answer:
""")

# 构建文本生成 pipeline（不使用停止策略与禁用词）
text_gen = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    max_new_tokens=256,    # ← 这里指定长度
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
)

def generate_emotion_advice(breed: str, emotion: str) -> str:
    # 规范化品种与情绪描述
    breed_clean = normalize_breed(breed)
    emotion_cn = emotion_map.get(emotion.lower(), emotion)
    # “思考”部分：检测提示（普通字体）
    thinking = f"🐶 我检测到这是一只 {breed_clean}，它现在可能感到 {emotion_cn}。\n\n"
    # 模板 prompt 仅生产建议正文
    prompt = template.format(breed_name_en=breed_clean, emotion_cn=emotion_cn)
    # 生成正文
    output = text_gen(prompt, return_full_text=False)[0]["generated_text"].strip()
    # “正式内容”部分加粗
    answer = f"**{output}**"
    return thinking + answer

def get_local_llm() -> HuggingFacePipeline:
    llm_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        device_map="auto"
    )
    return HuggingFacePipeline(pipeline=llm_pipe)

# —— 示例调用 —— 
if __name__ == "__main__":
    advice = generate_emotion_advice("berner_mountain_dog", "happy")
    print(advice)
"""