import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain.llms.base import LLM
from typing import List, Optional
from pathlib import Path
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# ✅ 加载 .env 配置
load_dotenv()
API_KEY = os.getenv("CUSTOM_API_KEY")
BASE_URL = os.getenv("CUSTOM_API_BASE")

if not API_KEY or not BASE_URL:
    raise ValueError("请在 .env 中配置 CUSTOM_API_KEY 和 CUSTOM_API_BASE")

# ✅ 初始化本地 OpenAI 接口兼容客户端
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# ✅ 加载提示词模板
PROMPT_PATH = Path("app/prompts/emotion_prompt_en.txt")
with PROMPT_PATH.open("r", encoding="utf-8") as f:
    prompt_str = f.read()

prompt_template = PromptTemplate(
    input_variables=["breed", "emotion"],
    template=prompt_str
)

# ✅ 情绪建议函数
def generate_emotion_advice(breed: str, emotion: str) -> str:
    prompt = prompt_template.format(breed=breed, emotion=emotion)
    response = client.chat.completions.create(
        model="Qwen/Qwen3-32B-AWQ",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# ✅ LangChain 封装
class Qwen3LLM(LLM):
    model: str = "Qwen/Qwen3-32B-AWQ"
    temperature: float = 0.3

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature
        )
        return response.choices[0].message.content.strip()

    @property
    def _llm_type(self) -> str:
        return "qwen3_local_api"

def get_qwen3_llm():
    return Qwen3LLM()

"""
import os
from huggingface_hub import InferenceClient
from langchain.llms.base import LLM
from typing import List, Optional
from dotenv import load_dotenv  # ✅ 新增

# ✅ 加载 .env 中的环境变量
load_dotenv()

# ✅ 从环境中获取 Hugging Face API key
HF_API_KEY = os.getenv("HF_API_KEY")
if not HF_API_KEY:
    raise ValueError("未检测到 HF_API_KEY，请确认 .env 配置正确")

# ✅ 创建推理客户端
client = InferenceClient(
    provider="novita",
    api_key=HF_API_KEY
)

# ✅ 用于情绪建议生成
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from pathlib import Path

PROMPT_PATH = Path("app/prompts/emotion_prompt_en.txt")
with PROMPT_PATH.open("r", encoding="utf-8") as f:
    prompt_str = f.read()

prompt_template = PromptTemplate(
    input_variables=["breed", "emotion"],
    template=prompt_str
)

# ✅ 情绪建议
def generate_emotion_advice(breed: str, emotion: str) -> str:
    prompt = prompt_template.format(breed=breed, emotion=emotion)
    response = client.chat.completions.create(
        model="Qwen/Qwen3-32B",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# ✅ QA 任务封装为 LangChain 可用接口
class Qwen3LLM(LLM):
    model: str = "Qwen/Qwen3-32B"
    temperature: float = 0.3

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature
        )
        return response.choices[0].message.content.strip()

    @property
    def _llm_type(self) -> str:
        return "qwen3_inference_api"

def get_qwen3_llm():
    return Qwen3LLM()
"""