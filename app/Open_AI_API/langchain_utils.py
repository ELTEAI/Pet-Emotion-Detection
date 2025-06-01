import os
from pathlib import Path
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI  # ✅ 新版 langchain-openai 的 ChatOpenAI

# ✅ 读取 OpenAI Key（可放 .env 文件中）
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# ✅ 读取外部 prompt 文件
PROMPT_PATH = Path("app/prompts/emotion_prompt_en.txt")
with PROMPT_PATH.open("r", encoding="utf-8") as f:
    prompt_str = f.read()

# ✅ 创建 PromptTemplate
prompt_template = PromptTemplate(
    input_variables=["breed", "emotion"],
    template=prompt_str
)

# ✅ 生成狗狗情绪建议
def generate_emotion_advice(breed: str, emotion: str) -> str:
    llm = ChatOpenAI(temperature=0.7, model="gpt-4")
    chain = LLMChain(llm=llm, prompt=prompt_template)
    return chain.run(breed=breed, emotion=emotion)

# ✅ 用于构建 QA chain 的 LLM 接口
def get_openai_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
