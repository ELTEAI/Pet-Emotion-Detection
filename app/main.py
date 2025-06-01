import os
import redis
import torch
import streamlit as st
from pathlib import Path
from streamlit_chat import message
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from model_utils import load_emotion_model, predict_emotion
from dog_breed_utils import predict_breed
from RAG.config import CHROMA_DB_PATH, EMBEDDING_MODEL_PATH

# —— 强制使用离线模式 ——
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# —— 页面配置 —— #
st.set_page_config(page_title="🐶 Dog Emotions + Chat Assistant", layout="centered")

# —— Redis 访问统计（带超时和备用方案）—— #
try:
    r = redis.Redis(
        host="dog-ai-redis",
        port=6379,
        db=0,
        socket_timeout=3,  # 3秒超时
        socket_connect_timeout=3
    )
    visits = r.incr("visit_count")
except redis.exceptions.ConnectionError:
    # Redis连接失败时使用会话状态作为后备
    if "visit_count" not in st.session_state:
        st.session_state.visit_count = 0
    st.session_state.visit_count += 1
    visits = st.session_state.visit_count
    st.sidebar.warning("⚠️ Redis connection failed, use local access statistics")
except Exception as e:
    visits = "Statistics not available"
    st.sidebar.error(f"❌ Access statistics error: {str(e)}")

# —— 展示访问次数 —— #
st.sidebar.markdown(f"👀 **Number of visits：{visits}**")

# —— RAG 配置 —— #
#CHROMA_DB_PATH = "path/to/your/chroma_db"  # 替换为实际路径
#EMBEDDING_MODEL_PATH = "path/to/your/embedding_model"  # 替换为实际路径

# —— 缓存资源加载 —— #
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)

@st.cache_resource
def load_vector_db():
    return Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=load_embedding_model(),
        collection_name="dog_wiki"
    )

# —— 模型切换逻辑 —— #
if "model_choice" not in st.session_state:
    st.session_state.model_choice = "OpenAI API"

model_choice = st.sidebar.selectbox(
    "🤖 Choosing a language model",
    ["OpenAI API", "本地 Lora DeepSeek", "Qwen3 API"],
    index=["OpenAI API", "本地 Lora DeepSeek", "Qwen3 API"].index(st.session_state.model_choice)
)

# 模型映射
model_map = {
    "OpenAI API": "openai",
    "本地 Lora DeepSeek": "local",
    "Qwen3 API": "qwen3"
}

# 检查模型是否切换
if model_choice != st.session_state.model_choice:
    st.session_state.model_choice = model_choice
    st.rerun()

current_model = model_map[model_choice]

# —— 初始化统一会话状态 —— #
if "app_state" not in st.session_state:
    st.session_state.app_state = {
        "messages": [],
        "breed": None,
        "emotion": None,
        "image_processed": False
    }

state = st.session_state.app_state

# —— 标题 —— #
st.title(f"🐾 Dog emotion recognition and intelligent chat assistant ({model_choice})")

# —— 1. 上传图片 & 识别 —— #
uploaded_file = st.file_uploader(
    "📤 Upload a photo of your dog for analysis", type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    # 重置图片处理状态
    state["image_processed"] = False
    state["breed"] = None
    state["emotion"] = None
    
    # 保存并展示图片
    upload_dir = os.path.join("app", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    img_path = os.path.join(upload_dir, uploaded_file.name)
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(img_path, caption="Photos of your dog that you uploaded", width=400)

    # 带状态指示器的识别过程
    with st.status("🔍 Analyzing dog photos...", expanded=True) as status:
        st.write("Identify the species...")
        breed = predict_breed(img_path)
        st.write("Analyzing emotions...")
        emotion = predict_emotion(img_path, load_emotion_model())
        
        # 更新状态
        state["breed"] = breed
        state["emotion"] = emotion
        
        status.update(label="✅ Analysis Complete!", state="complete")
    
    st.success(f"🔎 Identification result: Breed - **{breed}**；Emotion - **{emotion}**")

    # 仅首次处理图片时才生成建议
    if not state["image_processed"]:
        # 动态导入相应模型的函数
        if current_model == "openai":
            from Open_AI_API.langchain_utils import generate_emotion_advice
        elif current_model == "local":
            from lora_deepseek_r1_distill_qwen_1_5b.langchain_utils_lora_deepseek_r1_distill_qwen_1_5b import generate_emotion_advice
        elif current_model == "qwen3":
            from qwen3.langchain_utils_qwen3 import generate_emotion_advice
        
        with st.spinner("💡 Generating sentiment suggestions..."):
            advice = generate_emotion_advice(breed, emotion)
            bot_msg = (
                f"🐶 I detected this is a**{breed}**，"
                f"It may now feel**{emotion}**。\n\n👉 suggestion：{advice}"
            )
            state["messages"].append({
                "role": "assistant", "content": bot_msg
            })
            state["image_processed"] = True

# —— 空状态处理 —— #
if not state["messages"]:
    st.info("👋 Upload a photo of your dog to start analysis, or ask a question below")

# —— 2. 聊天式问答 —— #
query = st.chat_input("Please enter a question, for example: Is Samoyed afraid of heat?")

if query:
    # 更智能的品种上下文添加
    if state["breed"]:
        breed = state["breed"]
        # 检查品种是否已在问题中提到
        breed_mentioned = any([
            breed in query,
            breed.split()[0] in query,  # 处理多词品种名
            breed.replace(" ", "") in query.replace(" ", "")
        ])
        
        if not breed_mentioned:
            query = f"About{breed}Breed issues：" + query
    
    # 追加用户输入
    state["messages"].append({"role": "user", "content": query})
    
    # 动态导入相应模型的函数
    if current_model == "openai":
        from Open_AI_API.langchain_utils import get_openai_llm
        get_llm = get_openai_llm
    elif current_model == "local":
        from lora_deepseek_r1_distill_qwen_1_5b.langchain_utils_lora_deepseek_r1_distill_qwen_1_5b import get_local_llm
        get_llm = get_local_llm
    elif current_model == "qwen3":
        from qwen3.langchain_utils_qwen3 import get_qwen3_llm
        get_llm = get_qwen3_llm

    # RAG 检索 + LLM 回答
    with st.spinner("🤔 Thinking..."):
        try:
            # 使用缓存的向量数据库
            vectordb = load_vector_db()
            retriever = vectordb.as_retriever(search_kwargs={"k": 5})
            llm = get_llm()
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=False
            )
            answer = qa.run(query)
        except Exception as e:
            st.error(f"Error answering question: {str(e)}")
            answer = "Sorry, there was an error answering the question, please try again later。"

    # 追加机器人回答
    state["messages"].append({"role": "assistant", "content": answer})

# —— 3. 优化后的对话渲染 —— #
if state["messages"]:
    st.markdown("---")
    st.subheader("💬 Conversation history")
    
    # 仅渲染最后10条消息
    for msg in state["messages"][-10:]:
        message(
            msg["content"],
            is_user=(msg["role"] == "user"),
            key=f"{id(msg)}"  # 使用消息内容的ID作为key
        )