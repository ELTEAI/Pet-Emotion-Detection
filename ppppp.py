# app/main.py
""""""
import redis

import os
import streamlit as st
from streamlit_chat import message

# —— 页面配置 —— #
st.set_page_config(page_title="🐶 Dog Emotions + Chat Assistant", layout="centered")
# —— 访问统计 —— #
try:
    r = redis.Redis(host="dog-ai-redis", port=6379, db=0)
    visits = r.incr("visit_count")
except Exception as e:
    visits = "❌ Redis 连接失败"
    print(f"[访问计数错误] {e}")

# —— 展示访问次数 —— #
st.sidebar.markdown(f"👀 **Number of visits：{visits}**")


# —— 导入模型与工具 —— #
from model_utils import load_emotion_model, predict_emotion
from dog_breed_utils import predict_breed
from RAG.config import CHROMA_DB_PATH, EMBEDDING_MODEL_PATH
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# —— 获取当前页面参数 —— #
query_params = st.experimental_get_query_params()
#query_params = st.query_params
current_model = query_params.get("model", ["openai"])[0]

# —— 侧边栏：切换模型 —— #
model_choice = st.sidebar.selectbox(
    "🤖 Choosing a language model",
    ["OpenAI API", "本地 Lora DeepSeek", "Qwen3 API"],
    index=0 if current_model == "openai" else 1 if current_model == "local" else 2
)
# 旧代码：使用 st.experimental_set_query_params
# 当模型选择改变时，更新 URL 参数
if model_choice == "OpenAI API" and current_model != "openai":
    st.experimental_set_query_params(model="openai")
    #st.query_params = {"model": "openai"}
    st.rerun()
elif model_choice == "本地 Lora DeepSeek" and current_model != "local":
    st.experimental_set_query_params(model="local")
    #st.query_params = {"model": "local"}
    st.rerun()
elif model_choice == "Qwen3 API" and current_model != "qwen3":
    st.experimental_set_query_params(model="qwen3"); 
    #st.query_params = {"model": "qwen3"}
    st.rerun()
# 根据当前模型导入相应的工具
if current_model == "openai":
    from Open_AI_API.langchain_utils import generate_emotion_advice, get_openai_llm
    get_llm = get_openai_llm
elif current_model == "local":
    from lora_deepseek_r1_distill_qwen_1_5b.langchain_utils_lora_deepseek_r1_distill_qwen_1_5b import (
        generate_emotion_advice, get_local_llm
    )
    get_llm = get_local_llm
elif current_model == "qwen3":
    from qwen3.langchain_utils_qwen3 import generate_emotion_advice, get_qwen3_llm
    get_llm = get_qwen3_llm

# —— 初始化会话状态 —— #
model_prefix = "openai_" if current_model == "openai" else "local_"
if f"{model_prefix}messages" not in st.session_state:
    st.session_state[f"{model_prefix}messages"] = []
if f"{model_prefix}breed" not in st.session_state:
    st.session_state[f"{model_prefix}breed"] = None
if f"{model_prefix}emotion" not in st.session_state:
    st.session_state[f"{model_prefix}emotion"] = None
if f"{model_prefix}image_processed" not in st.session_state:
    st.session_state[f"{model_prefix}image_processed"] = False

# —— 标题 —— #
st.title(f"🐾 Dog Emotion Recognition & Smart Chat Assistant ({model_choice})")

# —— 1. 上传图片 & 识别 —— #
uploaded_file = st.file_uploader(
    "📤 Upload a photo of your dog for analysis", type=["jpg", "jpeg", "png"]
)
if uploaded_file:
    # 保存并展示图片
    upload_dir = os.path.join("app", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    img_path = os.path.join(upload_dir, uploaded_file.name)
    with open(img_path, "wb") as f:
        f.write(uploaded_file.read())
    st.image(img_path, caption="你上传的狗狗照片", width=400)

    # 识别品种与情绪
    with st.spinner("🐕 Identifying…"):
        breed = predict_breed(img_path)
        emotion = predict_emotion(img_path, load_emotion_model())
        # 更新状态
        st.session_state[f"{model_prefix}breed"] = breed
        st.session_state[f"{model_prefix}emotion"] = emotion

    st.success(f"🔎 Identification result: Breed - **{breed}**；Emotion: - **{emotion}**")

    # 仅首次处理图片时才生成建议
    if not st.session_state[f"{model_prefix}image_processed"]:
        advice = generate_emotion_advice(breed, emotion)
        bot_msg = (
            f"🐶 I detected that this was a **{breed}**，"
            f"It may now feel **{emotion}**。\n\n👉 建议：{advice}"
        )
        st.session_state[f"{model_prefix}messages"].append({
            "role": "assistant", "content": bot_msg
        })
        st.session_state[f"{model_prefix}image_processed"] = True

# —— 2. 聊天式问答 —— #
query = st.chat_input("Please enter a question, for example: Are Samoyeds afraid of heat?")
if query:
    breed = st.session_state[f"{model_prefix}breed"]
    # 自动补全品种上下文
    if breed and breed not in query:
        query = f"{breed}is dog`s breed,{query}"

    # 追加用户输入
    st.session_state[f"{model_prefix}messages"].append({"role": "user", "content": query})

    # RAG 检索 + LLM 回答
    with st.spinner("🤔 Answering..."):
        embed_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)
        vectordb = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embed_model,
            collection_name="dog_wiki"
        )
        retriever = vectordb.as_retriever(search_kwargs={"k": 5})
        llm = get_llm()
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False
        )
        answer = qa.run(query)

    # 追加机器人回答
    st.session_state[f"{model_prefix}messages"].append({"role": "assistant", "content": answer})

# —— 3. 统一渲染对话 —— #
if st.session_state[f"{model_prefix}messages"]:
    st.markdown("---")
    st.subheader("💬 Chat History")
    for idx, msg in enumerate(st.session_state[f"{model_prefix}messages"]):
        message(
            msg["content"],
            is_user=(msg["role"] == "user"),
            key=f"{model_prefix}msg_{idx}"
        )  代码中Cloudflare功能是哪个 