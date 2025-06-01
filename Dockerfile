# === 1. 使用完整版基础镜像（比 slim 更快构建 PyTorch 项目） ===
FROM python:3.10

# === 2. 设置容器工作目录 ===
WORKDIR /app

# === 3. 单独复制依赖文件，利用缓存机制优化构建速度 ===
COPY requirements.txt .

# === 4. 安装 Python 依赖项（PyTorch, LangChain, Streamlit 等） ===
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# === 5. 复制项目所有文件（包含模型和txt）到容器内 ===
COPY . .

# ✅ 6. 创建 chroma_db 文件夹（如不存在）
RUN mkdir -p /app/chroma_db



# ✅ 8. 设置 Streamlit 启动默认参数
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# ✅ 9. 启动 Streamlit 应用
CMD ["streamlit", "run", "app/main.py"]
