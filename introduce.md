

Title

Dog Emotion and Breed Recognition with LLM-Based Behavioral Advice via LangChain, Custom Deep Models, and Dockerized RAG Pipeline

Task

Develop a containerized multimodal system that allows users to upload dog images and first automatically recognize the dog's breed and emotional state using deep learning models.
Based on the recognition results, the system then generates structured five-part behavioral advice in natural English using large language models (LLMs).

* Emotion recognition is performed using a custom-trained ResNet-18 model, fine-tuned on a manually labeled emotion dataset.
* Breed classification is carried out using a deep learning model trained on the Stanford Dogs Dataset.

Once the breed and emotion are identified, they are injected into a LangChain-managed PromptTemplate, which is then passed to one of three selectable LLM backends for advice generation:

* OpenAI GPT-4 via API
* LoRA-finetuned DeepSeek-R1-Distill-Qwen-1.5B, hosted locally
* Qwen3-32B-AWQ, accessed via an OpenAI-compatible API endpoint

After receiving the initial advice, users can continue interacting with the system via chat-style follow-up Q\&A, powered by a LangChain RetrievalQA chain using a Chroma vector database. The vector store is constructed from Wikipedia articles on dog breeds and behavior, which are crawled, chunked into text segments, and embedded using the `BAAI/bge-small-en-v1.5` model.

The system is fully Dockerized, enabling reproducible, GPU-accelerated deployment. The frontend is built using Streamlit, supporting image upload, model backend switching, session-persistent dialogue, and real-time user feedback.

---

## ✅ 中文版

### 标题

**基于 LangChain、深度学习模型与 Docker 化 RAG 流程的狗狗情绪与品种识别行为建议系统**

### 任务描述

开发一个多模态容器化系统，允许用户上传狗狗图片，并**首先使用深度学习模型自动识别狗的品种和情绪状态**。
系统随后基于识别结果，调用大语言模型（LLM）生成**结构化的英文五段式行为建议**。

* **情绪识别**由一个**自定义训练的 ResNet-18 模型**完成，该模型在手动标注的情绪数据集上进行微调；
* **品种识别**使用一个基于**Stanford Dogs 数据集训练的深度学习模型**。

识别结果将被注入到一个由 \*\*LangChain 管理的提示词模板（PromptTemplate）\*\*中，然后传递至三种可选的大语言模型后端之一进行建议生成：

* 通过 API 调用的 **OpenAI GPT-4**
* 本地部署的 **LoRA 微调版 DeepSeek-R1-Distill-Qwen-1.5B**
* 通过 OpenAI 兼容 API 接口调用的 **Qwen3-32B-AWQ**

建议生成后，用户还可以通过**对话形式继续提出问题**，该功能由 **LangChain 的 RetrievalQA 链**支持，后端为一个 **Chroma 向量数据库**。该数据库由**爬取自 Wikipedia 的狗狗品种与行为相关文章**构建，经文本切块后使用 **`BAAI/bge-small-en-v1.5`** 模型嵌入生成。

系统整体采用 **Docker 容器化部署**，支持 GPU 加速和跨平台复现。前端基于 **Streamlit** 开发，提供图像上传、模型切换、对话状态管理及实时反馈功能。
