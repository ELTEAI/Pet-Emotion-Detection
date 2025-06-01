嵌入模型
BAAI/bge-small-en-v1.5
小型、快、效果好，支持英文任务极好（你的狗狗种类/情绪分析文本是英文+中文混合）
体积小（100M左右），适合 Docker 内存




使用Chroma向量数据库
特点：轻量、Python原生、直接在本地目录持久化，不需要单独部署数据库服务器。

Docker环境中也非常适合，只需简单挂载 volume。

配合 LangChain 支持极好。

RAG 流程测试非常快，开发周期短。

适合个人项目 / 中小规模数据集。

👉 可以用 chromadb Python包直接起服务，非常简单。



重排序模型：BAAI/bge-reranker-base
🚀 目前开源最好用的英文 reranker，支持英文/中英文任务，速度快，效果极好