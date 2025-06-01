mindmap
  root((app/main.py))
    配置
      set_page_config
    导入
      情绪识别工具
        model_utils.load_emotion_model
        model_utils.predict_emotion
        dog_breed_utils.predict_breed
      RAG 向量库
        RAG.config.CHROMA_DB_PATH
        RAG.config.EMBEDDING_MODEL_PATH
        HuggingFaceEmbeddings
        Chroma
        RetrievalQA
      建议生成 & LLM
        OpenAI API (gen_advice_openai, get_openai_llm)
        本地 Lora DeepSeek (gen_advice_lora, get_local_llm)
    UI 布局
      标题(title)
      侧边栏(selectbox: 语言模型)
    图片上传 & 情绪识别
      file_uploader
      保存文件(upload_dir)
      显示图片(st.image)
      识别品种(predict_breed)
      识别情绪(predict_emotion)
      显示结果(st.success)
      📘 情绪建议
        根据 model_choice 调用 gen_advice
    分隔线(st.markdown("---"))
    知识库问答
      文本输入(st.text_input)
      查询存在时
        加载嵌入模型(HuggingFaceEmbeddings)
        打开已有向量库(Chroma)
        构建检索器(as_retriever)
        选择 LLM(OpenAI 或 本地)
        构建检索问答链(RetrievalQA)
        执行问答(qa.run)
      显示回答(st.write)
