# app/RAG/split_txt_to_chunks.py

import os
import glob
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from app.RAG.embedding_utils import load_embedding_model, embed_texts
from app.RAG.config      import WIKI_DOGS_PATH, CHROMA_DB_PATH, EMBEDDING_MODEL_PATH

def main():
    # 1. 验证路径
    if not os.path.isdir(WIKI_DOGS_PATH):
        raise FileNotFoundError(f"WIKI_DOGS_PATH 不存在: {WIKI_DOGS_PATH}")
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)

    # 2. 读取所有 txt 文件
    txt_files = glob.glob(os.path.join(WIKI_DOGS_PATH, "*.txt"))
    print(f"📄 找到 {len(txt_files)} 个描述文件")
    docs = []
    for fn in tqdm(txt_files, desc="读取文件"):
        with open(fn, "r", encoding="utf-8") as f:
            docs.append({"text": f.read(), "source": os.path.basename(fn)})

    # 3. 切分成 chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks, metadatas = [], []
    for doc in tqdm(docs, desc="切分文本"):
        segs = splitter.split_text(doc["text"])
        chunks.extend(segs)
        metadatas.extend([{"source": doc["source"]}] * len(segs))
    print(f"🧩 总共切分出 {len(chunks)} 个 chunk")

    # 4. 加载本地 HuggingFace 嵌入模型
    print(f"🤖 正在加载嵌入模型: {EMBEDDING_MODEL_PATH}")
    embed_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)

    # 5. 实例化 Chroma 向量库 wrapper
    vectordb = Chroma(
        embedding_function=embed_model,
        persist_directory=CHROMA_DB_PATH,
        collection_name="dog_wiki"
    )

    # 6. 分批写入，避免超过底层 upsert 限制
    ids = [str(i) for i in range(len(chunks))]
    batch_size = 5000  # 小于底层最大 5461
    print(f"🔄 分批 (batch_size={batch_size}) 写入向量库…")
    for start in tqdm(range(0, len(chunks), batch_size), desc="写入批次"):
        end = min(start + batch_size, len(chunks))
        vectordb.add_texts(
            texts=chunks[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )

    # 7. Persist
    vectordb.persist()
    print("✅ 向量库构建完成，保存在", CHROMA_DB_PATH)


if __name__ == "__main__":
    main()
