from sentence_transformers import SentenceTransformer

def load_embedding_model(model_name):
    model = SentenceTransformer(model_name)
    return model

def embed_texts(model, texts):
    return model.encode(texts, show_progress_bar=True)
