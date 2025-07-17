from langchain.embeddings import HuggingFaceEmbeddings

def embeddings_function():
    return HuggingFaceEmbeddings(model_name="models/all-MiniLM-L12-v2")

#or load online
# from sentence_transformers import SentenceTransformer
# def embeddings_function():
#     return SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
