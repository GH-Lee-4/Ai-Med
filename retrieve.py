from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

db = FAISS.load_local("vector_store", OpenAIEmbeddings())

def retrieve_context(query, k=3):
    docs = db.similarity_search(query, k=k)
    return "\n\n".join(d.page_content for d in docs)
