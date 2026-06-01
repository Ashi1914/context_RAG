from langchain_groq import ChatGroq

from app.database.vector_store import VectorStore

from app.core.config import (
    MODEL_NAME,
    TEMPERATURE,
    MAX_TOKENS,
    GROQ_API_KEY
)


vector_store_instance = VectorStore(
    collection_name="rag_documents",
    persist_directory="./chroma_db"
)


def get_vector_store() -> VectorStore:

    return vector_store_instance


llm_instance = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name=MODEL_NAME,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS
)


def get_llm() -> ChatGroq:

    return llm_instance