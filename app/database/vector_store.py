from typing import List
import os

from langchain_core.documents import Document

from langchain_chroma import Chroma

from langchain_openai import OpenAIEmbeddings

from langchain_huggingface import (
    HuggingFaceEmbeddings
)


class VectorStore:

    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: str = "./chroma_db",
        embedding_model: str = "huggingface"
    ):

        self.collection_name = collection_name

        self.persist_directory = persist_directory

        os.makedirs(
            self.persist_directory,
            exist_ok=True
        )

        self.embeddings = self._load_embeddings(
            embedding_model
        )

        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

    # -----------------------------------
    # Load Embedding Model
    # -----------------------------------

    def _load_embeddings(
        self,
        embedding_model: str
    ):

        if embedding_model == "openai":

            api_key = os.getenv(
                "OPENAI_API_KEY"
            )

            if not api_key:

                raise ValueError(
                    "OPENAI_API_KEY not found"
                )

            return OpenAIEmbeddings(
                api_key=api_key
            )

        return HuggingFaceEmbeddings(
            model_name=
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    # -----------------------------------
    # Add Documents
    # -----------------------------------

    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 100
    ):

        try:

            for i in range(
                0,
                len(documents),
                batch_size
            ):

                batch = documents[
                    i:i + batch_size
                ]

                self.vector_store.add_documents(
                    batch
                )

        except Exception as e:

            raise Exception(
                f"Failed to add documents: {str(e)}"
            )

    # -----------------------------------
    # Similarity Search
    # -----------------------------------

    def similarity_search(
        self,
        query: str,
        k: int = 5
    ):

        try:

            return (
                self.vector_store
                .similarity_search(
                    query=query,
                    k=k
                )
            )

        except Exception as e:

            raise Exception(
                f"Search failed: {str(e)}"
            )

    # -----------------------------------
    # Similarity Search With Score
    # -----------------------------------

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = None
    ):

        try:

            results = (
                self.vector_store
                .similarity_search_with_score(
                    query=query,
                    k=k
                )
            )

            if score_threshold is not None:

                results = [

                    (doc, score)

                    for doc, score in results

                    if score <= score_threshold
                ]

            return results

        except Exception as e:

            raise Exception(
                f"Similarity search failed: {str(e)}"
            )

    # -----------------------------------
    # Retriever
    # -----------------------------------

    def as_retriever(
        self,
        k: int = 5
    ):

        return self.vector_store.as_retriever(

            search_type="mmr",

            search_kwargs={
                "k": k,
                "fetch_k": 20
            }
        )

    # -----------------------------------
    # Delete All Documents
    # -----------------------------------

    def delete_all(self):

        try:

            results = self.vector_store.get()

            ids = results.get(
                "ids",
                []
            )

            if ids:

                self.vector_store.delete(
                    ids=ids
                )

        except Exception as e:

            raise Exception(
                f"Delete failed: {str(e)}"
            )


# -----------------------------------
# Create Vector Store
# -----------------------------------

def create_vector_store(
    documents: List[Document],
    collection_name: str = "documents",
    persist_directory: str = "./chroma_db",
    embedding_model: str = "huggingface",
    clear_existing: bool = False
):

    vector_store = VectorStore(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_model=embedding_model
    )

    if clear_existing:

        vector_store.delete_all()

    if documents:

        vector_store.add_documents(
            documents
        )

    return vector_store