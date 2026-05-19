"""
Vector Store Management
Manages vector embeddings and similarity search using ChromaDB for RAG applications.
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os


class VectorStore:
    """
    Wrapper class for managing vector databases with ChromaDB.
    """
    
    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: str = "./chroma_db",
        embedding_model: str = "huggingface"
    ):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the vector database
            embedding_model: "openai" or "huggingface"
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Initialize embeddings
        if embedding_model == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY environment variable not set")
            self.embeddings = OpenAIEmbeddings()
        else:
            # Use HuggingFace embeddings (free, no API key needed)
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        
        # Initialize ChromaDB vector store
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of langchain Document objects
            
        Returns:
            List of document IDs
        """
        return self.vector_store.add_documents(documents)
    
    def similarity_search(
        self,
        query: str,
        k: int = 5
    ) -> List[Document]:
        """
        Search for documents similar to the query.
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of similar Document objects
        """
        return self.vector_store.similarity_search(query, k=k)
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5
    ) -> List[tuple]:
        """
        Search for documents with similarity scores.
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of tuples (Document, score)
        """
        return self.vector_store.similarity_search_with_score(query, k=k)
    
    
    def delete_collection(self):
        """Delete the current collection."""
        self.vector_store.delete_collection()
    
    def get_collection_count(self) -> int:
        """Get the number of documents in the collection."""
        return self.vector_store._collection.count()
    
    def as_retriever(self, **kwargs):
        """
        Return the vector store as a retriever.
        """
        return self.vector_store.as_retriever(**kwargs)
    
    def delete_all(self):
        """Clear all documents from the collection."""
        results = self.vector_store.get()
        if results and results.get('ids'):
            self.vector_store.delete(ids=results['ids'])


def create_vector_store(
    documents: List[Document],
    collection_name: str = "documents",
    persist_directory: str = "./chroma_db",
    embedding_model: str = "huggingface",
    clear_existing: bool = False
) -> VectorStore:
    """
    Create and populate a vector store with documents.
    
    Args:
        documents: List of langchain Document objects
        collection_name: Name of the ChromaDB collection
        persist_directory: Directory to persist the vector database
        embedding_model: "openai" or "huggingface"
        clear_existing: Whether to clear existing data before adding
        
    Returns:
        Initialized VectorStore with documents added
    """
    vector_store = VectorStore(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_model=embedding_model
    )
    
    if clear_existing:
        vector_store.delete_all()
        print(f"[INFO] Cleared existing collection: {collection_name}")
    
    if documents:
        vector_store.add_documents(documents)
        print(f"[OK] Added {len(documents)} documents to vector store")
    
    return vector_store


def create_vector_db(chunks: List[Document]):
    """
    Create a vector database from document chunks using OpenAI embeddings.
    
    Args:
        chunks: List of langchain Document objects
        
    Returns:
        Chroma vector database instance
    """
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(chunks, embeddings)
    return db


if __name__ == "__main__":
    # Example usage
    from document_loader import load_docx_directory
    
    # Load documents
    docs = load_docx_directory("./documents")
    
    # Create vector store
    vs = create_vector_store(docs)
    
    # Search example
    query = "What is machine learning?"
    results = vs.similarity_search(query, k=3)
    
    print(f"\nSearch results for: '{query}'")
    for i, doc in enumerate(results):
        print(f"\n{i+1}. {doc.page_content[:200]}")
