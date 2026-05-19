"""
Response Generator
Generates responses using Retrieval-Augmented Generation (RAG).
"""

from typing import List, Tuple, Any

from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

import os

load_dotenv()
if os.getenv("GROQ_API_KEY"):
    # Groq doesn't need a separate client like OpenAI for standard LangChain usage
    pass


class RAGGenerator:
    """
    RAG-based response generator using Groq + ChromaDB.
    """

    def __init__(
        self,
        vector_store,
        model_name: str = "llama-3.3-70b-versatile",
        temperature: float = 0.7,
        max_tokens: int = 1024
    ):
        """
        Initialize RAG generator.

        Args:
            vector_store: Chroma vector store instance
            model_name: Groq model name
            temperature: Creativity level
            max_tokens: Maximum response tokens
        """

        if not os.getenv("GROQ_API_KEY"):
            raise ValueError(
                "GROQ_API_KEY environment variable not set"
            )

        self.vector_store = vector_store

        self.llm = ChatGroq(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Improved RAG prompt
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are an intelligent AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not available in the context,
say:
"I could not find the answer in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""
        )

    def _retrieve_documents(
        self,
        query: str,
        k: int = 5
    ) -> List[Document]:
        """
        Retrieve relevant documents.
        """

        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": k}
        )

        return retriever.invoke(query)

    def _build_context(
        self,
        documents: List[Document]
    ) -> str:
        """
        Convert documents into context string.
        """

        if not documents:
            return "No relevant context found."

        return "\n\n".join(
            doc.page_content
            for doc in documents
        )

    def generate(
        self,
        query: str,
        k: int = 5,
        use_retrieval: bool = True
    ) -> str:
        """
        Generate RAG response.

        Args:
            query: User question
            k: Number of retrieved docs
            use_retrieval: Enable/disable RAG

        Returns:
            Generated answer
        """

        try:

            # LLM only mode
            if not use_retrieval:

                response = self.llm.invoke(query)

                return (
                    response.content
                    if hasattr(response, "content")
                    else str(response)
                )

            # Retrieve docs
            docs = self._retrieve_documents(query, k)

            # Build context
            context = self._build_context(docs)

            # Format prompt
            prompt = self.prompt_template.format(
                context=context,
                question=query
            )

            # Generate response
            response = self.llm.invoke(prompt)

            return (
                response.content
                if hasattr(response, "content")
                else str(response)
            )

        except Exception as e:
            return f"Error generating response: {str(e)}"

    def generate_with_context(
        self,
        query: str,
        k: int = 5
    ) -> Tuple[str, List[Document]]:
        """
        Generate response with source docs.

        Returns:
            Tuple:
            (response, retrieved_documents)
        """

        docs = self._retrieve_documents(query, k)

        response = self.generate(query, k)

        return response, docs

    def batch_generate(
        self,
        queries: List[str],
        k: int = 5
    ) -> List[str]:
        """
        Generate responses for multiple queries.
        """

        return [
            self.generate(query, k)
            for query in queries
        ]


def create_rag_generator(
    vector_store: Any,
    model_name: str = "llama-3.3-70b-versatile",
    temperature: float = 0.7,
    max_tokens: int = 1024
) -> RAGGenerator:
    """
    Factory function for RAGGenerator.
    """

    return RAGGenerator(
        vector_store=vector_store,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )


def generate_content_with_context(
    content_type: str,
    tone: str,
    topic: str,
    keywords: List[str],
    retriever
) -> str:
    """
    Generate contextual content using RAG.
    """

    try:

        from prompts import get_prompt

        # Retrieve relevant documents
        docs = retriever.invoke(topic)

        if not docs:
            return (
                "No relevant context found "
                "for content generation."
            )

        # Build context
        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        # Build user prompt
        user_prompt = get_prompt(
            content_type=content_type,
            tone=tone,
            topic=topic,
            keywords=keywords
        )

        final_prompt = f"""
Use the following context to generate content.

Context:
{context}

Task:
{user_prompt}

Rules:
- Use ONLY the provided context
- Do not hallucinate
- If information is missing, say:
  "Insufficient information available."
"""

        llm = ChatGroq(
            model_name="llama-3.3-70b-versatile",
            temperature=0.7
        )

        response = llm.invoke(final_prompt)

        return (
            response.content
            if hasattr(response, "content")
            else str(response)
        )

    except ImportError:
        return (
            "Error: prompts.py not found."
        )

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":

    from document_loader import load_docx_directory
    from vector_store import create_vector_store

    # Load documents
    docs = load_docx_directory("./documents")

    # Create vector store
    vector_store = create_vector_store(
        documents=docs,
        embedding_model="huggingface",
        clear_existing=False
    )

    # Create RAG generator
    generator = create_rag_generator(vector_store)

    # User query
    query = "What is the main topic?"

    # Generate response
    response, sources = generator.generate_with_context(
        query=query,
        k=3
    )

    print("\nQuery:")
    print(query)

    print("\nResponse:")
    print(response)

    print("\nRetrieved Sources:")
    for i, doc in enumerate(sources, start=1):

        print(f"\nSource {i}:")
        print(doc.page_content[:200])