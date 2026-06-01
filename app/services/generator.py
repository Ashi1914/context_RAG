from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate


MAX_CONTEXT_CHARS = 12000


class RAGGenerator:

    def __init__(
        self,
        vector_store,
        llm,
        k: int = 5
    ):

        self.vector_store = vector_store
        self.llm = llm
        self.k = k

        # Create retriever once
        self.retriever = (
            self.vector_store.as_retriever(
                k=self.k
            )
        )

        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a precise retrieval-augmented AI assistant.

You are given excerpts from documents. Each excerpt is labeled with its Section (heading and sub-heading) so you know exactly where in the document it comes from.

Use the section labels to understand the document structure and give accurate, section-aware answers.

Rules:
- Answer ONLY using the provided context.
- If a question asks about a specific heading or sub-heading, focus on that section.
- Do NOT use outside knowledge or fabricate information.
- If the answer is not found in the context, say: "I could not find the answer in the provided documents."
- When helpful, mention which section the answer comes from.

Context:
{context}

Question:
{question}

Answer:"""
        )

    # -----------------------------------
    # Retrieve Documents
    # -----------------------------------

    async def retrieve_documents(
        self,
        query: str
    ) -> List[Document]:

        return await self.retriever.ainvoke(query)

    # -----------------------------------
    # Build Context with Section Labels
    # -----------------------------------

    def build_context(
        self,
        documents: List[Document]
    ) -> str:

        context_parts = []

        for i, doc in enumerate(documents):
            metadata = doc.metadata

            # Extract heading info from metadata
            heading = metadata.get("heading", "")
            subheading = metadata.get("subheading", "")
            file_name = metadata.get("file_name", "Unknown file")
            page = metadata.get("page", "")

            # Build a rich source label
            if subheading:
                section_label = f"[{file_name}] Section: {heading} › {subheading}"
            elif heading:
                section_label = f"[{file_name}] Section: {heading}"
            else:
                section_label = f"[{file_name}]"

            if page:
                section_label += f" (Page {page + 1})"

            # Combine label + content
            chunk_text = f"--- Source {i+1}: {section_label} ---\n{doc.page_content}"
            context_parts.append(chunk_text)

        full_context = "\n\n".join(context_parts)
        return full_context[:MAX_CONTEXT_CHARS]

    # -----------------------------------
    # Generate Answer
    # -----------------------------------

    async def generate(
        self,
        query: str
    ):

        try:

            docs = await self.retrieve_documents(query)

            if not docs:
                return {
                    "answer": "No relevant documents found. Please upload documents first.",
                    "sources": []
                }

            context = self.build_context(docs)

            prompt = self.prompt_template.format(
                context=context,
                question=query
            )

            response = await self.llm.ainvoke(prompt)

            return {
                "answer": response.content,
                "sources": [
                    {
                        "content": doc.page_content[:300],
                        "metadata": doc.metadata
                    }
                    for doc in docs
                ]
            }

        except Exception as e:

            return {
                "answer": f"Generation failed: {str(e)}",
                "sources": []
            }