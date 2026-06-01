import os
import tempfile

from typing import List

from fastapi import UploadFile, HTTPException

from app.loaders.document_loader import load_and_split

from app.database.vector_store import (
    VectorStore,
    create_vector_store
)



CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "rag_documents"

# -----------------------------------
# Shared Vector Store (Singleton)
# -----------------------------------

vector_store = VectorStore(
    collection_name=COLLECTION_NAME,
    persist_directory=CHROMA_DIR
)

# -----------------------------------
# Upload Documents
# -----------------------------------

ALLOWED_EXTENSIONS = [".pdf", ".docx"]

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


async def upload_documents(
    files: List[UploadFile]
):

    try:

        documents = []

        for file in files:

            # -----------------------------------
            # Validate file extension
            # -----------------------------------

            filename = file.filename.lower()

            if not any(
                filename.endswith(ext)
                for ext in ALLOWED_EXTENSIONS
            ):

                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {filename}"
                )

            # -----------------------------------
            # Read file
            # -----------------------------------

            content = await file.read()

            # -----------------------------------
            # Validate file size
            # -----------------------------------

            if len(content) > MAX_FILE_SIZE:

                raise HTTPException(
                    status_code=400,
                    detail=f"{filename} exceeds maximum size limit"
                )

            # -----------------------------------
            # Save temporary file
            # -----------------------------------

            suffix = os.path.splitext(filename)[1]

            temp_path = None

            try:

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix
                ) as tmp:

                    tmp.write(content)

                    temp_path = tmp.name

                # -----------------------------------
                # Load and split document
                # -----------------------------------

                chunks = load_and_split(
                    temp_path
                )

                documents.extend(chunks)

            finally:

                # -----------------------------------
                # Cleanup temp file
                # -----------------------------------

                if temp_path and os.path.exists(temp_path):

                    os.remove(temp_path)

        # -----------------------------------
        # Create / Update Vector Store
        # -----------------------------------

        create_vector_store(
            documents=documents,
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_DIR,
            clear_existing=False
        )

        return {
            "message": "Documents uploaded successfully",
            "total_chunks": len(documents)
        }

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Document upload failed: {str(e)}"
        )

# -----------------------------------
# Search Documents
# -----------------------------------


async def search_documents(
    query: str,
    k: int = 5
):

    try:

        results = (
            vector_store
            .similarity_search_with_score(
                query=query,
                k=k
            )
        )

        formatted_results = []

        for doc, score in results:

            formatted_results.append({

                "content":
                    doc.page_content,

                "metadata":
                    doc.metadata,

                "score":
                    round(float(score), 4)
            })

        return {
            "query": query,
            "results_count": len(formatted_results),
            "results": formatted_results
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )
