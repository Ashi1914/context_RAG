from fastapi import APIRouter, Depends, UploadFile, File
from typing import List
from app.schemas.schemas import QueryRequest, QueryResponse
from fastapi import HTTPException

from app.dependencies.dependencies import (
    get_llm,  
    get_vector_store
)

from app.services.generator import RAGGenerator
from app.services.rag_service import upload_documents

router = APIRouter(
     prefix="/rag",
    tags=["RAG"]
)


@router.post(
    "/ask",
    response_model=QueryResponse
)

async def ask_question(
    request: QueryRequest,

    vector_store=Depends(
        get_vector_store
    ),

    llm=Depends(
        get_llm
    )
):

    try:

        generator = RAGGenerator(
            vector_store=vector_store,
            llm=llm
        )

        response = await generator.generate(
            request.question
        )

        return QueryResponse(**response)

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    try:
        return await upload_documents(files)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))