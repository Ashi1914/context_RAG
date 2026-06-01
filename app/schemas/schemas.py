from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class QueryRequest(BaseModel):

    question: str = Field(
        min_length=1,
        max_length=4000
    )


class SourceModel(BaseModel):

    content: str

    metadata: Dict[str, Any]


class QueryResponse(BaseModel):

    answer: str

    sources: Optional[List[SourceModel]] = []