from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.routes.rag_routes import router
import os

app = FastAPI(
    title="RAG Content API",
    description="AI-powered Retrieval Augmented Generation API",
    version="1.0.0"
)

app.include_router(router, prefix="/api/v1")

# Ensure frontend directory exists for mounting
os.makedirs("frontend", exist_ok=True)

app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def home():
    return FileResponse("frontend/index.html")