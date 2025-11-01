from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import uvicorn
from ingest import main as run_ingest
from model_loader import get_embedding_model
from rag_app import query_rag, search_external_sources, generate_answer_from_external
import shutil

app = FastAPI(title="AI Textbook Assistant API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model
embed_model = get_embedding_model()

# Session state simulation (in production, use database)
sessions = {}

class QueryRequest(BaseModel):
    query: str
    session_id: str = "default"
    top_k: int = 10
    similarity_threshold: float = 0.3

class ClearRequest(BaseModel):
    session_id: str = "default"

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    # Save file
    data_dir = "data"
    pdf_path = os.path.join(data_dir, "textbook.pdf")
    os.makedirs(data_dir, exist_ok=True)

    with open(pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Ingest
    try:
        run_ingest(embed_model)
        return {"message": "PDF ingested successfully", "book_title": file.filename.replace('.pdf', '')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.post("/query")
async def query(request: QueryRequest):
    session_id = request.session_id
    if session_id not in sessions:
        sessions[session_id] = {"messages": [], "book_title": "My Textbook"}

    try:
        result = query_rag(
            request.query,
            sessions[session_id]["book_title"],
            request.top_k,
            request.similarity_threshold,
            embed_model,
            stream=False
        )

        if len(result) == 4:
            answer, sources, retrieved_chunks, external_search = result
        else:
            answer, sources, retrieved_chunks = result
            external_search = False

        # Add to session
        sessions[session_id]["messages"].append({
            "role": "user",
            "content": request.query
        })
        sessions[session_id]["messages"].append({
            "role": "assistant",
            "answer": answer,
            "sources": sources
        })

        return {
            "answer": answer,
            "sources": sources,
            "external_search": external_search,
            "retrieved_chunks": retrieved_chunks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/external_search")
async def external_search(request: QueryRequest):
    try:
        external_results = search_external_sources(request.query)
        if external_results:
            external_answer = generate_answer_from_external(request.query, external_results)
            return {
                "answer": external_answer,
                "results": external_results[:5]
            }
        else:
            raise HTTPException(status_code=404, detail="No external results found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"External search failed: {str(e)}")

@app.post("/clear")
async def clear_conversation(request: ClearRequest):
    session_id = request.session_id
    if session_id in sessions:
        sessions[session_id]["messages"] = []
    return {"message": "Conversation cleared"}

@app.get("/status")
async def get_status(session_id: str = "default"):
    ingested = os.path.exists("faiss_index/textbook.index")
    book_title = sessions.get(session_id, {}).get("book_title", "My Textbook")
    return {
        "ingested": ingested,
        "book_title": book_title,
        "messages": sessions.get(session_id, {}).get("messages", [])
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
