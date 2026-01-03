from contextlib import redirect_stdout
import io
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.runner import run

ROOT_DIR = Path(__file__).resolve().parent


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    result: str


app = FastAPI(title="Personal Portfolio Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=ROOT_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(ROOT_DIR / "index.html")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def process_query(payload: QueryRequest):
    buffer = io.StringIO()
    try:
        with redirect_stdout(buffer):
            run(payload.query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Processing failed.") from exc

    result_text = buffer.getvalue().strip()
    if not result_text:
        raise HTTPException(status_code=500, detail="No output returned.")

    return QueryResponse(result=result_text)
