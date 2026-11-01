"""FastAPI 应用入口。"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import BASE_DIR

app = FastAPI(
    title="课程教材 RAG 知识库问答系统",
    description="上传教材/课件，基于检索增强生成回答，并给出原文页码引用。",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

WEB_DIR = BASE_DIR / "web"
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


@app.get("/api", tags=["meta"])
def index():
    return {
        "service": "course-rag",
        "docs": "/docs",
        "health": "/api/health",
        "endpoints": ["/api/documents", "/api/chat", "/api/chat/sync"],
        "web": Path("静态页面已挂载在 /") if WEB_DIR.exists() else "未找到 web 目录",
    }
