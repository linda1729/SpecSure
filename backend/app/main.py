from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .core.config import CNN_ROOT, ensure_cnn_directories
from .services import cnn_service

app = FastAPI(title="SpecSure Backend", version="0.2.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 暴露模型目录，便于前端直接访问训练/可视化产物
ensure_cnn_directories()
app.mount("/cnn-static", StaticFiles(directory=CNN_ROOT), name="cnn-static")


@app.get("/health")
async def health():
    return {"status": "ok"}


# 双重挂载，确保 /api/cnn/* 与无前缀路径均可访问，避免部署前缀不一致
app.include_router(cnn_service.router)
app.include_router(cnn_service.router, prefix="/api/cnn")
