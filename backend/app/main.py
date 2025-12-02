from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .core.bootstrap import ensure_demo_data
from .core.config import DATA_ROOT
from .services import dataset_service, evaluation_service, label_service, model_service, preprocess_service, visualization_service

app = FastAPI(title="SpecSure Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=DATA_ROOT), name="static")


@app.on_event("startup")
async def startup_event():
    ensure_demo_data()


@app.get("/health")
async def health():
    return {"status": "ok"}


app.include_router(dataset_service.router)
app.include_router(preprocess_service.router)
app.include_router(label_service.router)
app.include_router(model_service.router)
app.include_router(evaluation_service.router)
app.include_router(visualization_service.router)
