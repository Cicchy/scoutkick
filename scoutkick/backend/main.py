from fastapi import FastAPI
from scoutkick.backend.src.api.router import api_router

app = FastAPI(title="scoutkick EPA API", version="0.1.0")
app.include_router(api_router)


@app.get("/")
def root():
    return {"name": "scoutkick", "version": "0.1.0", "docs": "/docs"}

