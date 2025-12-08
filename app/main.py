import uvicorn
from fastapi import FastAPI
from app.api.endpoints import router


app = FastAPI(title="Классификатор постов ВК")

app.include_router(router)
