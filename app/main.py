import logging

from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(
    title="PyTorch to TFLite Converter",
    description="Convert supported torchvision PyTorch checkpoints to validated ONNX and INT8 TFLite artifacts.",
    version="1.0.0",
)

logging.basicConfig(level=logging.INFO)

app.include_router(router)


@app.get("/")
def home():

    return {
        "message":
        "Converter Running"
    }