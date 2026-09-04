from pathlib import Path

from app.services.model_service import load_checkpoint


def load_pytorch_model(
    model_path: Path,
    architecture: str
):

    return load_checkpoint(model_path, architecture)