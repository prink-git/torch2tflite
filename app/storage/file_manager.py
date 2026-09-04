import re
from pathlib import Path

from fastapi import UploadFile

from app.config import Settings, get_settings

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")


def safe_filename(filename: str | None) -> str:
    name = Path(filename or "model.pth").name
    name = _SAFE_NAME.sub("_", name).strip(".")
    if not name or name.lower() in {".", ".."}:
        raise ValueError("A valid model filename is required")
    return name


def ensure_directories(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.output_dir.mkdir(parents=True, exist_ok=True)


def save_uploaded_file(file: UploadFile, destination: Path, max_bytes: int) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    try:
        with destination.open("wb") as buffer:
            while chunk := file.file.read(1024 * 1024):
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(f"Uploaded model exceeds {max_bytes // (1024 * 1024)} MB")
                buffer.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        file.file.close()
    return total