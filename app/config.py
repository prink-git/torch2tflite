from dataclasses import dataclass
from pathlib import Path
import os


def _path(value: str, default: str) -> Path:
    return Path(value or default).expanduser().resolve()


@dataclass(frozen=True)
class Settings:
    upload_dir: Path
    output_dir: Path
    max_upload_mb: int = 512
    log_level: str = "INFO"


def get_settings() -> Settings:
    return Settings(
        upload_dir=_path(os.getenv("UPLOAD_DIR", "uploads"), "uploads"),
        output_dir=_path(os.getenv("OUTPUT_DIR", "outputs"), "outputs"),
        max_upload_mb=max(1, int(os.getenv("MAX_UPLOAD_MB", "512"))),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
    )