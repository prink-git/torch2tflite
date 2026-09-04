from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.responses import FileResponse

from app.config import get_settings
from app.models.schemas import ConversionAccepted, JobResponse
from app.services.conversion_service import run_conversion
from app.services.job_service import (
    create_job,
    generate_job_id,
    get_job,
    update_job,
)
from app.services.model_service import build_model
from app.storage.file_manager import (
    ensure_directories,
    safe_filename,
    save_uploaded_file,
)

router = APIRouter()


def parse_input_shape(value: str) -> tuple[int, ...]:
    try:
        shape = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise HTTPException(422, "input_shape must be comma-separated positive integers") from exc
    if len(shape) != 4 or any(dimension <= 0 for dimension in shape):
        raise HTTPException(422, "input_shape must be N,C,H,W with positive dimensions")
    return shape


@router.post("/convert", response_model=ConversionAccepted, status_code=202,
             summary="Start a PyTorch checkpoint conversion")
async def convert(
    background_tasks: BackgroundTasks,
    model_file: UploadFile = File(..., description="A .pth or .pt PyTorch module/checkpoint"),  # noqa: B008
    architecture: str = Query(..., description="Supported torchvision architecture"),
    input_shape: str = Query("1,3,224,224", description="N,C,H,W input shape"),
):
    settings = get_settings()
    ensure_directories(settings)
    if not model_file.filename or Path(model_file.filename).suffix.lower() not in {".pth", ".pt"}:
        raise HTTPException(415, "Only .pth and .pt uploads are supported")
    try:
        build_model(architecture)
        shape = parse_input_shape(input_shape)
        filename = safe_filename(model_file.filename)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    job_id = generate_job_id()
    create_job(job_id)
    job_folder = settings.output_dir / job_id
    model_path = job_folder / filename
    try:
        save_uploaded_file(model_file, model_path, settings.max_upload_mb * 1024 * 1024)
    except ValueError as exc:
        update_job(job_id, f"failed: {exc}", 0)
        raise HTTPException(413, str(exc)) from exc

    background_tasks.add_task(
        run_conversion,
        job_id,
        job_folder,
        model_path,
        architecture,
        shape,
    )

    return {
        "job_id": job_id,
        "status": "started"
    }


@router.get("/status/{job_id}", response_model=JobResponse)
def status(job_id: str):

    job = get_job(job_id)

    if not job:

        raise HTTPException(404, "Job not found")

    return job


@router.get("/download/{job_id}")
def download(job_id: str):
    if not get_job(job_id):
        raise HTTPException(404, "Job not found")
    job_folder = get_settings().output_dir / job_id

    tflite_file = (
        job_folder /
        "model_int8.tflite"
    )

    if not tflite_file.exists():

        raise HTTPException(409, "Model is not ready")

    return FileResponse(
        path=tflite_file,
        filename="model_int8.tflite",
        media_type="application/octet-stream"
    )


@router.get("/report/{job_id}")
def report(job_id: str):
    if not get_job(job_id):
        raise HTTPException(404, "Job not found")
    report_file = get_settings().output_dir / job_id / "report.json"

    if not report_file.exists():

        raise HTTPException(409, "Report is not ready")

    return FileResponse(
        path=report_file,
        filename="report.json",
        media_type="application/json"
    )