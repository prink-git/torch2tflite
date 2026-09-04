from enum import Enum

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    uploaded = "uploaded"
    exporting_onnx = "exporting_onnx"
    validating_onnx = "validating_onnx"
    converting_tensorflow = "converting_tensorflow"
    quantizing_int8 = "quantizing_int8"
    completed = "completed"
    failed = "failed"


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: int = Field(ge=0, le=100)
    error: str | None = None


class ConversionAccepted(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.uploaded


class HealthResponse(BaseModel):
    status: str