import uuid
from threading import Lock

jobs: dict[str, dict] = {}
_lock = Lock()


def generate_job_id() -> str:
    return str(uuid.uuid4())


def create_job(job_id: str):
    with _lock:
        jobs[job_id] = {"job_id": job_id, "status": "uploaded", "progress": 0, "error": None}


def update_job(
    job_id: str,
    status: str,
    progress: int
):

    with _lock:
        if job_id in jobs:
            failed = status.startswith("failed:")
            jobs[job_id].update(status="failed" if failed else status, progress=progress,
                                error=status[7:].lstrip() if failed else None)


def get_job(job_id: str):

    with _lock:
        return jobs.get(job_id, {}).copy() or None