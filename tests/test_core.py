from pathlib import Path

import pytest
import torch
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.api.routes import parse_input_shape
from app.main import app
from app.services.job_service import create_job, get_job, update_job
from app.services.model_service import load_checkpoint
from app.storage.file_manager import safe_filename


@pytest.fixture
def tiny_checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.pth"
    torch.save({"state_dict": {"weight": torch.tensor([[2.0]])}}, path)
    return path


def test_filename_is_confined_to_a_basename():
    assert safe_filename("..\\nested/model.pth") == "model.pth"


def test_shape_validation():
    assert parse_input_shape("1,3,32,32") == (1, 3, 32, 32)
    with pytest.raises(HTTPException):
        parse_input_shape("1,3,32")


def test_job_updates_are_structured():
    create_job("test-job")
    update_job("test-job", "failed: bad checkpoint", 0)
    assert get_job("test-job") == {
        "job_id": "test-job", "status": "failed", "progress": 0, "error": "bad checkpoint"
    }


def test_checkpoint_mismatch_is_explicit(tmp_path: Path):
    path = tmp_path / "wrong.pth"
    torch.save({"state_dict": {"wrong": torch.tensor([1.0])}}, path)
    with pytest.raises(ValueError, match="does not match architecture"):
        load_checkpoint(path, "resnet18")


def test_missing_job_is_not_success(client: TestClient):
    response = client.get("/status/not-a-job")
    assert response.status_code == 404


@pytest.fixture
def client():
    return TestClient(app)
