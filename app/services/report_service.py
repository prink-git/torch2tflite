import json
import os
from pathlib import Path


def generate_report(
    report_path: Path,
    original_model: Path,
    onnx_model: Path,
    tflite_model: Path,
    validation: dict | None = None,
):

    report = {

        "original_size_mb":

        round(
            os.path.getsize(
                original_model
            ) / (1024 * 1024),
            2
        ),

        "onnx_size_mb":

        round(
            os.path.getsize(
                onnx_model
            ) / (1024 * 1024),
            2
        ),

        "tflite_size_mb":

        round(
            os.path.getsize(
                tflite_model
            ) / (1024 * 1024),
            2
        ),

        "status": "success",
        "validation": validation or {},
    }

    with open(
        report_path,
        "w"
    ) as f:

        json.dump(
            report,
            f,
            indent=4
        )