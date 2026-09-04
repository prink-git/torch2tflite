from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch


def validate_onnx(
    onnx_path: Path,
    reference_model=None,
    input_shape: tuple[int, ...] = (1, 3, 224, 224),
):

    model = onnx.load(
        str(onnx_path)
    )

    onnx.checker.check_model(
        model
    )

    result = {"structural_check": True}
    if reference_model is not None:
        sample = np.random.default_rng(0).standard_normal(input_shape).astype(np.float32)
        with torch.inference_mode():
            expected = reference_model(torch.from_numpy(sample)).detach().cpu().numpy()
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        actual = session.run(None, {session.get_inputs()[0].name: sample})[0]
        result.update(max_absolute_error=float(np.max(np.abs(expected - actual))),
                      outputs_match=bool(np.allclose(expected, actual, rtol=1e-3, atol=1e-4)))
        if not result["outputs_match"]:
            raise ValueError(f"ONNX output mismatch (max absolute error {result['max_absolute_error']:.6g})")
    return result