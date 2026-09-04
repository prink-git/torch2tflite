from pathlib import Path

import torch

from app.services.pytorch_service import load_pytorch_model


def export_to_onnx(
    model_path: Path,
    architecture: str,
    output_path: Path,
    input_shape: tuple[int, ...] = (1, 3, 224, 224),
):

    model = load_pytorch_model(
        model_path,
        architecture
    )

    torch.manual_seed(0)
    dummy_input = torch.randn(*input_shape, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,

        opset_version=18,

        input_names=["input"],

        output_names=["output"],

        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    return output_path