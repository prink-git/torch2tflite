import torch
from torchvision.models import (
    efficientnet_b0,
    mobilenet_v2,
    resnet18,
    resnet34,
    resnet50,
)


def build_model(
    architecture: str
):

    models = {

        "resnet18": resnet18,

        "resnet34": resnet34,

        "resnet50": resnet50,

        "mobilenet_v2": mobilenet_v2,

        "efficientnet_b0": efficientnet_b0

    }

    if architecture not in models:

        raise ValueError(
            f"Unsupported architecture: {architecture}"
        )

    return models[architecture](weights=None)


def load_checkpoint(model_path, architecture: str):
    model = build_model(architecture)
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError("The .pth file could not be loaded as a PyTorch checkpoint") from exc
    if isinstance(checkpoint, torch.nn.Module):
        loaded = checkpoint
    else:
        if not isinstance(checkpoint, dict):
            raise TypeError("Checkpoint must be a torch.nn.Module or a state_dict dictionary")
        state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint.get("model")))
        if state_dict is None:
            state_dict = checkpoint
        if not isinstance(state_dict, dict) or not all(isinstance(key, str) for key in state_dict):
            raise ValueError("Checkpoint dictionary does not contain a supported state_dict")
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as exc:
            raise ValueError(f"Checkpoint does not match architecture '{architecture}'") from exc
        loaded = model
    loaded.eval().cpu()
    return loaded