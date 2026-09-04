import torch
from torchvision.models import resnet18

model = resnet18()

torch.save(
    model.state_dict(),
    "resnet18_test.pth"
)

print("Saved resnet18_test.pth")