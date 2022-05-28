import torch
from torchvision.models.resnet import resnet18

from deeplib.logging import LogWrapper

model = resnet18(pretrained=True)

model_l = LogWrapper(model)

input = torch.randn(1, 3, 224, 224)

output = model(input)
output_l = model_l(input)

assert torch.equal(output, output_l), "Outputs do not match."

log = model_l.get_log()

assert True