import torch
import torchvision
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torchvision.models.resnet import resnet18

from deeplib.logging import ModelAnalyzer

model = resnet18(pretrained=True).eval()
model_l = ModelAnalyzer(model)

# Get sample image
tf = torchvision.transforms.ToTensor()

image = Image.open(Path("data/images/dog.jpg"))
image = tf(image).unsqueeze(0)

output = model(image)
output_l = model_l(image)
assert torch.equal(output, output_l), "Outputs do not match."

activations = model_l.activations().get("relu")
weights = model_l.weights()

for a in activations:
    plt.figure(a.name, figsize=(20, 20))
    plt.imshow(a.image(), cmap="gray")

plt.show()

assert True
