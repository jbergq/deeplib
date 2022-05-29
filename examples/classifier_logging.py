# %%
import torch
import torchvision
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torchvision.models.mobilenetv2 import mobilenet_v2

from deeplib.logging import ModelAnalyzer
from deeplib.visualization import visualize_feature_map

tf = torchvision.transforms.ToTensor()

model = mobilenet_v2(pretrained=True).eval()
model_l = ModelAnalyzer(model)

# Get sample image
image = Image.open(Path("data/images/dog_2.jpg"))
image = tf(image).unsqueeze(0)

output = model(image)
output_l = model_l(image)

assert torch.equal(output, output_l), "Outputs do not match."

# %%
activations = model_l.activations().get("features.18")
weights = model_l.weights()  # layer_weights("conv")

# %%

for a in activations:
    plt.figure(a.name, figsize=(20, 20))
    plt.imshow(a.image(), cmap="gray")

plt.show()

assert True
# %%
