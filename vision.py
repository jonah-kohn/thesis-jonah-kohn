import torch
from torchvision.models import resnet50
from lucent.optvis import render, param, transform

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = resnet50(pretrained=True)
model.to(device).eval()

obj = "layer2:9" # a ResNet50 layer and channel
render.render_vis(model, obj)
