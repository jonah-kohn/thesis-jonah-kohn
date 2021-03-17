import torch
import lucent
from torchvision.models import vgg16
from lucent.optvis import render, param, transform

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = vgg16(pretrained=True)

lucent.lucent.modelzoo.util.get_model_layers(model)

model.to(device).eval()

obj = "layer2:9" # a ResNet50 layer and channel
render.render_vis(model, obj)
