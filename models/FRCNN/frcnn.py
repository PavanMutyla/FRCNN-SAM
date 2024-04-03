import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
import torchinfo

# create fr-cnn

weights = FasterRCNN_ResNet50_FPN_V2_Weights
model = fasterrcnn_resnet50_fpn_v2(weights=weights)

(torchinfo.summary(model, input_size=(32,3,224,224)))