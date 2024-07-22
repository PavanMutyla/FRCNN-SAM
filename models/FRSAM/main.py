from frsam import FRS
import torch 
import torchvision
import cv2
from torchvision.io.image import read_image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
import argparse 
from transform import transformer 
from mask import show_mask

def main():

    parser = argparse.ArgumentParser(description = 'Segment using FRCNN and SAM:')

    parser.add_argument('--frcnn_weights', type = str,  help = ' FasterRCNN_ResNet50_FPN_V2_Weights are default')
    parser.add_argument('sam_model_checkpoint', type = str, help = 'download sam model checkpoints, form https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints')
    parser.add_argument('sam_model_type', type = str, help = 'sam model type of your downloaded checkpoint, example"vit_b"')
    parser.add_argument('image', type = str, help = 'image path')
    parser.add_argument('device', type = str, help = 'use cuda ')

    args = parser.parse_args() 
    print("FRCNN Weights:", args.frcnn_weights)
    print("SAM Model Checkpoint:", args.sam_model_checkpoint)
    print("SAM Model Type:", args.sam_model_type)
    print("Image Path:", args.image)
    print("Device:", args.device)

    model = FRS(faster_rcnn_weights = args.frcnn_weights, sam_model_checkpoints = args.sam_model_checkpoint, sam_model_type = args.sam_model_type, device =args.device )

    image = args.image

    transformed_img, img = transformer(image, device = args.device)

    model.eval() 
    boxes, masks, scores, labels = model(img, transformed_img)

    list_masks = []
    for box, mask, score, label in zip(boxes,masks, scores, labels ):
        if score > 0.9 and label==1: # here m is 1 as I am only checking with human class
            list.append(mask)

    show_mask(list_masks, img)


    
if __name__== '__main__':
    main()

