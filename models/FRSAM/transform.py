import torch
import cv2 
import torchvision.transforms as transforms 

tr = transforms.Compose([transforms.ToTensor()])

def transformer(img, device):
    image = cv2.imread(img)
    transformed_img = tr(image)
    return transformed_img.unsqueeze(0).to(device), image 

