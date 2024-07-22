import torch
import cv2 
import torchvision.transforms as transforms 

tr = transforms.Compose([transforms.ToTensor()])

def transformer(img, device):
    '''
    transforms the image to tensor
    '''
    image = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
    transformed_img = tr(image)
    return transformed_img.unsqueeze(0).to(device), image 

