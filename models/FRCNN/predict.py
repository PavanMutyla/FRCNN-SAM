import torch
from frcnn import model
import torchvision.transforms as T
import cv2 
from torchvision.utils import draw_bounding_boxes



def draw(image, preds):
    boxes = preds[0]['boxes'].tolist()
    labels = preds[0]['labels'].tolist()
    img = image.copy()
    for box, label in zip(boxes, labels):
        # Convert box coordinates to integers
        box = [int(coord) for coord in box]
        # Draw bounding box rectangle
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        # Draw label text
        cv2.putText(image, f"Class {label}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return img

    




def predict(image_path:str, model = model, weights = None, n = None, device = 'cpu'):
    '''
    n: classes without background
    default device cpu
    '''
    if n is not None:
        model.roi_heads.box_predictor.cls_score = torch.nn.Linear(in_features=1024, out_features=n+1)
    if weights is not None:
        model.load_state_dict(torch.load(weights))
    model.eval()
    transforms = T.Compose([T.Resize((800,800)), T.ToTensor()])

    image = cv2.imread(image_path)
    image_data = transforms(image).unsqueeze(0).to(device) 

    predictions = model(image_data)

    pred_image = draw(image, predictions)

    return pred_image  # need to save it in a location , need to write that part 
    

### need to edit the loading of weights 

### also need to change this file for both pred and test (acc, precision ect.)