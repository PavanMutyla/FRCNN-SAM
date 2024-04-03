import numpy as np
import torch
import torch.nn as nn
import torchvision
import segment_anything
from segment_anything import SamPredictor, sam_model_registry
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

# integrated model 
class FRSAM(nn.Module):
    def __init__(self, faster_rcnn_weights = FasterRCNN_ResNet50_FPN_V2_Weights, sam_model_chechpoints = None, sam_model_type = None, device = None):
        super().__init__()
        self.device = device
        self.frcnn = fasterrcnn_resnet50_fpn_v2(weights = faster_rcnn_weights).to(device=self.device)
        self.sam = sam_model_registry[sam_model_type](checkpoint=sam_model_chechpoints).to(device=self.device)
        self.predictor = SamPredictor(self.sam)
    
    def forward(self, image):
        detections = self.frcnn(image)
        self.predictor.set_image(image)
        segments = self.segment(detections=detections)
        box_coords = detections[0]['boxes']
        if box_coords and segments:
            return box_coords, segments
        return 0
    # segment the box coords
    def segment(self, detections):
        try:
            if detections != None:
                boundary_boxes = detections[0]['boxes']
                boundary_boxes = boundary_boxes.to(device =self.device)
                segment_results = []
                for coord in range(len(boundary_boxes)):
                    box = np.array(boundary_boxes[coord].tolist())
                    # box segments
                    masks, _, _ = self.predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=box[None, :],
                    multimask_output=False,)
                    segment_results.append(masks)
                if segment_results:
                    return segment_results
                return 0
        except Exception as e:
            print('Implementation Error')
            return None
    

        
        
