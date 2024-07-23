import numpy as np
import torch
import torch.nn as nn
import torchvision
import segment_anything
from segment_anything import SamPredictor, sam_model_registry
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

# integrated model 

class FRS(nn.Module):
    def __init__(self, faster_rcnn_weights, sam_model_checkpoints, sam_model_type, device):
        super().__init__()
        self.device = device
        self.frcnn = fasterrcnn_resnet50_fpn_v2(weights=faster_rcnn_weights).to(device=self.device)
        self.sam = sam_model_registry.get(sam_model_type, None)(checkpoint=sam_model_checkpoints).to(device=self.device)
        self.predictor = SamPredictor(self.sam)

    def forward(self, image, tr_img):
        #img = self.transform_sam(image)
        #image = self.transform_frcnn(image)
        detections = self.frcnn(tr_img)
        self.predictor.set_image(image)
        segments = self.segment(detections=detections)
        box_coords = detections[0]['boxes']
        scores = detections[0]['scores']
        labels = detections[0]['labels']
        if box_coords is not None and segments is not None:
            return box_coords, segments, scores, labels
        return None
    def transform_frcnn(self, image):
         transform =transforms.Compose([transforms.ToTensor()])

         return transform(image)

    def transform_sam(self, image):
        image = cv2.cvtColor( image, cv2.COLOR_BGR2RGB)
        return image
    def segment(self, detections):
        try:
            if detections is not None:
                boundary_boxes = detections[0]['boxes'].to(device=self.device)
                segment_results = []
                for coord in range(len(boundary_boxes)):
                    box = np.array(boundary_boxes[coord].tolist())
                    masks, _, _ = self.predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=box[None, :], # takes box inputs and turn's off other input ways for SAM
                        multimask_output=False
                    )
                    segment_results.append(masks)
                if segment_results:
                    return segment_results
            return None
        except Exception as e:
            print('Error during segmentation: {}'.format(e))
            return None


        
        
