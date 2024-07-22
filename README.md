# Instance Segmentation with FRCNN and SAM

This project leverages the idea of using both Faster R-CNN (FRCNN) and the Segment Anything Model (SAM) from META for advanced instance segmentation. It aims to combine the object detection capabilities of FRCNN with the versatile segmentation capabilities of SAM to provide a comprehensive instance segmentation solution.

## Features

- **Faster R-CNN Integration**: Utilizes the Faster R-CNN model for accurate object detection.
- **Segment Anything Model (SAM)**: Employs SAM for fine-grained instance segmentation.
- **Overlay Results**: Combines detected objects with segmented masks and overlays them on the original image.

## Installation

To set up the project, follow these instructions:

1. **Clone the Repository**:

    ```bash
    git clone git@github.com:PavanMutyla/FRCNN-SAM.git
    ```

2. **Navigate to the Project Directory**:

    ```bash
    cd FRCNN-SAM
    ```

3. **Install Dependencies**:

    Make sure you have Python and pip installed, then run:

    ```bash
    pip install -r requirements.txt
    ```

    

## Usage

To run the instance segmentation, use the following command:

```bash
python amin.py --frcnn_weights path/to/frcnn_weights.pth sam_model_checkpoint path/to/sam_checkpoint.pth --sam_model_type vit_b image path/to/image.jpg device cuda
```

The confidence_threshold is set to 0.9 and the targeted class is 'Human', you can change it in  - [main.py](https://github.com/PavanMutyla/FRCNN-SAM/blob/main/models/FRSAM/main.py)






## Reference

- [SAM META](https://github.com/facebookresearch/segment-anything)
- [SAM model checkpoints](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints)
- [YOLOv8 with SAM](https://blog.roboflow.com/how-to-use-yolov8-with-sam/)


## Results

<p align="center">
  <img src="https://github.com/PavanMutyla/FRCNN-SAM/blob/main/Images/results/FRSAM.jpeg" alt="FRCNN-SAM" width="45%" style="display:inline-block; vertical-align:middle;">
  <img src="https://github.com/PavanMutyla/FRCNN-SAM/blob/main/Images/results/YOLOv8.jpeg" alt="YOLOv8-seg" width="45%" style="display:inline-block; vertical-align:middle;">
</p>
Figure 1: Instance segmentation results using FRCNN and SAM (left) and YOLOv8 segmentation (right)
