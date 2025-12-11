# Enhanced YOLOv8 Object Detection and U-Net Segmentation

##Developers
[Waseem Akhtar](https://github.com/waseem087)

[Naveed Ahmed](https://github.com/NaveedAhmed5)

## Project Overview
This project implements and compares two powerful deep learning architectures for waste detection and segmentation using the TACO (Trash Annotations in Context) dataset. The goal is to accurately identify and segment litter in images to aid in automated waste management systems.

The project consists of two main phases:
1.  **Object Detection**: Using **YOLOv8** (You Only Look Once) to detect and classify waste items with bounding boxes.
2.  **Semantic Segmentation**: Using **U-Net** with a ResNet34 backbone to generate pixel-perfect masks for waste items.

## Dataset
The project uses a subset of the **TACO Dataset**, an open image dataset of waste in the wild.
-   **Classes**: The dataset contains various classes of litter (e.g., Aluminium foil, Bottle, Carton, etc.).
-   **Preprocessing**:
    -   Images are resized to 640x640 (for YOLO) and 256x256 (for U-Net).
    -   Data augmentation techniques (Albumentations) such as HorizontalFlip, RandomBrightnessContrast, and ShiftScaleRotate are applied to improve model generalization.
    -   The dataset is split into **Train (60%)**, **Validation (20%)**, and **Test (20%)** sets.

## Methodology

### 1. YOLOv8 (Object Detection)
-   **Model**: `yolov8s.pt` (Small version).
-   **Goal**: Real-time object detection.
-   **Training Configuration**:
    -   **Epochs**: 50
    -   **Batch Size**: 4
    -   **Optimizer**: AdamW
    -   **Image Size**: 640x640
-   **Output**: Bounding boxes with class labels and confidence scores.

### 2. U-Net (Semantic Segmentation)
-   **Architecture**: U-Net with a pre-trained **ResNet34** encoder.
-   **Goal**: Precise pixel-level segmentation.
-   **Training Configuration**:
    -   **Epochs**: 3 (Optimized for demonstration/low RAM)
    -   **Batch Size**: 4
    -   **Loss Function**: Combined Cross-Entropy + Dice Loss
    -   **Optimizer**: Adam
-   **Output**: Binary masks indicating the presence of waste pixels.

## Installation & Usage

### Prerequisites
Ensure you have Python installed along with the following libraries:
```bash
pip install ultralytics torch torchvision opencv-python matplotlib pandas numpy albumentations segmentation-models-pytorch
```

### Running the Project
1.  Clone this repository.
2.  Open the Jupyter Notebook: `Enhanced YOLOv8 Object Detection and U-Net.ipynb`.
3.  Run the cells sequentially. The notebook handles:
    -   Dataset downloading and preparation.
    -   Data augmentation and visualization.
    -   YOLOv8 training and evaluation.
    -   U-Net training and evaluation.

## Results

### YOLOv8 Performance
-   **Precision**: 0.3162
-   **Recall**: 0.2500
-   **mAP@50**: 0.2923
-   **mAP@50-95**: 0.1127
-   *Note: Results may vary based on the specific subset and random seed.*

### U-Net Performance
-   **Best IoU (Intersection over Union)**: 0.7609
-   **Validation Loss**: 1.2060
-   The U-Net model successfully learned to segment waste items, achieving a high IoU score even with limited training epochs.

## Conclusion
-   **YOLOv8** is highly effective for real-time applications where speed and approximate localization are key. It excels at counting and classifying objects.
-   **U-Net** provides superior precision for determining the exact shape and area of the waste, which is crucial for robotic grasping or detailed analysis.
-   **Hybrid Approach**: A combination of both models (using YOLO for localization and U-Net for fine-grained segmentation) would likely yield the best results for a comprehensive waste management system.

## Acknowledgments
-   **TACO Dataset**: [http://tacodataset.org/](http://tacodataset.org/)
-   **Ultralytics YOLO**: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
-   **Segmentation Models PyTorch**: [https://github.com/qubvel/segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)
