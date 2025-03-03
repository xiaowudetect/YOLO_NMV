# YOLO_NMV: Lightweight Object Detection for Non-Motorized Vehicles

This repository contains the implementation of YOLO_NMV, a lightweight object detection model designed for resource-constrained environments, as proposed in our research paper.

## Model Overview

YOLO_NMV is an optimized lightweight object detection model specifically tailored for non-motorized vehicle applications. It introduces key innovations such as:

- **Enhanced Feature Extraction**: Utilizing StarNet_Adown as the backbone to improve representational power while maintaining efficiency.
- **Optimized Multi-Scale Feature Fusion**: Incorporating StarBlock into the C2f module to enhance detection robustness across various object sizes.
- **Novel Convolution Mechanism**: Implementing Linear Deformable T-Convolution (LDTConv) to dynamically adapt to object size and shape variations, reducing feature redundancy and computational overhead.

These improvements allow YOLO_NMV to achieve a superior balance between accuracy and computational efficiency, making it well-suited for deployment in real-world scenarios with limited processing resources.

## Repository Structure

- `modules/` - Contains the core modules used in the implementation, including model components, feature extraction methods, and custom convolution operations.
- `dataset/` - Stores the dataset used for training and evaluation. Ensure that the dataset is placed in this folder before running the training script.
- `result/` - Stores the training results, including model checkpoints, logs, and evaluation metrics.

### Installation

Ensure you have Python 3.8+ and install the required dependencies:

## Citation

If you use this code, please cite our paper:

```
@article{,
  author = {Xiao Wu},
  title = {Intelligent Pedestrian Detection for Non-Motorized Vehicles: The YOLO_NMV Algorithm},
  journal = {Neural Computing and Applications},
  year = {2025},
}
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
