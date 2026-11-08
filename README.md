# BTP-Diffusion-Models
# BTP-Diffusion-Models

### Diffusion-Based Object Detection for Aerial and Remote Sensing Images

This repository contains the implementation, experimentation, evaluation, and analysis of diffusion-based object detection methods for aerial and remote sensing imagery. The work was carried out as part of a Bachelor of Technology Project (BTP), focusing on horizontal and oriented object detection using diffusion-based detection architectures.

The project investigates the application of diffusion models to object detection, including model fine-tuning, inference, evaluation across multiple datasets, and analysis of detection performance.

---

## Table of Contents

- [Overview](#overview)
- [Methods and Architectures](#methods-and-architectures)
- [Datasets](#datasets)
- [Experimental Results](#experimental-results)
- [Experimental Details](#experimental-details)
- [Evaluation Notes](#evaluation-notes)
- [Environment and Dependencies](#environment-and-dependencies)
- [Repository Contents](#repository-contents)
- [Acknowledgements](#acknowledgements)
- [References](#references)

---

## Overview

Object detection in aerial imagery presents several challenges, including large variations in object scale, dense object distributions, arbitrary object orientations, and complex backgrounds.

Conventional object detectors generally rely on direct bounding-box regression, whereas diffusion-based detectors formulate object detection as an iterative denoising process. Starting from noisy object proposals, the detector progressively refines bounding-box predictions and classification scores to obtain the final detections.

This project explores diffusion-based approaches for both:

- **Horizontal Bounding-Box (HBB) Detection:** Detection using axis-aligned bounding boxes.
- **Oriented Bounding-Box (OBB) Detection:** Detection using rotated bounding boxes to represent objects with arbitrary orientations.

The experiments cover multiple aerial and remote sensing datasets and investigate the performance of different diffusion-based detection architectures.

## Methods and Architectures

The project primarily explores the following methods:

### 1. DiffusionDet

DiffusionDet formulates object detection as a generative denoising process. It begins with noisy bounding-box proposals and progressively refines them through a learned detection decoder.

Experiments were conducted using a ResNet-50 backbone with a Feature Pyramid Network (FPN).

The model was fine-tuned and evaluated on:

- DIOR
- NWPU VHR-10
- HRSC2016

### 2. ReDiffDet

ReDiffDet was investigated for oriented object detection in aerial imagery. The experiments involved inference and evaluation on datasets containing rotated object annotations.

The evaluated datasets included:

- HRSC2016
- DOTA-v1.0

### 3. GSDet

GSDet was used for inference on the DIOR-R dataset. The evaluation involved a ResNet-50/FPN-based configuration with 900 proposals, two refinement stages, and rotated non-maximum suppression.

No additional local fine-tuning was performed for this experiment.

---

## Datasets

The experiments cover the following datasets:

| Dataset | Detection Task | Description |
|---|---|---|
| DIOR | HBB | Large-scale optical remote sensing object detection |
| NWPU VHR-10 | HBB | Very-high-resolution aerial image object detection |
| HRSC2016 | HBB and OBB | Ship detection in aerial and maritime imagery |
| DIOR-R | OBB | Rotated version of the DIOR dataset |
| DOTA-v1.0 | OBB | Large-scale dataset for oriented object detection in aerial images |

The datasets were used according to their respective annotation formats and evaluation procedures.

**Note:** The datasets are not redistributed in this repository. Users should obtain them from their respective official sources and prepare the annotations and directory structures required by the corresponding configurations.

---

## Experimental Results

The following table summarizes the principal evaluation results obtained during the project.

All AP values are expressed as percentages.

| Dataset | Task | Method | AP / mAP | AP50 | AP75 |
|---|---|---|---:|---:|---:|
| DIOR | HBB | DiffusionDet | 45.4766 | 67.5089 | 48.8293 |
| NWPU VHR-10 | HBB | DiffusionDet | 61.76 | 94.68 | 69.21 |
| HRSC2016 | HBB | DiffusionDet | 84.4688 | 94.4234 | 90.7618 |
| DIOR-R | OBB | GSDet | 46.23 | 69.70 | 48.10 |
| HRSC2016 | OBB | ReDiffDet | 72.32 | 90.405 | 89.506 |
| DOTA-v1.0 | OBB | ReDiffDet | 47.6231 | 73.9373 | 52.3050 |

### Result Interpretation

- **DIOR:** DiffusionDet was fine-tuned and evaluated on the local DIOR test split.
- **NWPU VHR-10:** Multiple fine-tuning stages were investigated. The best recorded mAP was 61.76%, obtained at epoch 16.
- **HRSC2016 (HBB):** DiffusionDet achieved an AP of 84.4688%. The local evaluation used 453 test images, whereas the published test split contains 444 images.
- **DIOR-R:** The GSDet result was obtained through inference without additional local fine-tuning.
- **HRSC2016 (OBB):** The reported metrics were obtained from an oriented detection evaluation using the recorded ReDiffDet experimental setup.
- **DOTA-v1.0:** Predictions were submitted to the official DOTA evaluation server because the test-set ground-truth annotations were not locally available.

These results summarize the experiments conducted in this project. They should not automatically be interpreted as exact reproductions of published benchmark results, since experimental configurations and evaluation conditions may differ.

---

## Experimental Details

### Horizontal Object Detection

DiffusionDet was fine-tuned and evaluated on DIOR, NWPU VHR-10, and HRSC2016.

The experiments involved:

- ResNet-50 backbone with FPN.
- Fine-tuning from pretrained checkpoints.
- Evaluation at selected training checkpoints.
- COCO-style Average Precision metrics.
- Analysis of performance across object scales and individual categories.

For NWPU VHR-10, multiple sequential fine-tuning runs were conducted to investigate changes in detection performance across training stages.

### Oriented Object Detection

Oriented detection experiments involved ReDiffDet and GSDet.

The experiments covered:

- Inference on DIOR-R using GSDet.
- Oriented detection evaluation on HRSC2016 using ReDiffDet.
- Prediction generation and official evaluation of DOTA-v1.0 results.

The oriented detection experiments were evaluated using the corresponding rotated bounding-box evaluation procedures.

---

## Evaluation Notes

A few dataset and experimental details are important when interpreting the results:

1. **HRSC2016 HBB:** The local evaluation contained 453 test images, compared with the published split count of 444 images. This difference should be considered when comparing the local result with published results.

2. **HRSC2016 OBB:** The experimental records include checkpoint/state-dictionary compatibility messages and a `sub_data_root` configuration that should be considered when reproducing the evaluation.

3. **DIOR-R:** The GSDet result represents inference using an existing checkpoint, not a locally fine-tuned model.

4. **DOTA-v1.0:** The reported metrics were obtained through the official evaluation server, which evaluates submitted predictions using the withheld test annotations.

5. **Literature comparisons:** Direct numerical comparisons should be interpreted carefully because hardware, backbone architectures, input resolutions, training schedules, augmentations, dataset splits, and evaluation protocols may differ.

---

## Environment and Dependencies

The experiments were conducted using Python, PyTorch, CUDA, and GPU acceleration.

One of the recorded experimental environments was:

| Component | Configuration |
|---|---|
| Operating System | Ubuntu (WSL2) |
| Python | 3.10.22 |
| PyTorch | 2.4.0+cu121 |
| CUDA | 12.1 |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| Detection Framework | Detectron2-based DiffusionDet implementation |

Additional dependencies include:

- NumPy
- OpenCV
- PyTorch
- Torchvision
- Detectron2
- PyYAML
- pycocotools

The exact dependencies and compatible versions may vary between the individual detection architectures.

### Setup

Clone the repository:

```bash
git clone <YOUR_REPOSITORY_URL>
cd BTP-Diffusion-Models
```

Create and activate a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the dependencies specified for the relevant implementation:

```bash
pip install -r requirements.txt
```

**Important:** The installation commands above assume that a `requirements.txt` file is present in the repository. Refer to the individual model implementation and configuration files for architecture-specific installation requirements.

---

## Repository Contents

The repository is intended to organize the different model implementations, dataset configurations, training and evaluation scripts, experimental logs, and result summaries associated with the project.

The exact directory organization may differ between the individual implementations. Refer to the corresponding configuration files and scripts before running an experiment.

Large datasets, pretrained checkpoints, and generated prediction files may need to be obtained or generated separately.

---

## Acknowledgements

This project builds upon research in diffusion-based object detection and oriented object detection.

We acknowledge the authors and contributors of the underlying detection architectures, datasets, and open-source frameworks used throughout the experiments.

In particular, this work builds on the research and implementations associated with DiffusionDet, ReDiffDet, and GSDet.

---

## References

1. **DiffusionDet:** Diffusion Model for Object Detection.  
   [Research paper](https://arxiv.org/abs/2211.09788)

2. **DOTA:** Dataset for Object Detection in Aerial Images.  
   [Official DOTA website](https://captain-whu.github.io/DOTA/)

3. **DIOR:** Object Detection in Optical Remote Sensing Images.  
   [DIOR dataset](https://gcheng-nwpu.github.io/)

4. **NWPU VHR-10:** Ten-Class Object Detection Dataset for Remote Sensing Images.

5. **HRSC2016:** High Resolution Ship Collections 2016.

For the exact ReDiffDet and GSDet publication details, refer to the corresponding original papers and implementation repositories.

---

**Project:** BTP-Diffusion-Models  
**Focus:** Diffusion-Based Object Detection  
**Tasks:** Horizontal and Oriented Object Detection  
**Application Domain:** Aerial and Remote Sensing Imagery
