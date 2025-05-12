# Swimming Style Analysis Using SAM and MediaPipe

This project implements an automated underwater swimming style analysis system using image segmentation and pose estimation techniques. It leverages Meta’s Segment Anything Model (SAM) and MediaPipe to segment swimmers and classify four main strokes: freestyle, backstroke, breaststroke, and butterfly.

## Overview

The system consists of the following components:

1. **Video Frame Extraction Module**  
2. **Swimmer Segmentation Module (SAM)**  
3. **Pose Estimation Module (MediaPipe)**  
4. **Swimming Style Classification Module**

## Features

- High-quality video frame extraction
- Manual bounding box input for swimmer segmentation
- Swimmer segmentation using Segment Anything Model (SAM)
- Pose estimation using MediaPipe
- Swimming stroke classification (freestyle, backstroke, breaststroke, butterfly)
- Visualization of segmentation masks and pose landmarks
- Supports video formats: `.mp4`, `.avi`, `.mov`, `.mkv`

## Prerequisites

- Python 3.8 or higher  
- OpenCV  
- PyTorch  
- `segment-anything` (Meta AI's SAM)  
- MediaPipe  
- NumPy  
- Matplotlib  
- tqdm  

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/swimming-style-sam-mediapipe.git
cd swimming-style-sam-mediapipe
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Download SAM weights:

```bash
# Example: Download SAM weights (ViT-H)
wget https://github.com/facebookresearch/segment-anything/releases/download/v1/sam_vit_h.pth
```

## Usage

### 1. Frame Extraction

```python
from frame_extractor import FrameExtractor

extractor = FrameExtractor()
extractor.extract_frames(
    video_path='videos/swim_session.mp4',
    output_dir='frames/',
    sample_rate=2
)
```

### 2. Swimmer Segmentation (SAM)

```python
from sam_segmentation import SAMSwimmerSegmentation

segmenter = SAMSwimmerSegmentation(model_path='sam_vit_h.pth')
segmenter.segment_with_box(
    image_path='frames/frame001.jpg',
    box_coords=(637, 292, 338, 113),
    output_path='outputs/segmented_frame001.png'
)
```

### 3. Pose Estimation

```python
from pose_estimation import MediaPipePoseEstimator

pose_estimator = MediaPipePoseEstimator()
pose_estimator.estimate_pose(
    image_path='outputs/segmented_frame001.png',
    output_path='outputs/pose_overlay_frame001.png'
)
```

### 4. Swimming Style Classification

```python
from stroke_classification import StrokeClassifier

classifier = StrokeClassifier()
classifier.classify_stroke(
    pose_folder='outputs/poses/',
    output_file='results/classification.json'
)
```

## Project Structure

```
swimming-style-sam-mediapipe/
├── frame_extractor.py          # Frame extraction logic
├── sam_segmentation.py         # Swimmer segmentation using SAM
├── pose_estimation.py          # Pose detection using MediaPipe
├── stroke_classification.py    # Swimming style classification
├── outputs/                    # Processed results and visualizations
└── README.md                   # Project documentation
```

## Configuration

### Frame Extraction
- `sample_rate`: Frame sampling interval (default: 2)

### SAM Segmentation
- `box_coords`: Manually defined bounding box (x, y, width, height)

### Pose Estimation
- Uses MediaPipe’s full-body pose model
- Outputs landmark coordinates and visibility scores

### Stroke Classification
- Detects swimming style using landmark patterns and frame sequences
- Supports confidence scoring and optional smoothing

## Output

- Extracted frames from video
- Segmented swimmer images
- Pose overlay images
- JSON-style classification results

## Known Limitations

- Bounding boxes must be manually specified
- Detection may fail when swimmer is fully submerged or occluded
- Real-time analysis is not currently supported
- Lighting and water clarity affect pose estimation accuracy
- GPU recommended for best performance with SAM

---

Feel free to fork or contribute!
