# Car Image Anonymization & Enhancement using YOLO11 and SAM2

An advanced computer vision pipeline that uses YOLO11 for object detection and SAM2 for fine-grained segmentation to anonymize car images by blurring number plates, tinting windows, and removing backgrounds.

---

## Project Overview

This project is designed to anonymize car images while preserving essential details. It combines the power of **YOLO11** for accurate object detection and **Meta’s SAM2** (Segment Anything Model v2) for precise segmentation. By leveraging both models together, the system performs targeted edits like blurring number plates, tinting windows, and replacing or removing backgrounds — all with clean edges and object-level control.

---

## Key Features

- **Dual-Model Pipeline:** Combines YOLO11 (detection) and SAM2 (segmentation) for highly accurate and controlled image editing.
- **Anonymization Tasks:**
  - Blur car number plates
  - Tint windows for privacy
  - Remove or replace background
- **Custom Dataset:** Built and annotated from scratch to train and fine-tune both models effectively.
- **Refined Outputs:** Clear object boundaries using SAM2, which addresses edge quality limitations of YOLO alone.
- **Automation Ready:** Scalable to process batches of car images with minimal manual intervention.

---

## Technical Details

### Model Design and Training

- **YOLO11:**
  - Used for object detection.
  - Trained on a custom-annotated dataset.
  - Detects relevant objects like car plates, windows, and cars themselves.

- **SAM2:**
  - Fine-tuned to work with bounding boxes from YOLO.
  - Produces pixel-accurate segmentation masks for each detected object.

- **Why Both Models?**
  - YOLO alone lacks edge precision for segmentation.
  - SAM2 alone doesn't know *what* to segment.
  - Combined, YOLO detects objects, and SAM2 refines their boundaries — enabling accurate and actionable masking.

### Processing Pipeline

1. **Detection:** YOLO11 detects objects in the input image.
2. **Segmentation:** Detected bounding boxes are sent to SAM2 to get precise masks.
3. **Anonymization:**
   - **Number plates** are blurred.
   - **Windows** are tinted.
   - **Backgrounds** are removed or replaced with white or custom backgrounds.
4. **Export:** Final anonymized image is saved.

---

## Application Flow

1. **Input:** Provide a car image.
2. **Detection:** YOLO identifies object locations.
3. **Segmentation:** SAM2 generates accurate masks.
4. **Anonymization:** Apply custom processing to each masked region.
5. **Output:** Save or display the anonymized image.

---

## Use Cases

- **Privacy Protection:** Blur number plates in images collected from public roads or parking lots.
- **Data Anonymization:** Preprocess images for machine learning without revealing identities.
- **Surveillance Footage Processing:** Automatically anonymize people or vehicles in video frames.
- **Dashcam and Street View Data:** Ensure compliance with privacy laws before releasing visual datasets.
  
---
