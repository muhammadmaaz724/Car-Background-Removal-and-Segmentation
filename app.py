import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import os
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO, SAM
from build_sam import build_sam2
from sam2_image_predictor import SAM2ImagePredictor

def refine_mask_edges(mask):
    """Enhance small edges and fill gaps using adaptive methods and fine-tuned morphology."""
    mask = (mask * 255).astype(np.uint8)

    # Step 1: Denoise slightly to prevent over-detection
    mask = cv2.GaussianBlur(mask, (1, 1), 0)

    # Step 2: Fill small holes using morphology
    kernel_erode = np.ones((3, 3), np.uint8)
    kernel_close = np.ones((9, 9), np.uint8)
    mask = cv2.erode(mask, kernel_erode, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    # Step 3: Adaptive Canny thresholding
    median_val = np.median(mask)
    lower = int(max(0, 0.33 * median_val))
    upper = int(min(255, 1.0 * median_val))
    edges = cv2.Canny(mask, lower, upper)

    # Step 4: Dilate edges slightly to ensure continuity
    edge_dilated = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)

    # Step 5: Merge refined edges with original mask
    combined_mask = cv2.bitwise_or(mask, edge_dilated)

    # Step 6: Optional contour cleanup of tiny specs (optional, can be toggled)
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned_mask = np.zeros_like(combined_mask)
    for cnt in contours:
        if cv2.contourArea(cnt) > 20:  # Filter out very small regions
            cv2.drawContours(cleaned_mask, [cnt], -1, 255, -1)

    # Step 7: Final smoothing to reduce jaggedness
    final = cv2.GaussianBlur(cleaned_mask, (1, 1), 0)

    return final.astype(np.float32) / 255.0

def filter_region_mask(mask, min_area=500, smoothing_kernel_size=5):
    """
    Filter region masks to remove small artifacts and smooth edges.
    
    Args:
        mask: Input binary mask
        min_area: Minimum contour area to keep
        smoothing_kernel_size: Size of kernel for edge smoothing
    
    Returns:
        Filtered and smoothed mask
    """
    # Convert to 8-bit if needed
    mask_8bit = (mask * 255).astype(np.uint8) if mask.dtype != np.uint8 else mask
    
    # Find contours 
    contours, _ = cv2.findContours(mask_8bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create empty mask
    filtered_mask = np.zeros_like(mask_8bit)
    
    # Draw only contours with sufficient area
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            cv2.drawContours(filtered_mask, [contour], -1, 255, -1)
    
    # Smooth edges
    if smoothing_kernel_size > 0:
        kernel = np.ones((smoothing_kernel_size, smoothing_kernel_size), np.uint8)
        filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE, kernel)
        filtered_mask = cv2.GaussianBlur(filtered_mask, (smoothing_kernel_size, smoothing_kernel_size), 0)
        # Threshold back to binary
        _, filtered_mask = cv2.threshold(filtered_mask, 127, 255, cv2.THRESH_BINARY)
    
    return filtered_mask

def apply_black_tint(image, mask):
    # Filter the mask to remove small artifacts and smooth edges
    refined_mask = filter_region_mask(mask, min_area=500, smoothing_kernel_size=5)
    
    tinted_image = image.copy()
    black_color = np.array([100, 100, 100])  

    # Create a float version of mask for smoother transition
    mask_float = refined_mask.astype(np.float32) / 255.0
    
    # Apply a slight blur to the mask edges for smoother transition
    mask_float = cv2.GaussianBlur(mask_float, (5, 5), 0)

    for c in range(3):  
        tinted_image[:, :, c] = np.where(
            mask_float > 0.5,  
            0.85 * black_color[c] + 0.15 * image[:, :, c],  
            image[:, :, c]  
        )

    return tinted_image

def apply_blur(image, mask):
    """Apply blur with edge preservation"""
    mask = refine_mask_edges(mask)
    mask_8bit = (mask * 255).astype(np.uint8)
    
    # Strong blur for the main area
    blurred_image = cv2.GaussianBlur(image, (99, 99), 30)
    
    # Create edge mask
    edges = cv2.Canny(mask_8bit, 100, 200)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edge_mask = edges > 0
    
    # Combine original edges with blurred image
    result = image.copy()
    result[mask_8bit > 0] = blurred_image[mask_8bit > 0]
    result[edge_mask] = image[edge_mask]  # Preserve original edges
    
    return result

def smooth_car_edges(cropped_car, cropped_mask, edge_width=5):
    """
    Apply advanced edge smoothing to car cutout
    
    Args:
        cropped_car: Cropped car image
        cropped_mask: Binary mask of car region
        edge_width: Width of the transition area in pixels
    
    Returns:
        Tuple of (smoothed car image, smoothed mask)
    """
    # Create a float version of the mask
    mask_float = cropped_mask.astype(np.float32) / 255.0
    
    # Find edges using morphological operations
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(cropped_mask, kernel, iterations=edge_width)
    edge_mask = cropped_mask.copy()
    edge_mask[eroded > 0] = 0
    
    # Create gradient for smooth transition
    dist_transform = cv2.distanceTransform(edge_mask, cv2.DIST_L2, 3)
    max_dist = np.max(dist_transform)
    if max_dist > 0:  # Avoid division by zero
        gradient = dist_transform / max_dist
        gradient = 1.0 - gradient  # Invert so edges are transparent
    else:
        gradient = np.zeros_like(dist_transform)
    
    # Create a smooth transition mask
    smooth_mask = mask_float.copy()
    smooth_mask[edge_mask > 0] = gradient[edge_mask > 0]
    
    # Apply slight blur to the mask for even smoother edges
    smooth_mask = cv2.GaussianBlur(smooth_mask, (3, 3), 0.3)
    
    return cropped_car, smooth_mask.reshape(smooth_mask.shape[0], smooth_mask.shape[1], 1)

def normalize_car_distance(image, mask, background_path="background.png", offset_x=0, ground_offset=60, 
                         reference_car_height=600, width_adjustment_factors=(0.7, 1.3)):
    """
    Normalize car size accounting for different resolutions and angles.
    
    Args:
        image: Input image with car
        mask: Segmentation mask of the car
        background_path: Path to background image
        offset_x: Horizontal offset from center
        ground_offset: Distance from bottom of image to bottom of car
        reference_car_height: Target height for the car in pixels
        width_adjustment_factors: Tuple of (min_width_factor, max_width_factor) for angle compensation
    """
    # Refine mask and extract car
    mask = refine_mask_edges(mask)
    mask_8bit = (mask * 255).astype(np.uint8)
    car_pixels = cv2.bitwise_and(image, image, mask=mask_8bit)
    y_indices, x_indices = np.where(mask_8bit > 0)

    if not len(x_indices) or not len(y_indices):
        print("No valid car mask found.")
        return image, None

    # Get car bounding box
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    cropped_car = car_pixels[y_min:y_max + 1, x_min:x_max + 1]
    cropped_mask = mask_8bit[y_min:y_max + 1, x_min:x_max + 1]

    # Calculate current dimensions and aspect ratio
    current_height = y_max - y_min
    current_width = x_max - x_min
    aspect_ratio = current_width / current_height

    # Calculate scale factor based on reference height
    scale_factor = reference_car_height / current_height

    # Adjust width scaling based on aspect ratio (angle compensation)
    min_width_factor, max_width_factor = width_adjustment_factors
    width_adjustment = min(max_width_factor, max(min_width_factor, aspect_ratio * 1.5))
    scale_factor_width = scale_factor * width_adjustment

    # Calculate target dimensions
    target_height = reference_car_height
    target_width = int(current_width * scale_factor_width)

    # Ensure reasonable minimum width
    min_reasonable_width = int(reference_car_height * 0.5)
    target_width = max(target_width, min_reasonable_width)

    # Resize car with high-quality interpolation
    cropped_car = cv2.resize(cropped_car, (target_width, target_height), 
                           interpolation=cv2.INTER_CUBIC)
    cropped_mask = cv2.resize(cropped_mask, (target_width, target_height), 
                            interpolation=cv2.INTER_NEAREST)

    # Apply minimal edge smoothing - just enough to fix worst zigzags
    cropped_car, smooth_mask = smooth_car_edges(cropped_car, cropped_mask, edge_width=1)

    # Load and prepare background (1536x1024)
    bg_width, bg_height = 1536, 1024
    background = cv2.imread(background_path)
    if background is None:
        background = np.full((bg_height, bg_width, 3), 255, dtype=np.uint8)
    else:
        background = cv2.resize(background, (bg_width, bg_height))

    # Calculate position (centered horizontally)
    x_offset = max(0, (bg_width - target_width) // 2 + offset_x)
    y_offset = bg_height - target_height - ground_offset
    y_offset = max(0, min(y_offset, bg_height - target_height))
    x_offset = max(0, min(x_offset, bg_width - target_width))

    # Create an alpha compositing area
    roi = background[y_offset:y_offset + target_height, x_offset:x_offset + target_width]
    
    # Apply alpha compositing with smooth mask
    for c in range(3):
        background[y_offset:y_offset + target_height, x_offset:x_offset + target_width, c] = (
            smooth_mask[:, :, 0] * cropped_car[:, :, c] + 
            (1 - smooth_mask[:, :, 0]) * roi[:, :, c]
        )

    return background, None

def process_frames_with_depth_and_individual_tints(frames_folder, processed_folder, yolo_model_path, sam_config_path, sam_checkpoint_path,sam_model_path):
    # Load YOLO model
    yolo_model = YOLO(yolo_model_path)
    sam_model_P = SAM(sam_model_path)
    
    # Load SAM2 model with adjusted parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam_model = build_sam2(sam_config_path, sam_checkpoint_path, device=device)
    sam_model.eval()
    sam_model.to(device)
    sam_predictor = SAM2ImagePredictor(sam_model)
    sam_predictor.model.mask_threshold = 0.3  
    
    os.makedirs(processed_folder, exist_ok=True)
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith(('.jpg', '.png'))])

    print("Processing frames with enhanced edge detection...")

    region_classes = ["back", "front", "windb", "windf"]

    for frame_file in frame_files:
        frame_path = os.path.join(frames_folder, frame_file)
        image = cv2.imread(frame_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = yolo_model(image)
        class_names = results[0].names
        region_class_ids = {region: next((k for k, v in class_names.items() if v == region), None) for region in region_classes}
        plate_class_id = next((k for k, v in class_names.items() if v == "plate"), None)
        car_class_id = next((k for k, v in class_names.items() if v == "car"), None)

        region_boxes = {region: [] for region in region_classes}
        plate_boxes = []
        car_boxes = []

        for idx, cls in enumerate(results[0].boxes.cls):
            for region in region_classes:
                if int(cls) == region_class_ids[region]:
                    region_boxes[region].append(results[0].boxes.xyxy[idx].tolist())
            if int(cls) == plate_class_id:
                plate_boxes.append(results[0].boxes.xyxy[idx].tolist())
            if int(cls) == car_class_id:
                bbox = results[0].boxes.xyxy[idx].tolist()
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                car_boxes.append((bbox, area))

        # Set image for SAM predictor
        sam_predictor.set_image(image_rgb)

        # Process regions with multi-mask output for better edge capture
        for region in region_classes:
            boxes = region_boxes.get(region, [])
            if boxes:
                sam_results_region = sam_model_P(image_rgb, bboxes=np.array(boxes), verbose=False, save=False, device="cpu")
                region_masks = [result.masks.data.cpu().numpy() for result in sam_results_region]

                for masks in region_masks:
                    for mask in masks:
                        image = apply_black_tint(image, mask)

        # Process plates with edge preservation
        if plate_boxes:
            for box in plate_boxes:
                masks, _, _ = sam_predictor.predict(
                    box=np.array(box),
                    multimask_output=False
                )
                if len(masks) > 0:
                    image = apply_blur(image, masks[0])

        # Process cars with edge-aware normalization
        if car_boxes:
            largest_car_box = max(car_boxes, key=lambda x: x[1])[0]
            masks, _, _ = sam_predictor.predict(
                box=np.array(largest_car_box),
                multimask_output=False
            )
            
            if len(masks) > 0:
                
                processed_image, _ = normalize_car_distance(
                    image, 
                    masks[0],
                    background_path="Models/background.png",
                    offset_x= 0,                # Horizontal offset from center
                    ground_offset=100,           # Distance from bottom
                    reference_car_height=530,   # Target height in pixels
                    width_adjustment_factors=(0.85, 1.0)  # Min/max width adjustment for angles
                )
                
                output_path = os.path.join(processed_folder, f"processed_{frame_file}")
                cv2.imwrite(output_path, processed_image)
                print(f"Processed with edge refinement: {output_path}")

    print("All frames processed with enhanced edge detection.")

# Workflow Execution
if __name__ == "__main__":
    
    # Get the absolute path to the project root: "New"
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

    # Define paths relative to root
    frames_folder = os.path.join(ROOT_DIR, "User_Images")
    processed_folder = os.path.join(ROOT_DIR, "assets/images/processed")
    yolo_model_path = os.path.join(ROOT_DIR, "sam2/sam2/Models/lastnew2.pt")
    sam_config_path = os.path.join(ROOT_DIR, "sam2/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml")
    sam_checkpoint_path = os.path.join(ROOT_DIR, "sam2/sam2/Models/checkpoint50.pt")
    sam_model_path = os.path.join(ROOT_DIR, "sam2/sam2/Models/sam2.1_l.pt")

    process_frames_with_depth_and_individual_tints(
        frames_folder,
        processed_folder,
        yolo_model_path,
        sam_config_path,
        sam_checkpoint_path,
        sam_model_path
    )