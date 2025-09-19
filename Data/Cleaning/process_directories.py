import os
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import matplotlib.pyplot as plt
import csv
import argparse
from tqdm import tqdm


MIN_WIDTH = 256
MIN_HEIGHT = 256


class UltrasoundProcessor:
    """Optimized processor for ultrasound images and videos"""
    
    def __init__(self, model_name="Qwen/Qwen2.5-VL-3B-Instruct"):
        """Initialize the processor with optimized settings"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model {model_name} on {self.device}...")
        
        # Load model with optimizations
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        
        if self.device == "cuda":
            self.model = self.model.half().cuda()
        
        print("Model loaded successfully")
    
    def process_image(self, image_path):
        """Process a single ultrasound image"""
        # Load the image
        if isinstance(image_path, str):
            cv_image = cv2.imread(image_path)
        else:
            cv_image = cv2.cvtColor(np.array(image_path), cv2.COLOR_RGB2BGR)
        
        # Make sure we have a valid image
        if cv_image is None or cv_image.size == 0:
            print(f"Error: Failed to load image or empty image: {image_path}")
            return None, None, None
        
        # Remove top 20 pixels to eliminate header text
        if cv_image.shape[0] > 20:
            cv_image = cv_image[20:, :]
        
        # Convert for model input
        pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        
        # Run full analysis for each image
        try:
            # Try to run model with CUDA
            # Run full analysis for each image
            result = self._analyze_image(pil_image, cv_image)
        except RuntimeError as e:
            # If CUDA error, free cache and try again
            if "CUDA" in str(e):
                print("CUDA error detected, trying to recover...")
                # Clear CUDA cache
                torch.cuda.empty_cache()
                
                # Try again with model on CPU
                with torch.no_grad():
                    device_backup = self.device
                    self.device = "cpu"
                    result = self._analyze_image(pil_image, cv_image)
                    self.device = device_backup
            else:
                raise
        
        # Create the final processed image

        processed_image = self._create_processed_image(cv_image, result)
        
        return processed_image, result["site"], result["depth"]
    
    def _analyze_image(self, pil_image, cv_image):
        """Analyze image with Qwen model to extract B-tag, site, and depth"""
        # Run Qwen analysis
        qwen_result = self._run_qwen_analysis(pil_image)
        
        # Extract B-tag location
        b_tag_coords = qwen_result.get("b_tag_coords")
        
        # Remove B-tag using color-based method
        inpainted_image, b_tag_bbox = self.remove_tag(cv_image)
        
        # Remove ruler from right side
        ruler_removed = self.remove_ruler(inpainted_image)
        
        # Extract ROI
        roi_result = self.extract_roi(ruler_removed)
        
        # Combine all results
        result = {
            "b_tag_coords": b_tag_coords,
            "b_tag_bbox": b_tag_bbox,
            "site": qwen_result.get("site", "noSite"),
            "depth": qwen_result.get("depth", 15),
            "roi_coords": roi_result["roi_coords"],
            "processing_params": {
                "b_tag_bbox": b_tag_bbox,
                "roi_coords": roi_result["roi_coords"]
            }
        }
        
        return result
    
    def _apply_cached_processing(self, cv_image, cached_result):
        """Apply cached processing parameters to a video frame"""
        # Get cached parameters
        b_tag_bbox = cached_result.get("b_tag_bbox")
        roi_coords = cached_result.get("roi_coords")
        
        # Make a copy to avoid modifying the original
        processed = cv_image.copy()
        
        # Apply B-tag removal
        if b_tag_bbox:
            start_y, end_y, start_x, end_x = b_tag_bbox
            processed[start_y:end_y, start_x:end_x] = 0
        else:
            # If no specific bbox, run color-based removal
            inpainted_image, _ = self.remove_tag(processed)
            processed = inpainted_image
        
        # Remove ruler
        processed = self.remove_ruler(processed)
        
        # Apply ROI cropping
        if roi_coords:
            x1, y1, x2, y2 = roi_coords
            # Make sure the coordinates are valid
            if 0 <= y1 < y2 <= processed.shape[0] and 0 <= x1 < x2 <= processed.shape[1]:
                cropped = processed[y1:y2, x1:x2]
            else:
                # Extract a new ROI if coordinates are invalid
                roi_result = self.extract_roi(processed)
                cropped = roi_result["cropped_image"]
        else:
            # Extract a new ROI
            roi_result = self.extract_roi(processed)
            cropped = roi_result["cropped_image"]
        
        # Ensure the cropped image is valid
        if cropped is None or cropped.size == 0:
            return processed  # Return the processed but uncropped image
        
        enhanced = enhance_ultrasound_image(cropped)
    
        return enhanced
    
    def _create_processed_image(self, cv_image, result):
        """Create final processed image"""
        # Make a copy to avoid modifying the original
        processed = cv_image.copy()
        
        # Apply B-tag removal
        b_tag_bbox = result.get("b_tag_bbox")
        if b_tag_bbox:
            start_y, end_y, start_x, end_x = b_tag_bbox
            processed[start_y:end_y, start_x:end_x] = 0
        else:
            # If no specific bbox, run color-based removal
            inpainted_image, _ = self.remove_tag(processed)
            processed = inpainted_image
        
        # Remove ruler
        processed = self.remove_ruler(processed)
        
        # Apply ROI cropping
        roi_coords = result.get("roi_coords")
        if roi_coords:
            x1, y1, x2, y2 = roi_coords
            # Make sure the coordinates are valid
            if 0 <= y1 < y2 <= processed.shape[0] and 0 <= x1 < x2 <= processed.shape[1]:
                cropped = processed[y1:y2, x1:x2]
            else:
                # Extract a new ROI if coordinates are invalid
                roi_result = self.extract_roi(processed)
                cropped = roi_result["cropped_image"]
        else:
            # Extract a new ROI
            roi_result = self.extract_roi(processed)
            cropped = roi_result["cropped_image"]
        
        # Ensure the cropped image is valid
        if cropped is None or cropped.size == 0:
            return processed  # Return the processed but uncropped image

        enhanced = enhance_ultrasound_image(cropped)
        
        return enhanced
    
    def _run_qwen_analysis(self, pil_image):
        """Run Qwen VLM analysis on the image"""
        # Create a specific prompt for the VLM with explicit instructions about site location
        system_prompt = """
You are an AI specialized in analyzing ultrasound images. Your task is to examine the given ultrasound image and extract the following information:

1. B-tag: Identify the blue circular marker labeled with 'B' in the image. 
   - Provide its coordinates as (x, y)

2. Site: Find the text at the BOTTOM of the image that indicates the anatomical site or label code. Look for codes like "QSLG", "RLIG", etc. Only look at the bottom half of the image for a 3-4 letter code. DO NOT use labels from the top such as "poumon" or "abdomen".

3. Depth: Look at the ruler/scale on the right side of the image and determine the maximum depth shown (typically 5cm or 15cm).

Present your answer in this exact format:
B-tag: (x, y)
Site: [text]
Depth: [number] cm
"""

        # Prepare the message structure
        user_messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                ]
            }
        ]

        # Process with the model
        text_prompt = self.processor.apply_chat_template(user_messages, add_generation_prompt=True)
        inputs = self.processor(
            text=[text_prompt],
            images=[pil_image],
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            try:
                torch.cuda.empty_cache()
                output_ids = self.model.generate(**inputs, 
                max_new_tokens=1024,
                do_sample=False)
            except RuntimeError as e:
                if "CUDA" in str(e) or "probability tensor" in str(e):
                    print("CUDA probability error, falling back to CPU")
                    # Fall back to CPU processing
                    inputs = {k: v.cpu() for k, v in inputs.items()}
                    old_device = self.device
                    self.model = self.model.cpu()
                    self.device = "cpu"
                    
                    # Try with more conservative parameters
                    output_ids = self.model.generate(
                        **inputs, 
                        max_new_tokens=1024,
                        temperature=0.1,
                        do_sample=False
                    )
                    
                    # Move back to original device if needed
                    if old_device == "cuda":
                        self.model = self.model.cuda()
                        self.device = "cuda"
                else:
                    raise  # Re-ra
            
            
        generated_ids = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)
        ]
        qwen_output = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]
        
        # Parse the model output
        result = self._parse_qwen_response(qwen_output)
        
        return result
    
    def _parse_qwen_response(self, response):
        """Parse the model's response into structured data with site text normalization"""
        result = {
            "b_tag_coords": None,
            "site": None,
            "depth": None,
        }
        
        # Parse each line
        for line in response.split('\n'):
            line = line.strip()
            
            if line.startswith("B-tag:"):
                # Extract coordinates like (x, y)
                coords_text = line.replace("B-tag:", "").strip()
                import re
                coords_match = re.search(r'\((\d+),\s*(\d+)\)', coords_text)
                if coords_match:
                    x, y = int(coords_match.group(1)), int(coords_match.group(2))
                    result["b_tag_coords"] = (x, y)
            
            elif line.startswith("Site:"):
                site_text = line.replace("Site:", "").strip()
                # Normalize site text
                if site_text:
                    # Convert to uppercase and remove spaces
                    site_text = site_text.upper().replace(" ", "")
                    
                    # Normalize common variants
                    if site_text in ['QUIG', 'QUIG', 'QLUIG', 'QLUIG', 'QLIG']:
                        site_text = 'QLIG'
                    
                    # Remove unwanted words
                    for word in ['POUMON', 'ABDOMEN', 'THORAX']:
                        if word in site_text:
                            site_text = site_text.replace(word, '')
                    
                    # Only keep valid characters (alphanumeric)
                    site_text = ''.join(c for c in site_text if c.isalnum())
                    
                    # Strip and check if still valid
                    site_text = site_text.strip()
                    if not site_text or len(site_text) > 6:
                        site_text = 'UNKNOWN'
                else:
                    site_text = 'UNKNOWN'
                
                result["site"] = site_text
                
            elif line.startswith("Depth:"):
                depth_text = line.replace("Depth:", "").strip()
                # Extract just the number
                import re
                depth_match = re.search(r'(\d+)', depth_text)
                if depth_match:
                    result["depth"] = int(depth_match.group(1))
                else:
                    # Default depth if not found
                    result["depth"] = 15
        
        return result
    
    def process_video(self, video_path, output_path):
        """Process video by analyzing first frame and applying to all frames"""
        # Open the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None, None
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Read first frame for analysis
        ret, first_frame = cap.read()
        if not ret:
            print(f"Error: Could not read first frame from {video_path}")
            cap.release()
            return None, None
        
        # Remove top 20 pixels
        if first_frame.shape[0] > 20:
            first_frame = first_frame[20:, :]
        
        # Convert to PIL for model
        first_frame_pil = Image.fromarray(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))

        try:
            # Try to run model with CUDA
            # Run full analysis for each image
            result = self._analyze_image(first_frame_pil, first_frame)
        except RuntimeError as e:
            # If CUDA error, free cache and try again
            if "CUDA" in str(e):
                print("CUDA error detected, trying to recover...")
                # Clear CUDA cache
                torch.cuda.empty_cache()
                
                # Try again with model on CPU
                with torch.no_grad():
                    device_backup = self.device
                    self.device = "cpu"
                    result = self._analyze_image(first_frame_pil, first_frame)
                    self.device = device_backup
            else:
                raise


        
        # Full analysis on first frame
        #result = self._analyze_image(first_frame_pil, first_frame)
        site_code = result.get("site", "UNKNOWN")
        depth = result.get("depth", 15)
        
        # Process the first frame to get template
        processed_first = self._create_processed_image(first_frame, result)
        
        # Minimum dimensions for processed output
        min_width = MIN_WIDTH
        min_height = MIN_HEIGHT
        
        # Get dimensions of processed frame for consistent output
        if processed_first is not None and processed_first.size > 0:
            output_height, output_width = processed_first.shape[:2]
            
            # Ensure minimum dimensions
            output_width = max(output_width, min_width)
            output_height = max(output_height, min_height)
        else:
            # If processing failed, use minimum dimensions
            output_height, output_width = min_height, min_width
        
        # Create video writer with processed dimensions
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MPEG-4 codec
        out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
        
        if not out.isOpened():
            print(f"Error: Could not create output video file at {output_path}")
            cap.release()
            return None, None
        
        # Reset video to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Process all frames
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processed_frames = 0
        
        for _ in tqdm(range(frame_count), desc=f"Processing video {os.path.basename(video_path)}"):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Remove top 20 pixels
            if frame.shape[0] > 20:
                frame = frame[20:, :]
            
            # Process frame using cached parameters
            processed = self._apply_cached_processing(frame, result)
            
            # Ensure processed frame is valid
            if processed is None or processed.size == 0:
                continue
            
            # Ensure minimum dimensions and consistent size
            current_h, current_w = processed.shape[:2]
            if current_h < min_height or current_w < min_width or current_h != output_height or current_w != output_width:
                # Resize to match minimum dimensions while maintaining aspect ratio
                aspect = current_w / current_h
                
                if aspect > 1:  # Wider than tall
                    new_w = max(min_width, output_width)
                    new_h = int(new_w / aspect)
                    if new_h < min_height:
                        new_h = min_height
                        new_w = int(new_h * aspect)
                else:  # Taller than wide
                    new_h = max(min_height, output_height)
                    new_w = int(new_h * aspect)
                    if new_w < min_width:
                        new_w = min_width
                        new_h = int(new_w / aspect)
                
                # Resize to the calculated dimensions
                processed = cv2.resize(processed, (new_w, new_h))
                
                # If still not matching output dimensions, create a black canvas and paste centered
                if new_h != output_height or new_w != output_width:
                    canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
                    y_offset = (output_height - new_h) // 2
                    x_offset = (output_width - new_w) // 2
                    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = processed
                    processed = canvas
            
            # Convert to BGR for OpenCV if needed
            if len(processed.shape) == 3 and processed.shape[2] == 3:
                if processed.dtype != np.uint8:
                    processed = processed.astype(np.uint8)
                processed_bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
            else:
                # If grayscale, convert to BGR
                processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            
            # Write frame
            out.write(processed_bgr)
            processed_frames += 1
        
        # Clean up
        cap.release()
        out.release()
        
        print(f"Processed {processed_frames} frames for video {os.path.basename(video_path)}")
        
        # Verify the video was created properly
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0 and processed_frames > 0:
            return site_code, depth
        else:
            print(f"Warning: Output video file may be invalid: {output_path}")
            if os.path.exists(output_path):
                os.remove(output_path)
            return None, None
    
    def remove_tag(self, image):
        """Remove B-tag using color-based masking and inpainting"""
        # Make sure we have a valid image
        if image is None or image.size == 0:
            return image, None
            
        # Convert to different color spaces
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for detection (blue B-tag and orange artifacts)
        lower_blue_hsv = np.array([100, 100, 50])
        upper_blue_hsv = np.array([140, 255, 255])
        lower_orange_hsv = np.array([5, 50, 50])
        upper_orange_hsv = np.array([25, 255, 255])
        
        # Create masks
        mask_blue_hsv = cv2.inRange(image_hsv, lower_blue_hsv, upper_blue_hsv)
        mask_orange_hsv = cv2.inRange(image_hsv, lower_orange_hsv, upper_orange_hsv)
        mask_combined = cv2.bitwise_or(mask_blue_hsv, mask_orange_hsv)
        
        # Dilate the mask to ensure complete coverage
        kernel = np.ones((5, 5), np.uint8)
        mask_dilated = cv2.dilate(mask_combined, kernel, iterations=2)
        
        # Inpaint the image to remove the tags
        inpainted_image = cv2.inpaint(image_rgb, mask_dilated, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        
        # Get the coordinates of the modified area for future reference
        y_coords, x_coords = np.where(mask_dilated != 0)
        if len(y_coords) > 0 and len(x_coords) > 0:
            start_y, end_y = np.min(y_coords), np.max(y_coords)
            start_x, end_x = np.min(x_coords), np.max(x_coords)
            coords_b = (start_y, end_y, start_x, end_x)
        else:
            coords_b = None
        
        return inpainted_image, coords_b
    
    def remove_ruler(self, image):
        """Remove ruler at the right side of the image"""
        # Make sure we have a valid image
        if image is None or image.size == 0:
            return image
            
        height, width = image.shape[:2]
        
        # Assume ruler is in right 15% of image
        ruler_width = int(width * 0.05)
        
        # Create a mask to remove the right portion (ruler area)
        mask = np.ones((height, width), dtype=np.uint8)
        mask[:, width-ruler_width:] = 0
        
        # Apply mask to each channel
        result = image.copy()
        if len(image.shape) == 3:
            for c in range(image.shape[2]):
                result[:, :, c] = image[:, :, c] * mask
        else:
            result = image * mask
        
        return result
    
    def extract_roi(self, image):
        """Extract ROI focusing specifically on the ultrasound fan area and removing bottom labels"""
        # Make sure we have a valid image
        if image is None or image.size == 0:
            return {"cropped_image": image, "binary_image": None, "roi_coords": (0, 0, 0, 0)}
                
        # Convert to grayscale if needed
        if len(image.shape) > 2 and image.shape[2] == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image
        
        height, width = gray_image.shape[:2]
        
        # Remove the bottom 10% where site labels often appear
        bottom_cutoff = int(height * 0.9)
        # Create a mask to remove the bottom
        mask = np.ones_like(gray_image)
        mask[bottom_cutoff:, :] = 0
        
        # Apply mask to remove bottom section
        if len(image.shape) == 3:
            masked_image = image.copy()
            for c in range(3):
                masked_image[:, :, c] = image[:, :, c] * mask
        else:
            masked_image = image * mask
        
        # Create binary image with lower threshold to capture ultrasound data
        _, img_binary = cv2.threshold(gray_image * mask, 25, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean up
        kernel = np.ones((7, 7), np.uint8)
        img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel)
        
        # Find non-zero pixels (white areas)
        non_zero_points = cv2.findNonZero(img_binary)
        
        # Set minimum dimensions for ROI
        min_width = MIN_WIDTH  # Minimum width in pixels
        min_height = MIN_HEIGHT  # Minimum height in pixels
        
        if non_zero_points is not None and len(non_zero_points) > 0:
            # Get bounding rectangle around all non-zero pixels
            x, y, w, h = cv2.boundingRect(non_zero_points)
            
            # Add padding
            padding_x = int(w * 0.1)  # 10% horizontal padding
            padding_y = int(h * 0.1)  # 10% vertical padding
            
            # Ensure bounds are within image and not beyond bottom cutoff
            x = max(0, x - padding_x)
            y = max(0, y - padding_y)
            w = min(width - x, w + 2*padding_x)
            h = min(bottom_cutoff - y, h + padding_y)
            
            # Ensure minimum dimensions
            if w < min_width:
                # Center the expansion
                center_x = x + w//2
                new_x = max(0, center_x - min_width//2)
                w = min(width - new_x, min_width)
                x = new_x
            
            if h < min_height:
                # Center the expansion
                center_y = y + h//2
                new_y = max(0, center_y - min_height//2)
                h = min(bottom_cutoff - new_y, min_height)
                y = new_y
            
            # Crop to this region
            roi = masked_image[y:y+h, x:x+w]
            roi_coords = (x, y, x+w, y+h)
        else:
            # Fallback to center portion of image if no contours found
            center_x = width // 2
            center_y = (bottom_cutoff // 2)
            
            # Ensure minimum dimensions
            w = max(min_width, width // 2)
            h = max(min_height, bottom_cutoff // 2)
            
            # Center the crop region
            x = max(0, center_x - w//2)
            y = max(0, center_y - h//2)
            
            # Adjust if too close to edges
            if x + w > width:
                x = max(0, width - w)
            if y + h > bottom_cutoff:
                y = max(0, bottom_cutoff - h)
            
            roi = masked_image[y:y+h, x:x+w]
            roi_coords = (x, y, x+w, y+h)
        
        return {
            "cropped_image": roi,
            "binary_image": img_binary,
            "roi_coords": roi_coords
        }
def process_directory(input_folder, output_folder):
    """Process a directory containing patient folders with ultrasound images and videos"""
    # Initialize processor
    processor = UltrasoundProcessor()
    
    # Create output directories
    image_output_path = os.path.join(output_folder, 'images')
    video_output_path = os.path.join(output_folder, 'videos')
    os.makedirs(image_output_path, exist_ok=True)
    os.makedirs(video_output_path, exist_ok=True)
    
    # Path to CSV file
    csv_file_path = os.path.join(output_folder, 'processed_files.csv')
    
    # Check if CSV exists and load already processed patient IDs
    processed_patients = set()
    csv_mode = 'w'  # Default to write mode
    write_header = True  # Default to writing header
    
    if os.path.exists(csv_file_path) and os.path.getsize(csv_file_path) > 0:
        try:
            with open(csv_file_path, mode='r') as existing_file:
                reader = csv.reader(existing_file)
                try:
                    next(reader)  # Try to skip header
                    for row in reader:
                        if row and len(row) > 0:
                            processed_patients.add(row[0])  # Add patient ID to set
                    print(f"Found {len(processed_patients)} already processed patients")
                    
                    # Open file in append mode if we successfully read it
                    csv_mode = 'a'
                    write_header = False
                except StopIteration:
                    # File exists but is empty or corrupted
                    print("CSV file exists but is empty or corrupted. Creating new file.")
                    csv_mode = 'w'
                    write_header = True
        except Exception as e:
            print(f"Error reading existing CSV file: {e}. Creating new file.")
            csv_mode = 'w'
            write_header = True
    
    print(f"Opening CSV file in mode: {csv_mode}, with write_header={write_header}")
    
    # Use buffered writing to ensure data is written immediately
    with open(csv_file_path, mode=csv_mode, newline='', buffering=1) as csv_file:
        writer = csv.writer(csv_file)
        
        # Write header if needed
        if write_header:
            writer.writerow(['Patient ID', 'Site', 'Depth', 'Count', 'New File Name', 'Original File Name'])
            csv_file.flush()  # Force write to disk
            print("Header written to CSV")
        
        # Process each patient directory
        patient_dirs = [d for d in os.listdir(input_folder) if not d.startswith('.')]
        for patient_id in tqdm(patient_dirs, desc="Processing patients"):
            patient_folder = os.path.join(input_folder, patient_id)
            if not os.path.isdir(patient_folder):
                continue
                
            # Use first part of ID as the final patient ID
            final_patient_id = patient_id.split(' ')[0]
            
            # Skip if already processed
            # if final_patient_id in processed_patients:
            #     print(f"Skipping already processed patient: {final_patient_id}")
            #     continue
            
            print(f"Processing patient: {final_patient_id}")
            
            # Track counts for each site+depth combination
            site_depth_counts = {}
            patient_files_processed = 0
            
            # Process all files in patient folder
            for file_name in os.listdir(patient_folder):
                file_path = os.path.join(patient_folder, file_name)
                
                # Process PNG images
                if file_name.lower().endswith('.png'):
                    if '(1).png' in file_name or '(2).png' in file_name:
                        continue
                    try:
                        # Process the image
                        processed_image, site, depth = processor.process_image(file_path)
                        
                        if processed_image is not None and site is not None:
                            # Handle counts
                            site_depth_key = f"{site}_{depth}"
                            if site_depth_key not in site_depth_counts:
                                site_depth_counts[site_depth_key] = 0
                            count = site_depth_counts[site_depth_key]
                            site_depth_counts[site_depth_key] += 1
                            
                            # Save processed image
                            new_file_name = f"{final_patient_id}_{site}_{depth}_{count}.png"
                            save_path = os.path.join(image_output_path, new_file_name)
                            
                            # Convert to BGR for OpenCV saving
                            if len(processed_image.shape) == 3 and processed_image.shape[2] == 3:
                                save_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
                            else:
                                save_image = processed_image
                                
                            cv2.imwrite(save_path, save_image)
                            
                            # Write to CSV and flush immediately
                            writer.writerow([final_patient_id, site, depth, count, new_file_name, file_name])
                            csv_file.flush()  # Force write to disk
                            patient_files_processed += 1
                            
                            print(f"Processed image: {new_file_name}")
                    
                    except Exception as e:
                        print(f"Error processing image {file_name}: {e}")
                
                # Process MP4 videos
                elif file_name.lower().endswith('.mp4'):
                    if '(1).mp4' in file_name or '(2).mp4' in file_name:
                        continue
                    try:
                        # Process the video
                        output_path = os.path.join(video_output_path, f"{final_patient_id}_temp.mp4")
                        site, depth = processor.process_video(file_path, output_path)
                        
                        if site and depth:
                            # Handle counts
                            site_depth_key = f"{site}_{depth}"
                            if site_depth_key not in site_depth_counts:
                                site_depth_counts[site_depth_key] = 0
                            count = site_depth_counts[site_depth_key]
                            site_depth_counts[site_depth_key] += 1
                            
                            # Rename to final filename
                            new_file_name = f"{final_patient_id}_{site}_{depth}_{count}.mp4"
                            new_output_path = os.path.join(video_output_path, new_file_name)
                            
                            # Safely rename the file
                            if os.path.exists(output_path):
                                os.rename(output_path, new_output_path)
                                
                                # Write to CSV and flush immediately
                                writer.writerow([final_patient_id, site, depth, count, new_file_name, file_name])
                                csv_file.flush()  # Force write to disk
                                patient_files_processed += 1
                                
                                print(f"Processed video: {new_file_name}")
                            else:
                                print(f"Warning: Temp video file not found: {output_path}")
                            
                    except Exception as e:
                        print(f"Error processing video {file_name}: {e}")
                        # Clean up temp file if it exists
                        temp_path = os.path.join(video_output_path, f"{final_patient_id}_temp.mp4")
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
            
            # After processing all files for this patient
            print(f"Completed patient {final_patient_id}, processed {patient_files_processed} files")
            processed_patients.add(final_patient_id)
            
    # Double-check file was written
    if os.path.exists(csv_file_path):
        print(f"CSV file exists at: {csv_file_path}")
        print(f"CSV file size: {os.path.getsize(csv_file_path)} bytes")
    else:
        print(f"WARNING: CSV file does not exist at: {csv_file_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ultrasound images and videos from multiple patient directories")
    parser.add_argument("--input", required=True, help="Input root directory containing patient folders")
    parser.add_argument("--output", required=True, help="Output directory for processed files")
    
    args = parser.parse_args()
    process_directory(args.input, args.output)


#python3 /gpfs/gibbs/project/hartley/tjb76/artstuff_OPTIMIZEDWOOOO/Data/Cleaning/process_directories.py --input '/gpfs/gibbs/project/hartley/tjb76/artstuff_OPTIMIZEDWOOOO/Data/CLUSSTER-Benin/Uncleaned_Videos/Uncleaned_v9' --output '/gpfs/gibbs/project/hartley/tjb76/artstuff_OPTIMIZEDWOOOO/Data/CLUSSTER-Benin/cleaned_v2'


#python3 /gpfs/gibbs/project/hartley/tjb76/artstuff_OPTIMIZEDWOOOO/Data/Cleaning/process_directories.py --input '/gpfs/gibbs/project/hartley/tjb76/artstuff_OPTIMIZEDWOOOO/Data/CLUSSTER-Benin/Benin-Uncleaned5' --output '/gpfs/gibbs/project/hartley/tjb76/artstuff_OPTIMIZEDWOOOO/Data/CLUSSTER-Benin/cleaned_v2'


