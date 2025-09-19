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
import gc
import shutil
MIN_WIDTH = 256
MIN_HEIGHT = 256

from collections import OrderedDict

# SITE_MAPPING = OrderedDict(
#     [
#         ("<PAD>", 0),
#         ("QAID", 1),
#         ("QAIG", 2),
#         ("QASD", 3),
#         ("QASG", 4),
#         ("QLD", 5),
#         ("QLG", 6),
#         ("QPID", 7),
#         ("QPIG", 8),
#         ("QPSD", 9),
#         ("QPSG", 10),
#         ("APXD", 11),
#         ("APXG", 12),
#         ("QSLD", 13),
#         ("QSLG", 14),
#     ]
# )

class UltrasoundProcessor:
    
    def __init__(self, model_name="Qwen/Qwen2.5-VL-3B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model {model_name} on {self.device}...")
        
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        
        if self.device == "cuda":
            self.model = self.model.half().cuda()
        
        print("Model loaded successfully")
    
    def process_image(self, image_path):
        if isinstance(image_path, str):
            cv_image = cv2.imread(image_path)
        else:
            cv_image = cv2.cvtColor(np.array(image_path), cv2.COLOR_RGB2BGR)
        
        if cv_image is None or cv_image.size == 0:
            print(f"Error: Failed to load image or empty image: {image_path}")
            return None, None, None
        
        # Remove top 20 pixels to eliminate header text
        if cv_image.shape[0] > 20:
            cv_image = cv_image[20:, :]
        
        pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        
        try:
            result = self._analyze_image(pil_image, cv_image)
        except RuntimeError as e:
            # If CUDA error, free cache and try again
            if "CUDA" in str(e):
                print("CUDA error detected, trying to recover...")
                torch.cuda.empty_cache()
                
                # Try again with model on CPU
                with torch.no_grad():
                    device_backup = self.device
                    self.device = "cpu"
                    result = self._analyze_image(pil_image, cv_image)
                    self.device = device_backup
            else:
                raise
        

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
        b_tag_bbox = cached_result.get("b_tag_bbox")
        roi_coords = cached_result.get("roi_coords")
        
        processed = cv_image.copy()
        
        if b_tag_bbox:
            start_y, end_y, start_x, end_x = b_tag_bbox
            processed[start_y:end_y, start_x:end_x] = 0
        else:
            inpainted_image, _ = self.remove_tag(processed)
            processed = inpainted_image
        
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
            roi_result = self.extract_roi(processed)
            cropped = roi_result["cropped_image"]
        
        # Ensure the cropped image is valid
        if cropped is None or cropped.size == 0:
            return processed  
            
        return cropped
    
    def _create_processed_image(self, cv_image, result):
        """Create final processed image"""
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
            if 0 <= y1 < y2 <= processed.shape[0] and 0 <= x1 < x2 <= processed.shape[1]:
                cropped = processed[y1:y2, x1:x2]
            else:
                roi_result = self.extract_roi(processed)
                cropped = roi_result["cropped_image"]
        else:
            # Extract a new ROI
            roi_result = self.extract_roi(processed)
            cropped = roi_result["cropped_image"]
        
        # Ensure the cropped image is valid
        if cropped is None or cropped.size == 0:
            return processed  
        
        return cropped
    
    def _run_qwen_analysis(self, pil_image):
        pil_image = pil_image.convert('RGB')  # Ensure RGB mode
    
        # Check and resize if too large
        if pil_image.width > 1024 or pil_image.height > 1024:
            ratio = min(1024 / pil_image.width, 1024 / pil_image.height)
            new_width = int(pil_image.width * ratio)
            new_height = int(pil_image.height * ratio)
            pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
    
        def deep_validate_inputs(inputs):
            """Perform deep validation of input tensors"""
            print("\n--- Detailed Input Debugging ---")
            for key, tensor in inputs.items():
                print(f"\nKey: {key}")
                
                # Print basic details for all input types
                if isinstance(tensor, torch.Tensor):
                    print(f"Tensor Type: {type(tensor)}")
                    print(f"Shape: {tensor.shape}")
                    print(f"Dtype: {tensor.dtype}")
                    print(f"Device: {tensor.device}")
                    
                    # Detailed numeric checks for float tensors
                    if tensor.dtype in [torch.float16, torch.float32, torch.float64]:
                        print(f"Min value: {tensor.min()}")
                        print(f"Max value: {tensor.max()}")
                        print(f"Mean value: {tensor.mean()}")
                        print(f"Contains NaN: {torch.isnan(tensor).any()}")
                        print(f"Contains Inf: {torch.isinf(tensor).any()}")
                        print(f"Negative values: {(tensor < 0).sum()}")
                else:
                    print(f"Non-tensor type: {type(tensor)}")
                    print(f"Value: {tensor}")
            
            print("\n--- Image Details ---")
            # Add image-specific debugging
            print(f"Image mode: {pil_image.mode}")
            print(f"Image size: {pil_image.size}")

                
        """Run Qwen VLM analysis on the image with improved site detection"""
        system_prompt = """
    You are an AI specialized in analyzing ultrasound images. Your task is to carefully examine the given ultrasound image and extract the following information:

    1. B-tag: Identify the blue circular marker labeled with 'B' in the image. 
    - Provide its coordinates as (x, y)

    2. Site: Find the anatomical site code at the BOTTOM of the image. This is a 3-4 letter code like "QSLG", "QPSG", "QAIG", "QPID", etc. 
    - IMPORTANT: Only consider codes containing Q, A, P, S, L, I, D, G letters in the site code.
    - Common valid site codes are: QAID, QAIG, QASD, QASG, QLD, QLG, QPID, QPIG, QPSD, QPSG, APXD, APXG, QSLD, QSLG, QLSD, SLG, SLD, SAG, SAD, SPD, SPG
    - DO NOT use labels from the top such as "poumon" or "abdomen".
    - If you can't find a site code, respond with "UNKNOWN"

    3. Depth: Look at the ruler/scale on the right side of the image and determine the maximum depth shown (typically 5cm or 15cm).

    Present your answer in this exact format:
    B-tag: (x, y)
    Site: [code]
    Depth: [number] cm
    """

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
        #print("Applying chat tempalte")
        text_prompt = self.processor.apply_chat_template(user_messages, add_generation_prompt=True)
        #print("Started Processing Image")
        inputs = self.processor(
            text=[text_prompt],
            images=[pil_image],
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        #print("Processed Image")

        for key, tensor in inputs.items():
            if torch.is_tensor(tensor):
                # Check for NaN or Inf
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    print(f"Invalid tensor in {key}: NaN or Inf detected")
                    return None
                
                # Clip values to prevent extreme values
                tensor = torch.clamp(tensor, min=0, max=1)

        if 'pixel_values' in inputs:
            inputs['pixel_values'] = torch.clamp(inputs['pixel_values'], -3.0, 3.0)
    

        with torch.no_grad():
            try:
                torch.cuda.empty_cache()
                gc.collect()
                #print("Starting generation")
                #deep_validate_inputs(inputs)
                output_ids = self.model.generate(**inputs, 
                max_new_tokens=1024)
                #print("Finished Generation")
            except RuntimeError as e:
                print("Caught error")
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                gc.collect()
                # Handle specific CUDA errors
                if "CUDA" in str(e) or "device-side assert" in str(e) or "probability tensor" in str(e) or "out of memory" in str(e):
                    print(f"CUDA error during generation: {str(e)}")
                    # Fall back to CPU processing
                    print("Falling back to CPU for this generation")
                    
                    # Move everything to CPU
                    cpu_inputs = {k: v.cpu() for k, v in inputs.items()}
                    cpu_model = self.model.cpu()
                    
                    # Try generation on CPU
                    output_ids = self.model.generate(
                        **cpu_inputs, 
                        max_new_tokens=200,
                        do_sample=False,
                        temperature=1.0,
                        num_beams=1
                    )
                    
                    # Move model back to original device if possible
                    if self.device == "cuda":
                        try:
                            self.model = self.model.cuda()
                        except:
                            print("Could not move model back to CUDA")
                else:
                    # Re-raise non-CUDA errors
                    raise
            
        generated_ids = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)
        ]
        qwen_output = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]
        
        #print("Raw Qwen output:")
        #print(qwen_output)
        
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
                coords_text = line.replace("B-tag:", "").strip()
                import re
                coords_match = re.search(r'\((\d+),\s*(\d+)\)', coords_text)
                if coords_match:
                    x, y = int(coords_match.group(1)), int(coords_match.group(2))
                    result["b_tag_coords"] = (x, y)
            
            elif line.startswith("Site:"):
                site_text = line.replace("Site:", "").strip()
                if site_text:
                    site_text = site_text.upper().replace(" ", "")
                    
                    if site_text in ['QUIG', 'QUIG', 'QLUIG', 'QLUIG', 'QLIG']:
                        site_text = 'QLIG'
                    
                    for word in ['POUMON', 'ABDOMEN', 'THORAX']:
                        if word in site_text:
                            site_text = site_text.replace(word, '')
                    
                    site_text = ''.join(c for c in site_text if c.isalnum())
                    
                    site_text = site_text.strip()
                    if not site_text or len(site_text) > 6:
                        site_text = 'UNKNOWN'
                else:
                    site_text = 'UNKNOWN'
                
                result["site"] = site_text
                
            elif line.startswith("Depth:"):
                depth_text = line.replace("Depth:", "").strip()
                import re
                depth_match = re.search(r'(\d+)', depth_text)
                if depth_match:
                    result["depth"] = int(depth_match.group(1))
                else:
                    result["depth"] = 15
        
        return result
    
    def process_video(self, video_path, output_path):
        """Process video by analyzing first frame and applying to all frames"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None, None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        ret, first_frame = cap.read()
        if not ret:
            print(f"Error: Could not read first frame from {video_path}")
            cap.release()
            return None, None
        
        if first_frame.shape[0] > 20:
            first_frame = first_frame[20:, :]
        
        first_frame_pil = Image.fromarray(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))

        try:
            result = self._analyze_image(first_frame_pil, first_frame)
        except RuntimeError as e:
            # If CUDA error, free cache and try again
            if "CUDA" in str(e):
                print("CUDA error detected, trying to recover...")
                # Clear CUDA cache
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                gc.collect()
            
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
        
        min_width = MIN_WIDTH
        min_height = MIN_HEIGHT
        
        if processed_first is not None and processed_first.size > 0:
            output_height, output_width = processed_first.shape[:2]
            
            output_width = max(output_width, min_width)
            output_height = max(output_height, min_height)
        else:
            output_height, output_width = min_height, min_width
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MPEG-4 codec
        out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
        
        if not out.isOpened():
            print(f"Error: Could not create output video file at {output_path}")
            cap.release()
            return None, None
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processed_frames = 0
        
        for _ in tqdm(range(frame_count), desc=f"Processing video {os.path.basename(video_path)}"):
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame.shape[0] > 20:
                frame = frame[20:, :]
            
            processed = self._apply_cached_processing(frame, result)
            
            if processed is None or processed.size == 0:
                continue
            
            current_h, current_w = processed.shape[:2]
            if current_h < min_height or current_w < min_width or current_h != output_height or current_w != output_width:
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
                
                processed = cv2.resize(processed, (new_w, new_h))
                
                if new_h != output_height or new_w != output_width:
                    canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
                    y_offset = (output_height - new_h) // 2
                    x_offset = (output_width - new_w) // 2
                    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = processed
                    processed = canvas
            
            if len(processed.shape) == 3 and processed.shape[2] == 3:
                if processed.dtype != np.uint8:
                    processed = processed.astype(np.uint8)
                processed_bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
            else:
                processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            
            out.write(processed_bgr)
            processed_frames += 1
        
        # Clean up
        cap.release()
        out.release()
        
        print(f"Processed {processed_frames} frames for video {os.path.basename(video_path)}")
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0 and processed_frames > 0:
            return site_code, depth
        else:
            print(f"Warning: Output video file may be invalid: {output_path}")
            if os.path.exists(output_path):
                os.remove(output_path)
            return None, None
    
    def remove_tag(self, image):
        """Remove B-tag using color-based masking and inpainting"""
        if image is None or image.size == 0:
            return image, None
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        lower_blue_hsv = np.array([100, 100, 50])
        upper_blue_hsv = np.array([140, 255, 255])
        lower_orange_hsv = np.array([5, 50, 50])
        upper_orange_hsv = np.array([25, 255, 255])
        
        mask_blue_hsv = cv2.inRange(image_hsv, lower_blue_hsv, upper_blue_hsv)
        mask_orange_hsv = cv2.inRange(image_hsv, lower_orange_hsv, upper_orange_hsv)
        mask_combined = cv2.bitwise_or(mask_blue_hsv, mask_orange_hsv)
        
        kernel = np.ones((5, 5), np.uint8)
        mask_dilated = cv2.dilate(mask_combined, kernel, iterations=2)
        
        inpainted_image = cv2.inpaint(image_rgb, mask_dilated, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        
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
        if image is None or image.size == 0:
            return image
            
        height, width = image.shape[:2]
        
        ruler_width = int(width * 0.05)
        
        mask = np.ones((height, width), dtype=np.uint8)
        mask[:, width-ruler_width:] = 0
        
        result = image.copy()
        if len(image.shape) == 3:
            for c in range(image.shape[2]):
                result[:, :, c] = image[:, :, c] * mask
        else:
            result = image * mask
        
        return result
    
    def extract_roi(self, image):
        """Extract ROI with less aggressive cropping to preserve more ultrasound data"""
        if image is None or image.size == 0:
            return {"cropped_image": image, "binary_image": None, "roi_coords": (0, 0, 0, 0)}
                
        if len(image.shape) > 2 and image.shape[2] == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image
        
        height, width = gray_image.shape[:2]
        
     
        bottom_cutoff = int(height * 0.8)
        
        mask = np.ones_like(gray_image)
        mask[bottom_cutoff:, :] = 0
        
        if len(image.shape) == 3:
            masked_image = image.copy()
            for c in range(3):
                masked_image[:, :, c] = image[:, :, c] * mask
        else:
            masked_image = image * mask
        
        _, img_binary = cv2.threshold(gray_image * mask, 25, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((7, 7), np.uint8)
        img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel)
        
        non_zero_points = cv2.findNonZero(img_binary)
        
        min_width = width - int(width * 0.1)  # Only crop 10% of width at most
        min_height = height - int(height * 0.3)  # Only crop 10% of height at most
        
        if non_zero_points is not None and len(non_zero_points) > 0:
            x, y, w, h = cv2.boundingRect(non_zero_points)
            
            padding_x = int(w * 0.05)  # 5% horizontal padding
            padding_y = int(h * 0.05)  # 5% vertical padding
            
            x = max(0, x - padding_x)
            y = max(0, y - padding_y)
            w = min(width - x, w + 2*padding_x)
            h = min(bottom_cutoff - y, h + padding_y)
            
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
            
            margin_x = int(width * 0.05)  # 5% margin from sides
            margin_y = int(height * 0.05)  # 5% margin from top/bottom
            
            x = margin_x
            y = margin_y
            w = width - 2*margin_x
            h = bottom_cutoff - margin_y - 5  # Leave 5 pixels above bottom cutoff
            
            roi = masked_image[y:y+h, x:x+w]
            roi_coords = (x, y, x+w, y+h)
        
        return {
            "cropped_image": roi,
            "binary_image": img_binary,
            "roi_coords": roi_coords
        }




def process_directory(input_folder, output_folder, processed_folder = '/gpfs/gibbs/project/hartley/tjb76/artstuff_OPTIMIZEDWOOOO/Data/LUSSTER-SouthAfrica/Videos/Processed'):
    """Process a directory containing patient folders with ultrasound images and videos"""
    processor = UltrasoundProcessor()
    
    image_output_path = os.path.join(output_folder, 'images')
    video_output_path = os.path.join(output_folder, 'videos')
    os.makedirs(image_output_path, exist_ok=True)
    os.makedirs(video_output_path, exist_ok=True)
    
    csv_file_path = os.path.join(output_folder, 'processed_files.csv')
    
    processed_patients = set()
    csv_mode = 'w'  # Default to write mode
    write_header = True  # Default to writing header
    
    if os.path.exists(csv_file_path) and os.path.getsize(csv_file_path) > 0:
        try:
            with open(csv_file_path, mode='r') as existing_file:
                reader = csv.reader(existing_file)
                try:
                    next(reader)  
                    for row in reader:
                        if row and len(row) > 0:
                            processed_patients.add(row[0])  # Add patient ID to set
                    print(f"Found {len(processed_patients)} already processed patients")
                    
                    csv_mode = 'a'
                    write_header = False
                except StopIteration:
                    print("CSV file exists but is empty or corrupted. Creating new file.")
                    csv_mode = 'w'
                    write_header = True
        except Exception as e:
            print(f"Error reading existing CSV file: {e}. Creating new file.")
            csv_mode = 'w'
            write_header = True
    
    print(f"Opening CSV file in mode: {csv_mode}, with write_header={write_header}")
    
    with open(csv_file_path, mode=csv_mode, newline='', buffering=1) as csv_file:
        writer = csv.writer(csv_file)
        
        if write_header:
            writer.writerow(['Patient ID', 'Site', 'Depth', 'Count', 'New File Name', 'Original File Name', 'Type'])
            csv_file.flush()  
            print("Header written to CSV")
        
        patient_dirs = [d for d in os.listdir(input_folder) if not d.startswith('.')]
        for patient_id in tqdm(patient_dirs, desc="Processing patients"):
            patient_folder = os.path.join(input_folder, patient_id)
            if not os.path.isdir(patient_folder):
                continue
                
            final_patient_id = patient_id.split(' ')[0]
            
            
            print(f"Processing patient: {final_patient_id}")
            
            
            site_depth_type_counts = {}
            patient_files_processed = 0
            
            for file_name in os.listdir(patient_folder):
                file_path = os.path.join(patient_folder, file_name)
                
                if file_name.lower().endswith('.png'):
                    if '(1).png' in file_name or '(2).png' in file_name:
                        continue
                    try:
                        processed_image, site, depth = processor.process_image(file_path)
                        
                        
                        # valid_sites = list(SITE_MAPPING.keys())
                        # if site not in valid_sites and site != "UNKNOWN":
                        #     # Try to find closest match
                        #     if site == "QLSD":
                        #         site = "QSLD"
                        #     else:
                        #         site = "Unknown" + site
                           
                
                
                        if processed_image is not None and site is not None:
                            site_depth_type_key = f"{site}_{depth}_png"
                            if site_depth_type_key not in site_depth_type_counts:
                                site_depth_type_counts[site_depth_type_key] = 1
                            count = site_depth_type_counts[site_depth_type_key]
                            site_depth_type_counts[site_depth_type_key] += 1
                            
                            # Save processed image
                            new_file_name = f"{final_patient_id}_{site}_{depth}_{count}.png"
                            save_path = os.path.join(image_output_path, new_file_name)
                            
                            # Convert to BGR for OpenCV saving
                            if len(processed_image.shape) == 3 and processed_image.shape[2] == 3:
                                save_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
                            else:
                                save_image = processed_image
                                
                            cv2.imwrite(save_path, save_image)
                            
                            writer.writerow([final_patient_id, site, depth, count, new_file_name, file_name, "image"])
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
                        
                        
                        # valid_sites = list(SITE_MAPPING.keys())
                        # if site not in valid_sites and site != "UNKNOWN":
                        #     if site == "QLSD":
                        #         site = "QSLD"
                        #     else:
                        #         site = "Unknown" + site
                           
                        
                        if site and depth:
                            # Handle counts - separate for videos
                            site_depth_type_key = f"{site}_{depth}_mp4"
                            if site_depth_type_key not in site_depth_type_counts:
                                site_depth_type_counts[site_depth_type_key] = 1
                            count = site_depth_type_counts[site_depth_type_key]
                            site_depth_type_counts[site_depth_type_key] += 1
                            
                            new_file_name = f"{final_patient_id}_{site}_{depth}_{count}.mp4"
                            new_output_path = os.path.join(video_output_path, new_file_name)
                            
                            if os.path.exists(output_path):
                                os.rename(output_path, new_output_path)
                                
                                writer.writerow([final_patient_id, site, depth, count, new_file_name, file_name, "video"])
                                csv_file.flush()  # Force write to disk
                                patient_files_processed += 1
                                
                                print(f"Processed video: {new_file_name}")
                            else:
                                print(f"Warning: Temp video file not found: {output_path}")
                            
                    except Exception as e:
                        print(f"Error processing video {file_name}: {e}")
                        temp_path = os.path.join(video_output_path, f"{final_patient_id}_temp.mp4")
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
            
            print(f"Completed patient {final_patient_id}, processed {patient_files_processed} files")
            processed_patients.add(final_patient_id)

            if processed_folder and os.path.exists(patient_folder):
                dest_path = os.path.join(processed_folder, patient_id)
                print(f"Moving {patient_id} to processed folder: {dest_path}")
                try:
                    # Create the processed folder if it doesn't exist
                    os.makedirs(processed_folder, exist_ok=True)
                    
                    # Move the folder
                    shutil.move(patient_folder, dest_path)
                    print(f"Successfully moved {patient_id} to processed folder")
                except Exception as e:
                    print(f"Error moving folder {patient_id}: {e}")
    
            
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


#CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 python3 /gpfs/gibbs/project/hartley/tjb76/artstuff_OPTIMIZEDWOOOO/Data/Cleaning/process_directories_new.py --input '/gpfs/gibbs/project/hartley/tjb76/artstuff_OPTIMIZEDWOOOO/Data/LUSSTER-SouthAfrica/Videos/Uncleaned' --output '/gpfs/gibbs/project/hartley/tjb76/artstuff_OPTIMIZEDWOOOO/Data/LUSSTER-SouthAfrica/cleaned'




