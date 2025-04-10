import os
import json
import logging
import re
from typing import Dict, List, Optional, Tuple
import numpy as np
from inference_sdk import InferenceHTTPClient
from PIL import Image
import cv2
import random
import pytesseract

class OCRProcessor:
    def __init__(self, api_key: str):
        """Initialize the OCR processor with Roboflow API key."""
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)
        self.use_mock = False
        
        try:
            # Initialize the inference client
            self.client = InferenceHTTPClient(
                api_url="https://serverless.roboflow.com",
                api_key=api_key
            )
            
            # Try to get model IDs from Flask config if running in a Flask app
            try:
                from flask import current_app
                # Define model IDs from config if available
                self.wall_model_id = current_app.config.get('ROBOFLOW_WALL_MODEL_ID', "wall-detection-xi9ox/2")
                self.floor_model_id = current_app.config.get('ROBOFLOW_FLOOR_MODEL_ID', "floortest3.1/2")
                self.space_model_id = current_app.config.get('ROBOFLOW_SPACE_MODEL_ID', "builderformer-4/2")
                
                # Get OCR model IDs from config
                self.ocr_model_1_id = current_app.config.get('ROBOFLOW_OCR_MODEL_1_ID', "2d-floorplan-analysis/7")
                self.ocr_model_2_id = current_app.config.get('ROBOFLOW_OCR_MODEL_2_ID', "your-second-ocr-model-id/version")
                self.ocr_model_3_id = current_app.config.get('ROBOFLOW_OCR_MODEL_3_ID', "your-third-ocr-model-id/version")
                
                self.logger.info(f"Using model IDs from config: {self.wall_model_id}, {self.floor_model_id}, {self.space_model_id}")
                self.logger.info(f"Using OCR model IDs: {self.ocr_model_1_id}, {self.ocr_model_2_id}, {self.ocr_model_3_id}")
            except (ImportError, RuntimeError):
                # Not running in Flask context, use default values
                self.wall_model_id = "wall-detection-xi9ox/2"
                self.floor_model_id = "floortest3.1/2"
                self.space_model_id = "builderformer-4/2"
                
                # Default OCR model IDs
                self.ocr_model_1_id = "2d-floorplan-analysis/7"
                self.ocr_model_2_id = "your-second-ocr-model-id/version"
                self.ocr_model_3_id = "your-third-ocr-model-id/version"
                
        except Exception as e:
            self.logger.error(f"Error initializing Roboflow client: {str(e)}")
            self.use_mock = True
        
        # Define space classes from BuilderFormer dataset
        self.space_classes = [
            'space_balconi', 'space_bedroom', 'space_dressroom', 'space_front',
            'space_kitchen', 'space_living_room', 'space_staircase', 'space_toilet',
            'space_elevator', 'space_elevator_hall', 'space_multipurpose_space',
            'space_other', 'space_outdoor_room'
        ]
        
        # Setup tesseract and verify installation
        self.tesseract_available = False
        try:
            # Check if tesseract is installed and accessible
            version = pytesseract.get_tesseract_version()
            self.tesseract_available = True
            self.logger.info(f"Tesseract OCR detected, version: {version}")
            
            # Verify command is accessible (important for Windows)
            if os.name == 'nt':  # Windows
                # Check if tesseract_cmd is properly set
                cmd = pytesseract.pytesseract.tesseract_cmd
                if not os.path.exists(cmd):
                    default_paths = [
                        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
                    ]
                    for path in default_paths:
                        if os.path.exists(path):
                            pytesseract.pytesseract.tesseract_cmd = path
                            self.logger.info(f"Set tesseract path to: {path}")
                            break
                    else:
                        self.logger.warning("Tesseract command not found in default locations, OCR will be limited.")
                else:
                    self.logger.info(f"Using tesseract command at: {cmd}")
            
        except Exception as e:
            self.logger.warning(f"Tesseract not properly configured: {str(e)}")
            self.logger.warning("Text extraction and dimension detection will be limited.")
            
            # Try to provide helpful information about installation
            if os.name == 'nt':  # Windows
                self.logger.info("On Windows, make sure Tesseract is installed and added to PATH.")
                self.logger.info("Or set pytesseract.pytesseract.tesseract_cmd to the full path.")
            else:  # Linux/Mac
                self.logger.info("On Linux, install tesseract-ocr package. On Mac, use brew install tesseract.")
        
        # Material types and properties
        self.material_types = {
            'wall': {
                'default_thickness': 0.15,  # meters
                'materials': {
                    'concrete': {'density': 2400},  # kg/m³
                    'brick': {'density': 1800},
                    'wood': {'density': 600},
                    'steel': {'density': 7850},
                    'glass': {'density': 2500}
                }
            },
            'floor': {
                'default_thickness': 0.2,  # meters
                'materials': {
                    'concrete': {'density': 2400},
                    'wood': {'density': 600},
                    'tile': {'density': 2100},
                    'carpet': {'density': 300}
                }
            },
            'ceiling': {
                'default_thickness': 0.1,  # meters
                'materials': {
                    'drywall': {'density': 800},
                    'acoustic': {'density': 300},
                    'plaster': {'density': 1000}
                }
            }
        }
            
    # Image preprocessing functions for text extraction
    def _get_grayscale(self, image):
        """Convert image to grayscale"""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def _remove_noise(self, image):
        """Remove noise using median blur"""
        return cv2.medianBlur(image, 5)
    
    def _thresholding(self, image):
        """Apply thresholding"""
        return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    def _dilate(self, image):
        """Dilate image to connect text regions"""
        kernel = np.ones((5, 5), np.uint8)
        return cv2.dilate(image, kernel, iterations=1)
    
    def _erode(self, image):
        """Erode image to thin text"""
        kernel = np.ones((5, 5), np.uint8)
        return cv2.erode(image, kernel, iterations=1)
    
    def _opening(self, image):
        """Opening operation - erosion followed by dilation"""
        kernel = np.ones((5, 5), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    
    def _canny(self, image):
        """Apply Canny edge detection"""
        return cv2.Canny(image, 100, 200)
    
    def _deskew(self, image):
        """Deskew the image to straighten text"""
        try:
            coords = np.column_stack(np.where(image > 0))
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return rotated
        except Exception as e:
            self.logger.warning(f"Error during deskew: {str(e)}")
            return image
            
    def _preprocess_image_for_text(self, image_path: str):
        """
        Preprocess image for text extraction with multiple methods optimized for floor plans
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dict of preprocessed images using different methods
        """
        self.logger.info(f"Preprocessing image: {image_path}")
        
        # Read image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        # Standard preprocessing methods
        gray = self._get_grayscale(image)
        noise_removed = self._remove_noise(gray)
        thresh = self._thresholding(noise_removed)
        
        # Floor plan specific processing
        # 1. Adaptive thresholding for better text recognition in varying lighting
        adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY, 11, 2)
        
        # 2. Image inversion - floor plans often have dark text on light background
        inverted = cv2.bitwise_not(gray)
        
        # 3. Edge enhancement for dimensions that appear along lines
        canny = self._canny(gray)
        
        # 4. Enhanced dimension text detection - use morphological operations
        # First sharpen the image to make text clearer
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        
        # Then apply binary thresholding with higher threshold to isolate text
        _, enhanced_thresh = cv2.threshold(sharpened, 150, 255, cv2.THRESH_BINARY)
        
        # 5. Dilated image to connect broken text
        dilated = self._dilate(thresh)
        
        # 6. Opening to remove small noise
        opening = self._opening(gray)
        
        # 7. Specific preprocessing for floor plan dimensions, with adjusted contrast
        # This helps with dimensions often placed alongside lines in drawings
        alpha = 1.5  # Contrast control
        beta = 10    # Brightness control
        contrast_adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
        _, contrast_thresh = cv2.threshold(contrast_adjusted, 150, 255, cv2.THRESH_BINARY)
        
        # 8. Color filtering for dimension text that may be in specific colors
        # Convert to HSV for better color isolation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define range for red color (often used for dimensions)
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        red_mask1 = cv2.inRange(hsv, lower_red, upper_red)
        
        lower_red = np.array([170, 50, 50])
        upper_red = np.array([180, 255, 255])
        red_mask2 = cv2.inRange(hsv, lower_red, upper_red)
        
        # Combine masks and apply to get red text
        red_mask = red_mask1 + red_mask2
        red_text = cv2.bitwise_and(gray, gray, mask=red_mask)
        
        # Define range for blue color (sometimes used for dimensions)
        lower_blue = np.array([110, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        blue_text = cv2.bitwise_and(gray, gray, mask=blue_mask)
        
        # Combined color highlighted image
        color_highlighted = cv2.add(red_text, blue_text)
        
        self.logger.info("Completed image preprocessing with multiple methods")
        
        return {
            'original': image,
            'gray': gray,
            'noise_removed': noise_removed,
            'thresh': thresh,
            'opening': opening,
            'canny': canny,
            'adaptive_thresh': adaptive_thresh,
            'inverted': inverted,
            'dilated': dilated,
            'enhanced': enhanced_thresh,
            'contrast_adjusted': contrast_thresh,
            'color_highlighted': color_highlighted
        }
    
    def _detect_dimension_text(self, image, mode='detail'):
        """
        Detect text with dimension information and extract bounding boxes
        
        Args:
            image: The preprocessed image for OCR
            mode: 'detail' for detailed bounding boxes around each dimension, 
                  'paragraph' for paragraph-level boxes
        
        Returns:
            A list of dimension texts with bounding boxes
        """
        self.logger.info(f"Starting dimension text detection in mode: {mode}")
        
        # Configure Tesseract with optimal settings for floor plans/technical drawings
        # Floor plans typically have clear, well-separated text that's either horizontal or vertical
        if mode == 'detail':
            # --psm 11: Sparse text with OSD (Orientation and Script Detection)
            # --oem 3: Default, based on LSTM neural nets
            config = '--psm 11 --oem 3 -c preserve_interword_spaces=1 -c tessedit_char_whitelist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.-_\'\"xX×m²sqftincm" -c textord_min_linesize=2.5'
            self.logger.info(f"Using detail mode for OCR text detection with config: {config}")
        else:
            # For paragraph mode, use PSM 3 to treat the document as a single column of text
            config = '--psm 3 --oem 3 -c preserve_interword_spaces=1 -c tessedit_char_whitelist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.-_\'\"xX×m²sqftincm"'
            self.logger.info(f"Using paragraph mode for OCR text detection with config: {config}")
        
        # Get detailed OCR data including bounding boxes
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config=config)
        
        self.logger.info(f"OCR detected {len(data['text'])} text elements")
        
        # Log the raw text data for debugging
        clean_text = [t for t in data['text'] if t.strip()]
        self.logger.info(f"Raw detected text elements: {clean_text[:20]}")
        
        # Refined dimension patterns specifically for floor plans
        dimension_patterns = [
            # Common floor plan dimension formats (more specific to reduce false positives)
            r'^(?:\d{1,4}(?:\.\d{1,2})?)(?:\s*[xX×]\s*\d{1,4}(?:\.\d{1,2})?)$',  # 100x200 (exactly this format)
            r'^(?:\d{1,4}(?:\.\d{1,2})?)\s*(?:mm|cm|m|ft|in)$',  # 100mm, 200cm (exactly this format)
            r'^(?:\d{1,4}(?:\.\d{1,2})?\'(?:\-|\s)?\d{1,2}(?:\.\d{1,2})?\")$',  # 10'-6" (exactly this format)
            r'^(?:\d{1,4}(?:\.\d{1,2})?\')$',  # 10' (exactly this format)
            
            # Area measurements (common in floor plans)
            r'^(?:\d{1,5}(?:\.\d{1,2})?)\s*(?:sq\.ft|sq\.m|sqft|sqm|m²)$',  # Area with units (exact format)
            r'^(?:area|a)[\s\:]+\d{1,5}(?:\.\d{1,2})?\s*(?:sq\.ft|sq\.m|sqft|sqm|m²)$',  # Area labels
            
            # Room dimensions (common in residential floor plans)
            r'^(?:\d{1,3}(?:\.\d{1,2})?\'(?:\-|\s)?\d{1,2}(?:\.\d{1,2})?\"\s*[xX×]\s*\d{1,3}(?:\.\d{1,2})?\'(?:\-|\s)?\d{1,2}(?:\.\d{1,2})?\")$',  # 10'-6" x 12'-0" format
            r'^(?:\d{1,3}\'\s*[xX×]\s*\d{1,3}\')$'  # 10' x 12' format
        ]
        
        # More relaxed patterns for additional dimension checks
        additional_dimension_patterns = [
            r'\d+\s*[xX×]\s*\d+',                     # Like 100x200
            r'\d+\'(?:\d+(?:\.\d+)?)?\"?',            # Like 10'6" or 10'
            r'\d+(?:\.\d+)?[\'″"]',                   # Like 10.5' (just feet with decimal)
            r'(?:ø|\(ø\)|\(d\)|dia|diameter)[\s\:]*\d+', # Diameter dimensions
            r'(?<!\w)(?:\d+\')?(?:\d+(?:\.\d+)?\")?', # Architectural format like 10'6"
            r'(?<!\w)R\s*\d+(?:\.\d+)?',              # Radius indicators like "R20" or "R 5.5"
            r'(?<!\w)Ø\s*\d+(?:\.\d+)?'               # Diameter indicators like "Ø20" 
        ]
        
        # Material keywords to identify material specifications
        material_keywords = [
            'concrete', 'steel', 'wood', 'glass', 'brick', 'metal',
            'drywall', 'carpet', 'tile', 'acoustic', 'plaster',
            'granite', 'marble', 'stone', 'ceramic', 'vinyl'
        ]
        
        dimension_texts = []
        found_dimensions = 0
        found_materials = 0
        
        # First pass: find exact dimension matches with high confidence
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            if not text:
                continue
            
            # Higher confidence threshold for primary dimension detection
            confidence = int(data['conf'][i])
            if confidence < 50:
                continue
            
            # Check for exact matches with primary dimension patterns
            is_dimension = False
            matched_pattern = None
            
            for pattern in dimension_patterns:
                if re.match(pattern, text):
                    is_dimension = True
                    matched_pattern = pattern
                    found_dimensions += 1
                    break
                    
            # Process exact dimension matches
            if is_dimension:
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                self.logger.info(f"Found primary dimension text: '{text}' matching pattern: '{matched_pattern}' at position {x},{y}")
                
                # Add to results
                dimension_texts.append({
                    'text': text,
                    'bbox': {'x': x, 'y': y, 'width': w, 'height': h},
                    'confidence': confidence,
                    'type': 'dimension',
                    'match_type': 'primary'
                })
        
        # Second pass: find additional dimension matches with relaxed patterns
        # but only if we didn't find many primary dimensions
        if found_dimensions < 3:
            self.logger.info(f"Found only {found_dimensions} primary dimensions, searching for secondary dimensions")
            
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                if not text:
                    continue
                    
                # Lower confidence threshold for secondary dimension detection
                confidence = int(data['conf'][i])
                if confidence < 40:
                    continue
                    
                # Skip text already identified as a primary dimension
                already_identified = False
                for dim in dimension_texts:
                    if dim['text'] == text:
                        already_identified = True
                        break
                        
                if already_identified:
                    continue
                    
                # Check with additional patterns
                is_dimension = False
                matched_pattern = None
                
                for pattern in additional_dimension_patterns:
                    if re.search(pattern, text):
                        is_dimension = True
                        matched_pattern = pattern
                        found_dimensions += 1
                        break
                        
                # Check for material keywords
                is_material = False
                matched_material = None
                for material in material_keywords:
                    if material in text.lower():
                        is_material = True
                        matched_material = material
                        found_materials += 1
                        break
                        
                if is_dimension or is_material:
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    
                    if is_dimension:
                        self.logger.info(f"Found secondary dimension text: '{text}' matching pattern: '{matched_pattern}' at position {x},{y}")
                    if is_material:
                        self.logger.info(f"Found material text: '{text}' containing material: '{matched_material}' at position {x},{y}")
                    
                    # Get the surrounding context (more words around this one)
                    if mode == 'paragraph':
                        # Get paragraph or block ID to find related text
                        paragraph_id = data['block_num'][i]
                        paragraph_indices = [j for j in range(len(data['block_num'])) 
                                          if data['block_num'][j] == paragraph_id]
                        
                        # Combine all text in this paragraph
                        paragraph_text = ' '.join(data['text'][j] for j in paragraph_indices if data['text'][j])
                        
                        # Create bounding box around the entire paragraph
                        min_x = min(data['left'][j] for j in paragraph_indices)
                        min_y = min(data['top'][j] for j in paragraph_indices)
                        max_x = max(data['left'][j] + data['width'][j] for j in paragraph_indices)
                        max_y = max(data['top'][j] + data['height'][j] for j in paragraph_indices)
                        
                        dimension_texts.append({
                            'text': paragraph_text,
                            'bbox': {'x': min_x, 'y': min_y, 'width': max_x - min_x, 'height': max_y - min_y},
                            'confidence': confidence,
                            'type': 'dimension' if is_dimension else 'material',
                            'match_type': 'secondary'
                        })
                    else:
                        # Just use the individual word/text and its bounding box
                        dimension_texts.append({
                            'text': text,
                            'bbox': {'x': x, 'y': y, 'width': w, 'height': h},
                            'confidence': confidence,
                            'type': 'dimension' if is_dimension else 'material',
                            'match_type': 'secondary'
                        })
        
        # Filter out duplicate or overlapping boxes
        filtered_texts = []
        for i, box1 in enumerate(dimension_texts):
            # Skip if already marked for removal
            if box1.get('remove', False):
                continue
            
            bbox1 = box1['bbox']
            center1_x = bbox1['x'] + bbox1['width'] / 2
            center1_y = bbox1['y'] + bbox1['height'] / 2
            
            # Check against all other boxes
            for j, box2 in enumerate(dimension_texts):
                if i == j or box2.get('remove', False):
                    continue
                
                bbox2 = box2['bbox']
                
                # Check for significant overlap or containment
                if (bbox1['x'] <= bbox2['x'] <= bbox1['x'] + bbox1['width'] and
                    bbox1['y'] <= bbox2['y'] <= bbox1['y'] + bbox1['height'] and
                    bbox1['x'] <= bbox2['x'] + bbox2['width'] <= bbox1['x'] + bbox1['width'] and
                    bbox1['y'] <= bbox2['y'] + bbox2['height'] <= bbox1['y'] + bbox1['height']):
                    
                    # The second box is contained in the first box
                    if box1.get('match_type') == 'primary' and box2.get('match_type') == 'secondary':
                        # Keep the primary match, mark secondary for removal
                        box2['remove'] = True
                    elif box1.get('match_type') == 'secondary' and box2.get('match_type') == 'primary':
                        # Keep the primary match, mark this box for removal
                        box1['remove'] = True
                        break
                    elif box1['confidence'] >= box2['confidence']:
                        # Keep the higher confidence box
                        box2['remove'] = True
                    else:
                        # The other box has higher confidence
                        box1['remove'] = True
                        break
            
            if not box1.get('remove', False):
                filtered_texts.append(box1)
        
        self.logger.info(f"Found {len(filtered_texts)} unique dimension/material texts after filtering")
        
        return filtered_texts
        
    def extract_text(self, image_path: str, filter_dimensions=True, detect_bboxes=True) -> Dict:
        """
        Extract text from an image, with optional filtering for dimensions/material info.
        
        Args:
            image_path: Path to the image file
            filter_dimensions: Whether to filter for dimension/material text only
            detect_bboxes: Whether to detect bounding boxes for dimension text
            
        Returns:
            Dictionary with extracted text and optional bounding boxes
        """
        try:
            self.logger.info(f"Starting text extraction from {image_path}")
            self.logger.info(f"Filter dimensions: {filter_dimensions}, Detect bboxes: {detect_bboxes}")
            
            # Preprocess image with multiple methods
            processed_images = self._preprocess_image_for_text(image_path)
            
            # Try different preprocessing methods to find the best one for dimensions
            results = {}
            best_text = ""
            best_method = None
            best_dimension_count = 0
            
            # Dimension patterns specifically for floor plans
            dimension_patterns = [
                r'\d+\s*[xX×]\s*\d+',                     # Like 100x200
                r'\d+\'(?:\d+(?:\.\d+)?)?\"?',            # Like 10'6" or 10'
                r'\d+(?:\.\d+)?[\'″"]',                   # Like 10.5' (just feet with decimal)
                r'\d+(?:\.\d+)?\s*(?:mm|cm|m|ft|in|m²|m2|sqm|sq\.ft|sq\.m|sqft)',  # Measurements with units
            ]
            
            # When looking for floor plan dimensions, prioritize methods that enhance line drawings
            # For detailed technical drawings, prioritize these preprocessing methods
            priority_methods = ['adaptive_thresh', 'contrast_adjusted', 'enhanced', 'color_highlighted', 'inverted']
            
            self.logger.info(f"Testing preprocessing methods with OCR to find the best for dimensions")
            
            # For each preprocessing method, check how many dimension patterns are found
            for method, image in processed_images.items():
                if method == 'original':
                    continue
                    
                # Configure Tesseract for technical drawings/floor plans
                # PSM 11 (sparse text) often works better for dimensions in floor plans
                config = '--psm 11 --oem 3 -c preserve_interword_spaces=1 -c tessedit_char_whitelist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.-_\'\"xX×m²sqftincm"'
                
                # Extract text with the current preprocessing method
                text = pytesseract.image_to_string(image, config=config)
                self.logger.info(f"Method {method} extracted {len(text.strip())} characters")
                
                # Count how many dimension patterns are found in this text
                dimension_count = 0
                for pattern in dimension_patterns:
                    dimension_count += len(re.findall(pattern, text))
                    
                self.logger.info(f"Method {method} found {dimension_count} potential dimensions")
                
                # Choose the method with the most dimensions, or prioritize certain methods
                if dimension_count > best_dimension_count:
                    best_dimension_count = dimension_count
                    best_text = text
                    best_method = method
                elif dimension_count == best_dimension_count:
                    # If tied, prefer methods that are typically better for floor plans
                    if (method in priority_methods and 
                        (best_method not in priority_methods or 
                         priority_methods.index(method) < priority_methods.index(best_method))):
                        best_text = text
                        best_method = method
                    
            # If no dimensions found, try again with a different PSM mode
            if best_dimension_count == 0:
                self.logger.info("No dimensions found with PSM 11, trying with PSM 6")
                for method, image in processed_images.items():
                    if method == 'original' or method not in priority_methods:
                        continue
                        
                    # Try with PSM 6 (uniform block of text)
                    config = '--psm 6 --oem 3 -c preserve_interword_spaces=1'
                    
                    text = pytesseract.image_to_string(image, config=config)
                    
                    # Count dimensions again
                    dimension_count = 0
                    for pattern in dimension_patterns:
                        dimension_count += len(re.findall(pattern, text))
                        
                    self.logger.info(f"PSM 6 with method {method} found {dimension_count} potential dimensions")
                    
                    if dimension_count > best_dimension_count:
                        best_dimension_count = dimension_count
                        best_text = text
                        best_method = method
            
            self.logger.info(f"Best OCR preprocessing method: {best_method} with {best_dimension_count} dimensions")
            
            # Filter out dimension and material text if requested
            if filter_dimensions:
                # Use regular expressions to find dimension text
                dimension_patterns = [
                    r'\d+\s*[xX×]\s*\d+',  # Like 100x200
                    r'\d+\s*[xX×]\s*\d+\s*[xX×]\s*\d+',  # Like 100x200x300
                    r'\d+\s*(?:mm|cm|m|ft|in)(?:\s*[xX×]\s*\d+\s*(?:mm|cm|m|ft|in))+',  # Like 100mm x 200mm
                    r'\d+(?:\.\d+)?\s*(?:mm|cm|m|ft|in|m²|m2|sqm|sq\.ft|sq\.m|sqft)',  # Measurements with units
                    r'\d+(?:\.\d+)?\'(?:\-|\s)\d+(?:\.\d+)?\"',  # Like 10'-6" (feet-inches)
                    r'\d+\'(?:\d+(?:\.\d+)?)?\"?',  # Like 10'6" or 10'
                ]
                
                # Find matches for dimension patterns
                filtered_lines = []
                for line in best_text.split('\n'):
                    if any(re.search(pattern, line) for pattern in dimension_patterns):
                        filtered_lines.append(line)
                    elif any(material in line.lower() for material in 
                           ['concrete', 'steel', 'wood', 'glass', 'brick', 'metal', 
                            'drywall', 'carpet', 'tile', 'acoustic', 'plaster']):
                        filtered_lines.append(line)
                
                filtered_text = '\n'.join(filtered_lines)
                self.logger.info(f"Filtered text contains {len(filtered_lines)} lines with dimensions/materials")
                
                results = {
                    'text': best_text,
                    'filtered_text': filtered_text,
                    'preprocessing': best_method,
                    'full_text': best_text
                }
            else:
                results = {
                    'text': best_text,
                    'preprocessing': best_method
                }
            
            # Detect bounding boxes for dimension text if requested
            if detect_bboxes:
                self.logger.info("Detecting bounding boxes for dimension text")
                
                # Try multiple preprocessing methods for dimension detection
                all_dimension_boxes = []
                
                # Primary image methods for dimension detection
                primary_methods = ['adaptive_thresh', 'contrast_adjusted', 'enhanced', 'color_highlighted']
                image_methods_to_try = []
                
                # Always include the best method
                if best_method:
                    image_methods_to_try.append(best_method)
                    
                # Add other prioritized methods
                for method in primary_methods:
                    if method in processed_images and method != best_method:
                        image_methods_to_try.append(method)
                        
                self.logger.info(f"Trying {len(image_methods_to_try)} preprocessing methods for dimension box detection")
                
                for method in image_methods_to_try:
                    self.logger.info(f"Detecting dimension boxes with method: {method}")
                    # Try detail mode first
                    dimension_boxes = self._detect_dimension_text(processed_images[method], mode='detail')
                    
                    if not dimension_boxes:
                        # Fall back to paragraph mode if no detections
                        self.logger.info(f"No dimension boxes found with detail mode, trying paragraph mode")
                        dimension_boxes = self._detect_dimension_text(processed_images[method], mode='paragraph')
                    
                    if dimension_boxes:
                        self.logger.info(f"Found {len(dimension_boxes)} dimension boxes with method {method}")
                        
                        # Add method info to each box for debugging
                        for box in dimension_boxes:
                            box['detection_method'] = method
                            
                        all_dimension_boxes.extend(dimension_boxes)
                        
                        # If we found more than a few dimensions, we can stop
                        if len(all_dimension_boxes) >= 5 and len(image_methods_to_try) > 1:
                            self.logger.info(f"Found sufficient dimension boxes ({len(all_dimension_boxes)}), stopping search")
                            break
                
                # Remove duplicates by comparing text and position
                unique_boxes = []
                for box in all_dimension_boxes:
                    # Check if this box is similar to any already included box
                    is_duplicate = False
                    for existing_box in unique_boxes:
                        # Same text or nearly same position indicates duplicate
                        if (box['text'] == existing_box['text'] or 
                            (abs(box['bbox']['x'] - existing_box['bbox']['x']) < 15 and 
                             abs(box['bbox']['y'] - existing_box['bbox']['y']) < 15)):
                             
                            # Keep the one with higher confidence
                            if box.get('confidence', 0) > existing_box.get('confidence', 0):
                                existing_box.update(box)
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        unique_boxes.append(box)
                
                # Filter out any boxes with very low confidence
                valid_boxes = [box for box in unique_boxes if box.get('confidence', 0) >= 40]
                
                self.logger.info(f"Found {len(valid_boxes)} valid dimension boxes after filtering")
                results['dimension_boxes'] = valid_boxes
                
                # Debug output of detected dimensions 
                if valid_boxes:
                    for i, box in enumerate(valid_boxes):
                        self.logger.info(f"Dimension box #{i+1}: '{box['text']}' at position {box['bbox']['x']},{box['bbox']['y']} (method: {box.get('detection_method', 'unknown')})")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error extracting text: {str(e)}")
            return {'error': str(e), 'text': ''}
    
    def calculate_material_takeoff(self, dimension_texts, element_type='wall', material='concrete'):
        """
        Calculate material takeoff based on dimension texts
        
        Args:
            dimension_texts: List of detected dimension text objects with bounding boxes
            element_type: Type of construction element (wall, floor, ceiling)
            material: Material type
            
        Returns:
            Dictionary with material takeoff calculations
        """
        self.logger.info(f"Starting material takeoff calculation for {len(dimension_texts)} dimension texts")
        self.logger.info(f"Element type: {element_type}, Material: {material}")
        
        # Check if we have any dimension texts to process
        if not dimension_texts:
            self.logger.warning("No dimension texts provided for material takeoff calculation")
            return {
                'success': False,
                'error': 'No dimension texts detected',
                'takeoff': {}
            }
            
        # Get material properties
        material_info = self.material_types.get(element_type, {})
        if not material_info:
            self.logger.warning(f"Unknown element type: {element_type}")
            return {
                'success': False,
                'error': f'Unknown element type: {element_type}',
                'takeoff': {}
            }
            
        default_thickness = material_info.get('default_thickness', 0.1)  # meters
        material_props = material_info.get('materials', {}).get(material, {})
        
        if not material_props:
            self.logger.warning(f"Unknown material: {material} for element type: {element_type}")
            return {
                'success': False,
                'error': f'Unknown material: {material} for element type: {element_type}',
                'takeoff': {}
            }
            
        density = material_props.get('density', 1000)  # kg/m³
        
        self.logger.info(f"Using default thickness: {default_thickness}m and density: {density}kg/m³")
        
        # Initialize results
        material_takeoff = {
            'element_types': {
                element_type: {
                    'count': 0,
                    'total_volume': 0.0,
                    'total_area': 0.0,
                    'unit_weight': 0.0,
                    'total_weight': 0.0,
                    'dimensions': [],
                    'materials': {
                        material: {
                            'count': 0,
                            'volume': 0.0,
                            'area': 0.0,
                            'weight': 0.0,
                            'dimensions': []
                        }
                    }
                }
            },
            'materials': {
                material: {
                    'count': 0,
                    'total_volume': 0.0,
                    'total_area': 0.0,
                    'total_weight': 0.0,
                    'element_types': [element_type],
                    'density': 0.0
                }
            }
        }
        
        # Set density in results
        material_takeoff['materials'][material]['density'] = density
        
        # Parse dimensions from text and calculate areas/volumes
        pattern_2d = re.compile(r'(\d+(?:\.\d+)?)\s*(?:mm|cm|m|in|ft)?\s*[xX]\s*(\d+(?:\.\d+)?)\s*(?:mm|cm|m|in|ft)?')
        pattern_3d = re.compile(r'(\d+(?:\.\d+)?)\s*(?:mm|cm|m|in|ft)?\s*[xX]\s*(\d+(?:\.\d+)?)\s*(?:mm|cm|m|in|ft)?\s*[xX]\s*(\d+(?:\.\d+)?)\s*(?:mm|cm|m|in|ft)?')
        
        for item in dimension_texts:
            text = item['text']
            
            # Check for 3D dimensions (length x width x height)
            match_3d = pattern_3d.search(text)
            if match_3d:
                length = float(match_3d.group(1))
                width = float(match_3d.group(2))
                height = float(match_3d.group(3))
                
                # Convert to meters if units can be detected
                if 'mm' in text:
                    length, width, height = length/1000, width/1000, height/1000
                elif 'cm' in text:
                    length, width, height = length/100, width/100, height/100
                elif 'in' in text:
                    length, width, height = length*0.0254, width*0.0254, height*0.0254
                elif 'ft' in text:
                    length, width, height = length*0.3048, width*0.3048, height*0.3048
                
                # Calculate volume and area
                volume = length * width * height
                area = 2 * (length * width + length * height + width * height)  # Surface area
                
                # Assume this is a full 3D object
                thickness = min(length, width, height)
                
            else:
                # Check for 2D dimensions (length x width)
                match_2d = pattern_2d.search(text)
                if match_2d:
                    length = float(match_2d.group(1))
                    width = float(match_2d.group(2))
                    
                    # Convert to meters if units can be detected
                    if 'mm' in text:
                        length, width = length/1000, width/1000
                    elif 'cm' in text:
                        length, width = length/100, width/100
                    elif 'in' in text:
                        length, width = length*0.0254, width*0.0254
                    elif 'ft' in text:
                        length, width = length*0.3048, width*0.3048
                    
                    # Calculate with default thickness for elements like walls
                    thickness = default_thickness
                    volume = length * width * thickness
                    area = length * width
                else:
                    # Skip if no dimensions found
                    continue
            
            # Calculate weight
            weight = volume * density  # kg
            
            # Update the material takeoff data
            material_takeoff['element_types'][element_type]['count'] += 1
            material_takeoff['element_types'][element_type]['total_volume'] += volume
            material_takeoff['element_types'][element_type]['total_area'] += area
            material_takeoff['element_types'][element_type]['unit_weight'] = density
            material_takeoff['element_types'][element_type]['total_weight'] += weight
            
            # Add dimension details
            dimension_info = {
                'text': text,
                'bbox': item['bbox'],
                'length': length,
                'width': width,
                'thickness': thickness,
                'volume': volume,
                'area': area,
                'weight': weight
            }
            
            material_takeoff['element_types'][element_type]['dimensions'].append(dimension_info)
            material_takeoff['element_types'][element_type]['materials'][material]['count'] += 1
            material_takeoff['element_types'][element_type]['materials'][material]['volume'] += volume
            material_takeoff['element_types'][element_type]['materials'][material]['area'] += area
            material_takeoff['element_types'][element_type]['materials'][material]['weight'] += weight
            material_takeoff['element_types'][element_type]['materials'][material]['dimensions'].append(dimension_info)
            
            # Update material totals
            material_takeoff['materials'][material]['count'] += 1
            material_takeoff['materials'][material]['total_volume'] += volume
            material_takeoff['materials'][material]['total_area'] += area
            material_takeoff['materials'][material]['total_weight'] += weight
        
        return material_takeoff

    def _generate_mock_results(self, image_path):
        """Generate mock results for testing without API key."""
        self.logger.info("Using mock data for floor plan analysis since API key is invalid or unavailable")
        image = cv2.imread(image_path)
        height, width, _ = image.shape
        
        results = {
            'predictions': []
        }
        
        # Generate mock walls
        for _ in range(random.randint(3, 8)):
            x = random.randint(0, width - 100)
            y = random.randint(0, height - 100)
            w = random.randint(100, 300)
            h = random.randint(20, 40)
            results['predictions'].append({
                'class': 'wall',
                'confidence': random.uniform(0.7, 0.95),
                'x': x + w/2,
                'y': y + h/2,
                'width': w,
                'height': h
            })
            
        # Generate mock doors and windows
        for _ in range(random.randint(2, 5)):
            x = random.randint(0, width - 80)
            y = random.randint(0, height - 80)
            w = random.randint(40, 80)
            h = random.randint(20, 40)
            element_type = random.choice(['door', 'window'])
            results['predictions'].append({
                'class': element_type,
                'confidence': random.uniform(0.7, 0.95),
                'x': x + w/2,
                'y': y + h/2,
                'width': w,
                'height': h
            })
            
        # Generate mock spaces
        for _ in range(random.randint(1, 3)):
            x = random.randint(0, width - 300)
            y = random.randint(0, height - 300)
            w = random.randint(200, 400)
            h = random.randint(200, 400)
            space_type = random.choice(self.space_classes)
            results['predictions'].append({
                'class': space_type,
                'confidence': random.uniform(0.7, 0.95),
                'x': x + w/2,
                'y': y + h/2,
                'width': w,
                'height': h
            })
        
        return results
            
    def process_image(self, image_path: str) -> Dict:
        """Process an image through all models and return combined results."""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image at {image_path}")
            
            if self.use_mock:
                # Use mock data if API key is invalid
                mock_results = self._generate_mock_results(image_path)
                wall_results = mock_results
                floor_results = mock_results
                space_results = mock_results
                ocr_results_1 = mock_results
                ocr_results_2 = mock_results
                ocr_results_3 = mock_results
            else:
                # Try API calls
                try:
                    # Get predictions from each model
                    self.logger.info(f"Processing with wall model: {self.wall_model_id}")
                    wall_results = self.client.infer(image_path, model_id=self.wall_model_id)
                    
                    self.logger.info(f"Processing with floor model: {self.floor_model_id}")
                    floor_results = self.client.infer(image_path, model_id=self.floor_model_id)
                    
                    self.logger.info(f"Processing with space model: {self.space_model_id}")
                    space_results = self.client.infer(image_path, model_id=self.space_model_id)
                    
                    # Add the three OCR models processing
                    self.logger.info(f"Processing with OCR model 1: {self.ocr_model_1_id}")
                    ocr_results_1 = self.client.infer(image_path, model_id=self.ocr_model_1_id)
                    
                    # Only process with the other OCR models if they have valid IDs
                    ocr_results_2 = {}
                    ocr_results_3 = {}
                    
                    if not self.ocr_model_2_id.startswith('your-'):
                        self.logger.info(f"Processing with OCR model 2: {self.ocr_model_2_id}")
                        ocr_results_2 = self.client.infer(image_path, model_id=self.ocr_model_2_id)
                    else:
                        self.logger.warning("OCR model 2 ID not set correctly, skipping")
                        
                    if not self.ocr_model_3_id.startswith('your-'):
                        self.logger.info(f"Processing with OCR model 3: {self.ocr_model_3_id}")
                        ocr_results_3 = self.client.infer(image_path, model_id=self.ocr_model_3_id)
                    else:
                        self.logger.warning("OCR model 3 ID not set correctly, skipping")
                        
                except Exception as e:
                    self.logger.error(f"Error with Roboflow API: {str(e)}")
                    # Fall back to mock data if API call fails
                    self.use_mock = True
                    mock_results = self._generate_mock_results(image_path)
                    wall_results = mock_results
                    floor_results = mock_results
                    space_results = mock_results
                    ocr_results_1 = mock_results
                    ocr_results_2 = mock_results
                    ocr_results_3 = mock_results
            
            # Process and combine results
            results = {
                'spaces': self._process_space_predictions(space_results),
                'walls': self._process_wall_predictions(wall_results),
                'floor_elements': self._process_floor_predictions(floor_results)
            }
            
            # Add OCR results from the three models
            ocr_elements = []
            
            # Process OCR results from model 1
            ocr_elements.extend(self._process_ocr_predictions(ocr_results_1, 'ocr_model_1'))
            
            # Process OCR results from models 2 and 3 if available
            if ocr_results_2.get('predictions'):
                ocr_elements.extend(self._process_ocr_predictions(ocr_results_2, 'ocr_model_2'))
                
            if ocr_results_3.get('predictions'):
                ocr_elements.extend(self._process_ocr_predictions(ocr_results_3, 'ocr_model_3'))
                
            # Add the OCR elements to the results
            results['ocr_elements'] = ocr_elements
            
            # Also extract text if needed (not included in results by default)
            # Can be called separately with extract_text method
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            raise
            
    def _process_space_predictions(self, predictions: Dict) -> List[Dict]:
        """Process space detection predictions."""
        processed = []
        for pred in predictions.get('predictions', []):
            # Check if class is in our space classes or starts with "space_"
            if pred['class'] in self.space_classes or pred['class'].startswith('space_'):
                # Extract coordinates from inference SDK format
                x = pred['x'] - pred['width'] / 2
                y = pred['y'] - pred['height'] / 2
                w = pred['width']
                h = pred['height']
                
                processed.append({
                    'class': pred['class'],
                    'confidence': pred['confidence'],
                    'bbox': {'x': x, 'y': y, 'width': w, 'height': h},
                    'center': (pred['x'], pred['y'])
                })
        return processed
        
    def _process_wall_predictions(self, predictions: Dict) -> List[Dict]:
        """Process wall detection predictions."""
        processed = []
        for pred in predictions.get('predictions', []):
            if pred['class'] == 'wall':
                # Extract coordinates from inference SDK format
                x = pred['x'] - pred['width'] / 2
                y = pred['y'] - pred['height'] / 2
                w = pred['width']
                h = pred['height']
                
                processed.append({
                    'class': 'wall',
                    'confidence': pred['confidence'],
                    'bbox': {'x': x, 'y': y, 'width': w, 'height': h},
                    'center': (pred['x'], pred['y'])
                })
        return processed
        
    def _process_floor_predictions(self, predictions: Dict) -> List[Dict]:
        """Process floor element predictions (doors, windows)."""
        processed = []
        for pred in predictions.get('predictions', []):
            if pred['class'] in ['door', 'window']:
                # Extract coordinates from inference SDK format
                x = pred['x'] - pred['width'] / 2
                y = pred['y'] - pred['height'] / 2
                w = pred['width']
                h = pred['height']
                
                processed.append({
                    'class': pred['class'],
                    'confidence': pred['confidence'],
                    'bbox': {'x': x, 'y': y, 'width': w, 'height': h},
                    'center': (pred['x'], pred['y'])
                })
        return processed
        
    def _process_ocr_predictions(self, predictions: Dict, source: str) -> List[Dict]:
        """Process OCR predictions from any model."""
        processed = []
        for pred in predictions.get('predictions', []):
            # Extract coordinates from inference SDK format
            x = pred['x'] - pred['width'] / 2
            y = pred['y'] - pred['height'] / 2
            w = pred['width']
            h = pred['height']
            
            element_class = pred['class']
            confidence = pred['confidence']
            
            processed.append({
                'class': element_class,
                'confidence': confidence,
                'bbox': {'x': x, 'y': y, 'width': w, 'height': h},
                'center': (pred['x'], pred['y']),
                'source': source  # Track which model detected this element
            })
        return processed
        
    def save_results(self, results: Dict, output_path: str) -> None:
        """Save processed results to a JSON file."""
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            raise 

    def process_linked_dimensions(self, linked_dimensions: List[Dict]) -> Dict:
        """
        Process dimensions that have been linked to walls to calculate more accurate measurements.
        
        Args:
            linked_dimensions: List of dimension-wall link objects
            
        Returns:
            Dictionary with calculated measurements for linked dimensions
        """
        try:
            self.logger.info(f"Processing {len(linked_dimensions)} linked dimensions")
            
            results = {
                'links': [],
                'walls': {},
                'dimensions': {}
            }
            
            for link in linked_dimensions:
                try:
                    dimension = link.get('dimension', {})
                    wall = link.get('wall', {})
                    
                    # Skip if missing essential data
                    if not dimension or not wall:
                        continue
                        
                    # Extract dimension text if available
                    dimension_text = dimension.get('text', '')
                    dimension_value = None
                    dimension_unit = 'mm'  # Default unit
                    
                    # Try to parse dimension text to extract value and unit
                    if dimension_text:
                        # Look for numeric values
                        num_match = re.search(r'(\d+(?:\.\d+)?)', dimension_text)
                        if num_match:
                            dimension_value = float(num_match.group(1))
                            
                            # Detect unit
                            if 'mm' in dimension_text:
                                dimension_unit = 'mm'
                                dimension_value = dimension_value / 1000  # Convert to meters
                            elif 'cm' in dimension_text:
                                dimension_unit = 'cm'
                                dimension_value = dimension_value / 100  # Convert to meters
                            elif 'm' in dimension_text:
                                dimension_unit = 'm'
                            elif 'ft' in dimension_text or "'" in dimension_text:
                                dimension_unit = 'ft'
                                dimension_value = dimension_value * 0.3048  # Convert to meters
                            elif 'in' in dimension_text or '"' in dimension_text:
                                dimension_unit = 'in'
                                dimension_value = dimension_value * 0.0254  # Convert to meters
                    
                    # If we couldn't parse a value from text, calculate from the wall
                    if dimension_value is None:
                        # Determine if the wall is horizontal or vertical
                        wall_width = wall.get('width', 0)
                        wall_height = wall.get('height', 0)
                        
                        # Assume the larger dimension is the wall length
                        dimension_value = max(wall_width, wall_height) / 100  # Pixel to meter conversion (approximate)
                    
                    # Calculate wall properties
                    wall_length = dimension_value  # In meters
                    wall_height = 2.4  # Default wall height in meters
                    wall_thickness = 0.15  # Default wall thickness in meters
                    
                    # Calculate wall area and volume
                    wall_area = wall_length * wall_height  # m²
                    wall_volume = wall_area * wall_thickness  # m³
                    
                    # Store wall data
                    wall_id = wall.get('id', f"wall_{len(results['walls']) + 1}")
                    results['walls'][wall_id] = {
                        'length': wall_length,
                        'height': wall_height,
                        'thickness': wall_thickness,
                        'area': wall_area,
                        'volume': wall_volume,
                        'length_pixels': max(wall_width, wall_height),
                        'unit': 'm'
                    }
                    
                    # Store dimension data
                    dimension_id = dimension.get('id', f"dimension_{len(results['dimensions']) + 1}")
                    results['dimensions'][dimension_id] = {
                        'value': dimension_value,
                        'text': dimension_text,
                        'parsed_value': dimension_value,
                        'unit': dimension_unit,
                        'original_unit': dimension_unit
                    }
                    
                    # Store link data
                    results['links'].append({
                        'dimension_id': dimension_id,
                        'wall_id': wall_id,
                        'valid': dimension_value is not None
                    })
                    
                    self.logger.info(f"Processed link between dimension '{dimension_text}' and wall {wall_id}")
                    
                except Exception as e:
                    self.logger.error(f"Error processing link: {str(e)}")
                    continue
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing linked dimensions: {str(e)}")
            return {'error': str(e), 'links': []} 