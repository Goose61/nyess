import os
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
import cv2
import numpy as np
from flask import Blueprint, render_template, request, jsonify, send_file, current_app
from PIL import Image, ImageDraw, ImageFont
import io
import time
import re
import pytesseract

ocr_viewer = Blueprint('ocr_viewer', __name__)
logger = logging.getLogger(__name__)

class OCRViewer:
    def __init__(self):
        """Initialize the OCR viewer."""
        # Define distinct colors for different elements 
        self.colors = {
            # Space classes (BuilderFormer)
            'space_balconi': (0, 255, 0, 100),        # Light green with alpha
            'space_bedroom': (100, 255, 100, 100),    # Medium green with alpha
            'space_dressroom': (150, 255, 150, 100),  # Light teal with alpha
            'space_front': (0, 255, 150, 100),        # Teal with alpha
            'space_kitchen': (0, 255, 255, 100),      # Cyan with alpha
            'space_living_room': (100, 255, 200, 100), # Sea green with alpha
            'space_staircase': (50, 200, 150, 100),   # Dark teal with alpha
            'space_toilet': (150, 255, 255, 100),     # Light cyan with alpha
            'space_elevator': (0, 150, 255, 100),     # Light blue with alpha
            'space_elevator_hall': (0, 100, 255, 100), # Blue with alpha
            'space_multipurpose_space': (50, 150, 200, 100), # Slate blue with alpha
            'space_other': (100, 150, 255, 100),      # Purple blue with alpha
            'space_outdoor_room': (150, 200, 255, 100), # Light purple with alpha
            
            # Wall detection
            'wall': (255, 0, 0, 255),              # Red, fully opaque
            
            # Floor elements (doors and windows)
            'door': (0, 0, 255, 255),              # Blue, fully opaque
            'window': (255, 255, 0, 255),          # Yellow, fully opaque
            
            # Dimension text
            'dimension': (255, 0, 255, 180),       # Magenta, semi-transparent
            'material': (0, 150, 150, 180)         # Teal, semi-transparent
        }
        
    def visualize_results(self, image_path: str, results: Dict) -> bytes:
        """Create a visualization of the OCR results on the original image."""
        try:
            logger.info(f"Starting visualization of results for {image_path}")
            # Read the original image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image at {image_path}")
                
            # Convert to PIL Image for drawing
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Create a transparent overlay for spaces
            overlay = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
            draw_overlay = ImageDraw.Draw(overlay)
            
            # Create a drawing context for wall and other elements
            draw = ImageDraw.Draw(pil_image)
            
            # Log what we're about to visualize
            logger.info(f"Visualizing: {len(results.get('spaces', []))} spaces, {len(results.get('walls', []))} walls, {len(results.get('floor_elements', []))} floor elements")
            
            # Check for dimensions boxes
            dimension_boxes = results.get('dimension_boxes', [])
            logger.info(f"Found {len(dimension_boxes)} dimension boxes to visualize")
            
            # Draw spaces with transparency on overlay
            for space in results.get('spaces', []):
                bbox = space['bbox']
                class_name = space['class']
                color = self.colors.get(class_name, (0, 255, 0, 100))  # Default to green if not found
                
                x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
                draw_overlay.rectangle(((x, y), (x + w, y + h)), fill=color, outline=color[:3])
                
                # Add label on original image
                draw.text((x, y - 15), class_name, fill=color[:3])
                
            # Combine overlay with original image
            pil_image = pil_image.convert('RGBA')
            pil_image = Image.alpha_composite(pil_image, overlay).convert('RGB')
            draw = ImageDraw.Draw(pil_image)
                
            # Draw walls with solid lines
            for wall in results.get('walls', []):
                bbox = wall['bbox']
                color = self.colors['wall'][:3]  # No alpha for lines
                self._draw_bbox(draw, bbox, color, 'wall')
                
            # Draw floor elements (doors, windows)
            for element in results.get('floor_elements', []):
                bbox = element['bbox']
                class_name = element['class']
                color = self.colors.get(class_name, (255, 0, 255))[:3]  # Default to magenta if not found
                self._draw_bbox(draw, bbox, color, class_name)
            
            # Create a separate dimension overlay to make dimensions stand out
            dim_overlay = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
            draw_dim = ImageDraw.Draw(dim_overlay)
            
            # Sort dimension boxes by confidence
            if dimension_boxes:
                # Sort by confidence if available
                sorted_boxes = sorted(dimension_boxes, key=lambda x: x.get('confidence', 0), reverse=True)
                
                # Separate primary and secondary matches if available
                primary_boxes = [box for box in sorted_boxes if box.get('match_type') == 'primary']
                secondary_boxes = [box for box in sorted_boxes if box.get('match_type') == 'secondary']
                
                # If match_type not available, use all boxes
                if not primary_boxes and not secondary_boxes:
                    primary_boxes = sorted_boxes
                
                logger.info(f"Visualizing {len(primary_boxes)} primary dimension boxes")
                
                # Draw primary dimension boxes with enhanced visibility
                for i, box in enumerate(primary_boxes):
                    bbox = box['bbox']
                    text = box.get('text', '')
                    box_type = box.get('type', 'dimension')
                    match_type = box.get('match_type', 'primary')
                    
                    # Use a more prominent color for primary matches
                    color = (255, 0, 255)  # Magenta for dimensions
                    if box_type == 'material':
                        color = (0, 150, 150)  # Teal for materials
                    
                    # Draw the dimension box with a thicker border and semi-transparent fill
                    x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
                    
                    # Draw on dimension overlay with semi-transparent background
                    draw_dim.rectangle(((x, y), (x + w, y + h)), 
                                    fill=(color[0], color[1], color[2], 60), 
                                    outline=color, width=4)
                    
                    # Add a prominent label with a white background
                    text_label = f"#{i+1}: {text[:30]}"
                    
                    # Get text dimensions
                    try:
                        # Try newest PIL version method
                        text_bbox = draw_dim.textbbox((0, 0), text_label)
                        text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
                    except (AttributeError, TypeError):
                        # Fallback to estimated size for older PIL versions
                        text_w, text_h = len(text_label) * 7, 15
                    
                    # Draw white background with colored border
                    draw_dim.rectangle(((x, y - text_h - 5), (x + text_w + 10, y - 5)), 
                                    fill=(255, 255, 255, 220), outline=color, width=2)
                    
                    # Draw text with the appropriate color
                    draw_dim.text((x + 5, y - text_h - 3), text_label, fill=color)
                    
                    logger.info(f"Drew primary dimension box #{i+1} with text: {text[:20]} at position {x},{y}")
                
                # Draw secondary dimension boxes (if any)
                if secondary_boxes:
                    logger.info(f"Visualizing {len(secondary_boxes)} secondary dimension boxes")
                    for i, box in enumerate(secondary_boxes):
                        bbox = box['bbox']
                        text = box.get('text', '')
                        box_type = box.get('type', 'dimension')
                        
                        # Use a less prominent color for secondary matches
                        color = (200, 0, 200)  # Lighter magenta for dimensions
                        if box_type == 'material':
                            color = (0, 120, 120)  # Lighter teal for materials
                        
                        # Draw the dimension box
                        x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
                        
                        # Draw on dimension overlay with semi-transparent background
                        draw_dim.rectangle(((x, y), (x + w, y + h)), 
                                        fill=(color[0], color[1], color[2], 40), 
                                        outline=color, width=2)
                        
                        # Simpler label for secondary matches
                        text_label = f"#{i+1+len(primary_boxes)}: {text[:20]}"
                        
                        # Get text dimensions
                        try:
                            text_bbox = draw_dim.textbbox((0, 0), text_label)
                            text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
                        except (AttributeError, TypeError):
                            text_w, text_h = len(text_label) * 6, 12
                        
                        # Draw text background
                        draw_dim.rectangle(((x, y - text_h - 3), (x + text_w + 6, y - 3)), 
                                        fill=(255, 255, 255, 180), outline=color)
                        
                        # Draw text
                        draw_dim.text((x + 3, y - text_h - 2), text_label, fill=color)
            else:
                logger.warning("No dimension boxes to visualize - this may indicate an OCR issue")
            
            # Combine the dimension overlay with the main image
            pil_image = pil_image.convert('RGBA')
            pil_image = Image.alpha_composite(pil_image, dim_overlay).convert('RGB')
            draw = ImageDraw.Draw(pil_image)
            
            # Add material takeoff info if available
            if 'material_takeoff' in results:
                logger.info("Material takeoff data found, adding to visualization")
                self._add_material_takeoff_summary(pil_image, results['material_takeoff'])
            
            # Add help text when no dimensions found
            if not dimension_boxes:
                width, height = pil_image.size
                draw.rectangle(((10, 10), (width-10, 70)), fill=(255, 255, 255, 200), outline=(0, 0, 0))
                draw.text((20, 20), "No dimensions detected.", fill=(255, 0, 0))
                draw.text((20, 40), "Try uploading a clearer floor plan with visible dimension text.", fill=(0, 0, 0))
                
            # Add a simple legend
            self._add_legend(pil_image)
                
            # Convert back to bytes
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            logger.info("Visualization complete")
            return img_byte_arr
            
        except Exception as e:
            logger.error(f"Error visualizing results: {str(e)}")
            raise
    
    def _add_material_takeoff_summary(self, pil_image, material_takeoff):
        """Add a summary of material takeoff data to the image."""
        try:
            draw = ImageDraw.Draw(pil_image)
            width, height = pil_image.size
            
            # Set summary position in top right
            summary_x = width - 300
            summary_y = 20
            
            # Draw summary background
            draw.rectangle(((summary_x - 10, summary_y - 10), (summary_x + 290, summary_y + 190)),
                          fill=(255, 255, 255, 200), outline=(0, 0, 0))
            
            # Add title
            draw.text((summary_x, summary_y), "MATERIAL TAKEOFF SUMMARY:", fill=(0, 0, 0))
            summary_y += 25
            
            # Add element type summaries
            for element_type, data in material_takeoff.get('element_types', {}).items():
                draw.text((summary_x, summary_y), f"{element_type.upper()}: {data['count']} items", fill=(0, 0, 0))
                summary_y += 20
                draw.text((summary_x + 20, summary_y), f"Volume: {data['total_volume']:.2f} m³", fill=(0, 0, 0))
                summary_y += 20
                draw.text((summary_x + 20, summary_y), f"Area: {data['total_area']:.2f} m²", fill=(0, 0, 0))
                summary_y += 20
                draw.text((summary_x + 20, summary_y), f"Weight: {data['total_weight']:.2f} kg", fill=(0, 0, 0))
                summary_y += 30
                
                # Materials breakdown
                for material, mat_data in data.get('materials', {}).items():
                    draw.text((summary_x + 20, summary_y), f"{material.upper()}: {mat_data['count']} items", fill=(0, 0, 0))
                    summary_y += 20
            
            logger.info("Added material takeoff summary to visualization")
        except Exception as e:
            logger.error(f"Error adding material takeoff summary: {str(e)}")
    
    def _add_legend(self, pil_image):
        """Add a simple legend to the image."""
        draw = ImageDraw.Draw(pil_image)
        width, height = pil_image.size
        
        # Set legend position in bottom right
        legend_x = width - 200
        legend_y = height - 200  # Extended to fit more entries
        
        # Draw legend background
        draw.rectangle(((legend_x - 10, legend_y - 10), (legend_x + 190, legend_y + 180)), 
                      fill=(255, 255, 255, 200), outline=(0, 0, 0))
        
        # Add title
        draw.text((legend_x, legend_y), "Legend:", fill=(0, 0, 0))
        
        # Add entries
        entries = [
            ("Spaces", self.colors['space_living_room'][:3]),
            ("Walls", self.colors['wall'][:3]),
            ("Doors", self.colors['door'][:3]),
            ("Windows", self.colors['window'][:3]),
            ("Dimensions", self.colors['dimension'][:3]),
            ("Materials", self.colors['material'][:3])
        ]
        
        for i, (label, color) in enumerate(entries):
            y_pos = legend_y + 25 + i * 20
            draw.rectangle(((legend_x, y_pos), (legend_x + 15, y_pos + 15)), fill=color)
            draw.text((legend_x + 25, y_pos), label, fill=(0, 0, 0))
            
    def _draw_bbox(self, draw: Any, bbox: Dict, color: tuple, label: str) -> None:
        """Draw a bounding box with label on the image."""
        x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
        draw.rectangle(((x, y), (x + w, y + h)), outline=color, width=2)
        draw.text((x, y - 10), label, fill=color)

    def visualize_text_extraction(self, image_path: str, text_results: Dict) -> bytes:
        """Create a visualization of the extracted text on the original image."""
        try:
            # Read the original image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image at {image_path}")
                
            # Create side-by-side comparison
            # Left: Original image, Right: Processed image used for extraction
            
            # Get the preprocessed image that was used for best results
            preprocess_method = text_results.get('preprocessing', 'thresh')
            
            # Get a preprocessed image to show
            from app.models.ocr_processor import OCRProcessor
            processor = OCRProcessor(api_key='')  # Empty api_key for text preprocessing only
            processed_images = processor._preprocess_image_for_text(image_path)
            processed_image = processed_images.get(preprocess_method, processed_images.get('thresh'))
            
            # Make sure processed image is RGB for display
            if len(processed_image.shape) == 2:  # If grayscale
                processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
            
            # Resize images to match height
            h_orig, w_orig = image.shape[:2]
            h_proc, w_proc = processed_image.shape[:2]
            
            # Create new canvas with combined width
            combined_width = w_orig + w_proc
            combined_height = max(h_orig, h_proc)
            
            # Create combined image
            combined_image = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
            combined_image[:h_orig, :w_orig] = image
            combined_image[:h_proc, w_orig:w_orig+w_proc] = processed_image
            
            # Convert to PIL image for adding text and dimension boxes
            pil_image = Image.fromarray(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            
            # Draw dimension bounding boxes on both original and processed image
            dimension_boxes = text_results.get('dimension_boxes', [])
            if dimension_boxes:
                for box in dimension_boxes:
                    bbox = box['bbox']
                    box_type = box.get('type', 'dimension')
                    color = self.colors.get(box_type, self.colors['dimension'])[:3]
                    
                    # Draw on original image
                    x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
                    draw.rectangle(((x, y), (x + w, y + h)), outline=color, width=2)
                    
                    # Draw a shortened version of the text
                    short_text = box['text'][:20] + "..." if len(box['text']) > 20 else box['text']
                    draw.text((x, y - 15), short_text, fill=color)
                    
                    # Draw on processed image with offset
                    draw.rectangle(((x + w_orig, y), (x + w + w_orig, y + h)), outline=color, width=2)
            
            # Add text extraction results below the images
            y_offset = combined_height + 10
            draw.rectangle(((0, y_offset-10), (combined_width, y_offset + 400)), 
                          fill=(255, 255, 255), outline=(0, 0, 0))
            
            # Add titles
            draw.text((10, y_offset), "Original Image", fill=(0, 0, 0))
            draw.text((w_orig + 10, y_offset), f"Preprocessed Image ({preprocess_method})", fill=(0, 0, 0))
            
            # Add extracted text
            y_offset += 30
            if 'filtered_text' in text_results:
                draw.text((10, y_offset), "Extracted Dimension/Material Text:", fill=(0, 0, 0))
                y_offset += 20
                
                # Format the filtered text
                filtered_text = text_results.get('filtered_text', '')
                for line in filtered_text.split('\n'):
                    if line.strip():  # Skip empty lines
                        draw.text((30, y_offset), line, fill=(0, 0, 0))
                        y_offset += 20
                        
                # Also add full text
                y_offset += 20
                draw.text((10, y_offset), "Full Extracted Text:", fill=(0, 0, 255))
                y_offset += 20
                
                full_text = text_results.get('full_text', '')
                # Limit text display to prevent huge images
                lines = full_text.split('\n')[:15]  # Show max 15 lines
                for line in lines:
                    if line.strip():  # Skip empty lines
                        draw.text((30, y_offset), line, fill=(100, 100, 100))
                        y_offset += 20
                        
                if len(full_text.split('\n')) > 15:
                    draw.text((30, y_offset), "...", fill=(100, 100, 100))
            else:
                # Just show the extracted text
                draw.text((10, y_offset), "Extracted Text:", fill=(0, 0, 0))
                y_offset += 20
                
                text = text_results.get('text', '')
                lines = text.split('\n')[:20]  # Show max 20 lines
                for line in lines:
                    if line.strip():  # Skip empty lines
                        draw.text((30, y_offset), line, fill=(0, 0, 0))
                        y_offset += 20
                        
                if len(text.split('\n')) > 20:
                    draw.text((30, y_offset), "...", fill=(0, 0, 0))
            
            # Convert back to bytes
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            return img_byte_arr
            
        except Exception as e:
            logger.error(f"Error visualizing text extraction: {str(e)}")
            raise
    
    def visualize_material_takeoff(self, image_path: str, material_takeoff: Dict, text_results: Dict) -> bytes:
        """Create a visualization of material takeoff calculations with the extracted dimensions."""
        try:
            # Read the original image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image at {image_path}")
            
            # Convert to PIL Image for drawing
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            
            # Draw dimension bounding boxes on the image
            dimension_boxes = text_results.get('dimension_boxes', [])
            if dimension_boxes:
                for i, box in enumerate(dimension_boxes):
                    bbox = box['bbox']
                    box_type = box.get('type', 'dimension')
                    color = self.colors.get(box_type, self.colors['dimension'])[:3]
                    
                    x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
                    draw.rectangle(((x, y), (x + w, y + h)), outline=color, width=2)
                    
                    # Draw box number for reference in the data table
                    draw.text((x, y - 15), f"#{i+1}: {box['text'][:15]}", fill=color)
            
            # Create a larger canvas to include material takeoff data
            width, height = pil_image.size
            canvas_width = max(1000, width)
            canvas_height = height + 600  # Extra space for material takeoff data
            
            # Create new canvas
            canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
            canvas.paste(pil_image, (0, 0))
            draw_canvas = ImageDraw.Draw(canvas)
            
            # Add title for material takeoff section
            y_offset = height + 20
            draw_canvas.text((20, y_offset), "MATERIAL TAKEOFF ANALYSIS", fill=(0, 0, 0))
            y_offset += 30
            
            # Add summary for each element type
            for element_type, data in material_takeoff.get('element_types', {}).items():
                draw_canvas.text((20, y_offset), f"Element Type: {element_type.upper()}", fill=(0, 0, 0))
                y_offset += 25
                draw_canvas.text((40, y_offset), f"Count: {data['count']}", fill=(0, 0, 0))
                y_offset += 20
                draw_canvas.text((40, y_offset), f"Total Volume: {data['total_volume']:.2f} m³", fill=(0, 0, 0))
                y_offset += 20
                draw_canvas.text((40, y_offset), f"Total Area: {data['total_area']:.2f} m²", fill=(0, 0, 0))
                y_offset += 20
                draw_canvas.text((40, y_offset), f"Total Weight: {data['total_weight']:.2f} kg", fill=(0, 0, 0))
                y_offset += 30
                
                # Materials breakdown
                draw_canvas.text((40, y_offset), "Materials:", fill=(0, 0, 0))
                y_offset += 20
                
                for material, mat_data in data.get('materials', {}).items():
                    draw_canvas.text((60, y_offset), f"{material.upper()}: {mat_data['count']} items", fill=(0, 0, 0))
                    y_offset += 20
                    draw_canvas.text((80, y_offset), f"Volume: {mat_data['volume']:.2f} m³", fill=(0, 0, 0))
                    y_offset += 20
                    draw_canvas.text((80, y_offset), f"Area: {mat_data['area']:.2f} m²", fill=(0, 0, 0))
                    y_offset += 20
                    draw_canvas.text((80, y_offset), f"Weight: {mat_data['weight']:.2f} kg", fill=(0, 0, 0))
                    y_offset += 30
            
            # Draw dimension details table
            y_offset += 20
            draw_canvas.text((20, y_offset), "DIMENSION DETAILS:", fill=(0, 0, 0))
            y_offset += 30
            
            headers = ["#", "Dimension Text", "Length (m)", "Width (m)", "Thickness (m)", "Volume (m³)", "Area (m²)", "Weight (kg)"]
            column_widths = [30, 300, 80, 80, 100, 100, 80, 100]
            total_width = sum(column_widths)
            x_positions = [20]
            for width in column_widths[:-1]:
                x_positions.append(x_positions[-1] + width)
            
            # Draw table headers
            for i, header in enumerate(headers):
                draw_canvas.text((x_positions[i], y_offset), header, fill=(0, 0, 0))
            
            # Draw header separator line
            y_offset += 20
            draw_canvas.line([(20, y_offset), (20 + total_width, y_offset)], fill=(0, 0, 0), width=1)
            y_offset += 10
            
            # Add each dimension's details
            for element_type, data in material_takeoff.get('element_types', {}).items():
                for i, dim in enumerate(data.get('dimensions', [])):
                    col_values = [
                        f"{i+1}",
                        dim['text'][:40] + "..." if len(dim['text']) > 40 else dim['text'],
                        f"{dim['length']:.3f}",
                        f"{dim['width']:.3f}",
                        f"{dim['thickness']:.3f}",
                        f"{dim['volume']:.3f}",
                        f"{dim['area']:.3f}",
                        f"{dim['weight']:.1f}"
                    ]
                    
                    for j, value in enumerate(col_values):
                        draw_canvas.text((x_positions[j], y_offset), value, fill=(0, 0, 0))
                    
                    y_offset += 25
                    
                    # Add a separator line between items
                    draw_canvas.line([(20, y_offset-5), (20 + total_width, y_offset-5)], fill=(200, 200, 200), width=1)
            
            # Convert back to bytes
            img_byte_arr = io.BytesIO()
            canvas.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            return img_byte_arr
            
        except Exception as e:
            logger.error(f"Error visualizing material takeoff: {str(e)}")
            raise
    
@ocr_viewer.route('/viewer')
@ocr_viewer.route('/floor-plan-analyzer')
def viewer():
    """Render the 2D Floor Plan Analyzer page."""
    return render_template('ocr_viewer.html')
    
@ocr_viewer.route('/process_image', methods=['POST'])
@ocr_viewer.route('/analyze-floor-plan', methods=['POST'])
def process_image():
    """Process an uploaded floor plan image and return visualization."""
    try:
        logger.info("Starting floor plan analysis process")
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
            
        image_file = request.files['image']
        if not image_file.filename:
            return jsonify({'error': 'No image selected'}), 400
            
        # Save uploaded image
        temp_path = os.path.join('temp', image_file.filename)
        os.makedirs('temp', exist_ok=True)
        image_file.save(temp_path)
        logger.info(f"Saved uploaded image to {temp_path}")
        
        # Process image
        from app.models.ocr_processor import OCRProcessor
        
        # Try to get API key from environment first, then from config
        api_key = os.getenv('ROBOFLOW_API_KEY')
        if api_key is None:
            # Try to get from config
            api_key = current_app.config.get('ROBOFLOW_API_KEY')
            
        if api_key is None or api_key == "YOUR_ROBOFLOW_API_KEY_HERE":
            return jsonify({
                'error': 'Roboflow API key not configured. Please set the ROBOFLOW_API_KEY in your environment or config file.'
            }), 500
            
        # Create processor and process the image for room detection
        logger.info("Processing floor plan image with OCR processor")
        processor = OCRProcessor(api_key=api_key)
        results = processor.process_image(temp_path)
        
        # Process for text and dimension extraction
        logger.info("Extracting dimensions and material text")
        text_results = processor.extract_text(temp_path, filter_dimensions=True, detect_bboxes=True)
        
        # Calculate material takeoff if dimensions were found
        dimension_boxes = text_results.get('dimension_boxes', [])
        logger.info(f"Found {len(dimension_boxes)} dimension boxes")
        
        if dimension_boxes:
            # Default to wall material takeoff with concrete
            logger.info("Calculating material takeoff from dimensions")
            element_type = request.form.get('element_type', 'wall')
            material = request.form.get('material', 'concrete')
            material_takeoff = processor.calculate_material_takeoff(dimension_boxes, element_type, material)
            
            # Add material takeoff and dimension boxes to results
            results['material_takeoff'] = material_takeoff
            results['dimension_boxes'] = dimension_boxes
            
            logger.info(f"Material takeoff calculation complete: {len(material_takeoff.get('element_types', {}).get(element_type, {}).get('dimensions', []))} dimensions processed")
        else:
            logger.warning("No dimensions found for material takeoff calculation")
        
        # Create visualization
        logger.info("Creating visualization of analysis results")
        viewer = OCRViewer()
        visualization = viewer.visualize_results(temp_path, results)
        
        # Clean up
        os.remove(temp_path)
        logger.info("Floor plan analysis complete")
        
        return send_file(
            io.BytesIO(visualization),
            mimetype='image/png'
        )
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return jsonify({'error': str(e)}), 500

@ocr_viewer.route('/text-extraction')
def text_extraction_page():
    """Render the text extraction page."""
    return render_template('text_extraction.html')

@ocr_viewer.route('/extract_text', methods=['POST'])
def extract_text():
    """Extract text from an uploaded image with preprocessing."""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
            
        image_file = request.files['image']
        if not image_file.filename:
            return jsonify({'error': 'No image selected'}), 400
        
        # Get filter preference (default to filtering dimensions/materials only)
        filter_dimensions = request.form.get('filter_dimensions', 'true').lower() == 'true'
            
        # Save uploaded image
        temp_path = os.path.join('temp', image_file.filename)
        os.makedirs('temp', exist_ok=True)
        image_file.save(temp_path)
        
        # Process image for text extraction
        from app.models.ocr_processor import OCRProcessor
        
        # Text extraction doesn't need Roboflow API key
        processor = OCRProcessor(api_key='')
        results = processor.extract_text(temp_path, filter_dimensions=filter_dimensions, detect_bboxes=True)
        
        # Format for return if no visualization requested
        if request.form.get('visualize', 'true').lower() != 'true':
            # Clean up temp file
            os.remove(temp_path)
            return jsonify(results)
        
        # Create visualization of text extraction
        viewer = OCRViewer()
        visualization = viewer.visualize_text_extraction(temp_path, results)
        
        # Clean up
        os.remove(temp_path)
        
        return send_file(
            io.BytesIO(visualization),
            mimetype='image/png'
        )
        
    except Exception as e:
        logger.error(f"Error extracting text: {str(e)}")
        return jsonify({'error': str(e)}), 500

@ocr_viewer.route('/material_takeoff', methods=['POST'])
def material_takeoff():
    """Calculate material takeoff from extracted text dimensions."""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
            
        image_file = request.files['image']
        if not image_file.filename:
            return jsonify({'error': 'No image selected'}), 400
        
        # Get element type and material
        element_type = request.form.get('element_type', 'wall')
        material = request.form.get('material', 'concrete')
            
        # Save uploaded image
        temp_path = os.path.join('temp', image_file.filename)
        os.makedirs('temp', exist_ok=True)
        image_file.save(temp_path)
        
        # Process image for text extraction with dimension detection
        from app.models.ocr_processor import OCRProcessor
        
        processor = OCRProcessor(api_key='')
        text_results = processor.extract_text(temp_path, filter_dimensions=True, detect_bboxes=True)
        
        # Calculate material takeoff from extracted dimensions
        dimension_boxes = text_results.get('dimension_boxes', [])
        material_takeoff = processor.calculate_material_takeoff(dimension_boxes, element_type, material)
        
        # Format for return if no visualization requested
        if request.form.get('visualize', 'true').lower() != 'true':
            # Clean up temp file
            os.remove(temp_path)
            return jsonify({
                'text_results': text_results,
                'material_takeoff': material_takeoff
            })
        
        # Create visualization of material takeoff
        viewer = OCRViewer()
        visualization = viewer.visualize_material_takeoff(temp_path, material_takeoff, text_results)
        
        # Clean up
        os.remove(temp_path)
        
        return send_file(
            io.BytesIO(visualization),
            mimetype='image/png'
        )
        
    except Exception as e:
        logger.error(f"Error calculating material takeoff: {str(e)}")
        return jsonify({'error': str(e)}), 500

@ocr_viewer.route('/process_selection', methods=['POST'])
def process_selection():
    """Process a selected region from the canvas for OCR."""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
            
        image_file = request.files['image']
        if not image_file.filename:
            return jsonify({'error': 'No image selected'}), 400
        
        # Get box coordinates if provided
        box_coordinates = {}
        if 'box_coordinates' in request.form:
            try:
                box_coordinates = json.loads(request.form['box_coordinates'])
            except:
                pass
            
        # Save temporary image
        temp_path = os.path.join('temp', f'selection_{int(time.time())}.png')
        os.makedirs('temp', exist_ok=True)
        image_file.save(temp_path)
        
        # Process image with tesseract directly for better control
        from app.models.ocr_processor import OCRProcessor
        processor = OCRProcessor(api_key='')  # No API key needed for OCR
        
        # Use different OCR settings based on what's being detected
        # For dimensions, use specific settings to detect numbers and dimension patterns
        config = '--psm 6 -c preserve_interword_spaces=1'
        
        try:
            # Resize the image to improve OCR detection (especially for small bounding boxes)
            image = cv2.imread(temp_path)
            
            # Different preprocessing for different sizes
            if image.shape[0] < 50 or image.shape[1] < 100:  # Very small selection
                # Resize to larger dimensions for better OCR
                image = cv2.resize(image, (0, 0), fx=4, fy=4)
                # Apply processing specific to dimension text
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # Apply brightness/contrast adjustment (as suggested by StackOverflow solution)
                alpha = 1.2  # Contrast control
                beta = -20   # Brightness control
                image = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
            else:
                # Preprocessing for larger selections
                # Convert to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # Apply adaptive threshold
                image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 11, 2)
            
            # Save preprocessed image for OCR
            cv2.imwrite(temp_path, image)
            
            # Run OCR on the preprocessed image
            text = pytesseract.image_to_string(image, config=config).strip()
            
            # For dimension detection, try to find numeric patterns
            confidence = 90  # Default high confidence
            
            # Check for dimension patterns
            dimension_pattern = r'\d+(?:\.\d+)?(?:\s*[xX×]\s*\d+(?:\.\d+)?)?(?:\s*(?:mm|cm|m|ft|in|\'|\"))?'
            dimension_matches = re.findall(dimension_pattern, text)
            
            if dimension_matches:
                # If found dimension patterns, use the first one
                text = dimension_matches[0]
                # Clean up text if it's a dimension
                text = re.sub(r'\s+', '', text)  # Remove spaces
                logger.info(f"Found dimension text: {text}")
            elif len(text) == 0:
                # If no text was found, try other PSM modes
                for psm in [3, 7, 8, 10]:
                    alt_config = f'--psm {psm}'
                    alt_text = pytesseract.image_to_string(image, config=alt_config).strip()
                    if alt_text:
                        text = alt_text
                        confidence = 70  # Lower confidence for alternative modes
                        break
                
            # Return processed text
            result = {
                'text': text,
                'confidence': confidence,
                'box_coordinates': box_coordinates
            }
            
            os.remove(temp_path)
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Error in OCR processing: {str(e)}")
            return jsonify({'error': str(e), 'text': ''}), 500
            
    except Exception as e:
        logger.error(f"Error processing selection: {str(e)}")
        return jsonify({'error': str(e)}), 500

@ocr_viewer.route('/save_annotations', methods=['POST'])
def save_annotations():
    """Save annotations created by the user."""
    try:
        data = request.json
        
        if not data:
            return jsonify({'error': 'No annotation data provided'}), 400
            
        # Get individual annotation sets
        ocr_boxes = data.get('ocr', [])
        user_annotations = data.get('user', [])
        dimension_links = data.get('links', [])
        
        # Create timestamp for unique filename
        timestamp = int(time.time())
        
        # Create annotation directory if it doesn't exist
        annotation_dir = os.path.join('static', 'annotations')
        os.makedirs(annotation_dir, exist_ok=True)
        
        # Save annotations to file
        annotation_file = os.path.join(annotation_dir, f'annotations_{timestamp}.json')
        
        with open(annotation_file, 'w') as f:
            json.dump(data, f, indent=2)
            
        # Store in database if needed (stub for now)
        # db.store_annotations(timestamp, data)
        
        logger.info(f"Saved annotations to {annotation_file}")
        
        return jsonify({
            'success': True,
            'message': 'Annotations saved successfully',
            'filename': annotation_file,
            'count': {
                'ocr': len(ocr_boxes),
                'user': len(user_annotations),
                'links': len(dimension_links)
            }
        })
        
    except Exception as e:
        logger.error(f"Error saving annotations: {str(e)}")
        return jsonify({'error': str(e)}), 500

@ocr_viewer.route('/process_linked_dimensions', methods=['POST'])
def process_linked_dimensions():
    """Process dimensions that have been linked to walls."""
    try:
        data = request.json
        
        if not data or 'links' not in data:
            return jsonify({'error': 'No dimension links provided'}), 400
            
        links = data.get('links', [])
        logger.info(f"Processing {len(links)} dimension links")
        
        # Process the linked dimensions
        from app.models.ocr_processor import OCRProcessor
        processor = OCRProcessor(api_key='')
        
        results = processor.process_linked_dimensions(links)
        
        # Return the processed dimensions with calculations
        return jsonify({
            'success': True,
            'results': results,
            'count': {
                'links': len(results.get('links', [])),
                'walls': len(results.get('walls', {})),
                'dimensions': len(results.get('dimensions', {}))
            }
        })
        
    except Exception as e:
        logger.error(f"Error processing linked dimensions: {str(e)}")
        return jsonify({'error': str(e)}), 500 