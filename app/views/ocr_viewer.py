import os
import json
import logging
from typing import Dict, List, Optional, Any
import cv2
import numpy as np
from flask import Blueprint, render_template, request, jsonify, send_file, current_app
from PIL import Image, ImageDraw
import io

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
            'window': (255, 255, 0, 255)           # Yellow, fully opaque
        }
        
    def visualize_results(self, image_path: str, results: Dict) -> bytes:
        """Create a visualization of the OCR results on the original image."""
        try:
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
            
            # Add a simple legend
            self._add_legend(pil_image)
                
            # Convert back to bytes
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            return img_byte_arr
            
        except Exception as e:
            logger.error(f"Error visualizing results: {str(e)}")
            raise
    
    def _add_legend(self, pil_image):
        """Add a simple legend to the image."""
        draw = ImageDraw.Draw(pil_image)
        width, height = pil_image.size
        
        # Set legend position in bottom right
        legend_x = width - 200
        legend_y = height - 120
        
        # Draw legend background
        draw.rectangle(((legend_x - 10, legend_y - 10), (legend_x + 190, legend_y + 110)), 
                      fill=(255, 255, 255, 200), outline=(0, 0, 0))
        
        # Add title
        draw.text((legend_x, legend_y), "Legend:", fill=(0, 0, 0))
        
        # Add entries
        entries = [
            ("Spaces", self.colors['space_living_room'][:3]),
            ("Walls", self.colors['wall'][:3]),
            ("Doors", self.colors['door'][:3]),
            ("Windows", self.colors['window'][:3])
        ]
        
        for i, (label, color) in enumerate(entries):
            y_pos = legend_y + 25 + i * 20
            draw.rectangle(((legend_x, y_pos), (legend_x + 15, y_pos + 15)), fill=color)
            draw.text((legend_x + 25, y_pos), label, fill=(0, 0, 0))
            
    def _draw_bbox(self, draw: Any, bbox: Dict, color: tuple, label: str) -> None:
        """Draw a bounding box with label on the image."""
        x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
        draw.rectangle([(x, y), (x + w, y + h)], outline=color, width=2)
        draw.text((x, y - 10), label, fill=color)
        
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
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
            
        image_file = request.files['image']
        if not image_file.filename:
            return jsonify({'error': 'No image selected'}), 400
            
        # Save uploaded image
        temp_path = os.path.join('temp', image_file.filename)
        os.makedirs('temp', exist_ok=True)
        image_file.save(temp_path)
        
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
            
        processor = OCRProcessor(api_key=api_key)
        results = processor.process_image(temp_path)
        
        # Create visualization
        viewer = OCRViewer()
        visualization = viewer.visualize_results(temp_path, results)
        
        # Clean up
        os.remove(temp_path)
        
        return send_file(
            io.BytesIO(visualization),
            mimetype='image/png'
        )
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return jsonify({'error': str(e)}), 500 