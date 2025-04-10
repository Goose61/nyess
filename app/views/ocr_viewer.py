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
import traceback

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
            
            # Check for OCR elements from the three models
            ocr_elements = results.get('ocr_elements', [])
            logger.info(f"Found {len(ocr_elements)} OCR elements to visualize")
            
            # Check for dimensions boxes
            dimension_boxes = results.get('dimension_boxes', [])
            logger.info(f"Found {len(dimension_boxes)} dimension boxes to visualize")
            
            # Check for text extraction results
            text_extraction = results.get('text_extraction', {})
            extracted_texts = text_extraction.get('extracted_texts', [])
            text_boxes = text_extraction.get('text_boxes', [])
            logger.info(f"Found {len(extracted_texts)} extracted texts and {len(text_boxes)} text boxes to visualize")
            
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
            
            # Draw OCR elements from the three models
            for element in ocr_elements:
                bbox = element['bbox']
                class_name = element['class']
                source = element.get('source', 'unknown')
                
                # Choose color based on element type
                if class_name == 'wall':
                    color = self.colors['wall'][:3]
                elif class_name == 'door':
                    color = self.colors['door'][:3]
                elif class_name == 'window':
                    color = self.colors['window'][:3]
                else:
                    # Use a different color for other OCR elements to distinguish them
                    color = (128, 0, 128)  # Purple for other OCR elements
                
                # Draw with dashed line to differentiate from main models
                x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
                
                # Draw dashed rectangle (simulated with short line segments)
                dash_length = 5
                for i in range(0, int(w), dash_length * 2):
                    draw.line(((x + i, y), (x + min(i + dash_length, w), y)), fill=color, width=2)
                    draw.line(((x + i, y + h), (x + min(i + dash_length, w), y + h)), fill=color, width=2)
                
                for i in range(0, int(h), dash_length * 2):
                    draw.line(((x, y + i), (x, y + min(i + dash_length, h))), fill=color, width=2)
                    draw.line(((x + w, y + i), (x + w, y + min(i + dash_length, h))), fill=color, width=2)
                
                # Add label with source info
                draw.text((x, y - 15), f"{class_name} ({source})", fill=color)
            
            # Create a separate dimension overlay to make dimensions stand out
            dim_overlay = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
            draw_dim = ImageDraw.Draw(dim_overlay)
            
            # Draw extracted text boxes if available
            if text_boxes:
                for i, box in enumerate(text_boxes):
                    x, y, w, h = box['x'], box['y'], box['width'], box['height']
                    text = box.get('text', '')
                    confidence = box.get('confidence', 0)
                    
                    # Draw with semi-transparent background
                    draw_dim.rectangle(((x, y), (x + w, y + h)), 
                                    fill=(0, 200, 255, 60),  # Light blue semi-transparent
                                    outline=(0, 150, 255), width=2)
                    
                    # Add label with a white background
                    if text:
                        text_label = f"{text[:20]}"
                        
                        # Get text dimensions
                        try:
                            # Try newest PIL version method
                            text_bbox = draw_dim.textbbox((0, 0), text_label)
                            text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
                        except (AttributeError, TypeError):
                            # Fallback to estimated size for older PIL versions
                            text_w, text_h = len(text_label) * 7, 15
                        
                        # Draw text background
                        draw_dim.rectangle(((x, y - text_h - 3), (x + text_w + 6, y - 3)), 
                                        fill=(255, 255, 255, 220), 
                                        outline=(0, 150, 255), width=1)
                        
                        # Draw text
                        draw_dim.text((x + 3, y - text_h - 3), text_label, fill=(0, 100, 200))
            
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
                        draw_dim.text((x + 3, y - text_h - 3), text_label, fill=color)
            else:
                logger.warning("No dimension boxes to visualize - this may indicate an OCR issue")
            
            # Combine the dimension overlay with the main image
            pil_image = pil_image.convert('RGBA')
            dim_overlay = dim_overlay.convert('RGBA')
            pil_image = Image.alpha_composite(pil_image, dim_overlay).convert('RGB')
            
            # Add material takeoff summary if available
            if 'material_takeoff' in results:
                pil_image = self._add_material_takeoff_summary(pil_image, results['material_takeoff'])
            
            # Add text extraction summary if available
            if 'text_extraction' in results and extracted_texts:
                pil_image = self._add_text_extraction_summary(pil_image, text_extraction)
            
            # Add legend
            pil_image = self._add_legend(pil_image)
            
            # Save the visualization to a bytes buffer
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            buffer.seek(0)
            
            logger.info("Visualization completed successfully")
            return buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Error in visualization: {str(e)}")
            # Return a simple error image
            error_img = Image.new('RGB', (800, 600), color=(255, 255, 255))
            draw = ImageDraw.Draw(error_img)
            draw.text((50, 50), f"Error in visualization: {str(e)}", fill=(255, 0, 0))
            buffer = io.BytesIO()
            error_img.save(buffer, format='PNG')
            buffer.seek(0)
            return buffer.getvalue()
    
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
            return pil_image
        except Exception as e:
            logger.error(f"Error adding material takeoff summary: {str(e)}")
            return pil_image
    
    def _add_text_extraction_summary(self, pil_image, text_extraction):
        """Add a summary of extracted text to the visualization."""
        try:
            extracted_texts = text_extraction.get('extracted_texts', [])
            if not extracted_texts:
                return pil_image
            
            # Create a new image that's taller for the summary
            padding = 180  # Increased height for the summary
            new_height = pil_image.height + padding
            summary_img = Image.new('RGB', (pil_image.width, new_height), color=(255, 255, 255))
            summary_img.paste(pil_image, (0, 0))
            
            # Draw the summary
            draw = ImageDraw.Draw(summary_img)
            
            # Draw section title
            title = "Extracted Text Summary"
            draw.rectangle(((0, pil_image.height), (pil_image.width, pil_image.height + 30)), 
                          fill=(230, 230, 255))
            draw.text((10, pil_image.height + 5), title, fill=(0, 0, 0))
            
            # Display up to 5 most relevant extracted texts
            start_y = pil_image.height + 35
            max_display = min(5, len(extracted_texts))
            display_texts = extracted_texts[:max_display]
            
            for i, text in enumerate(display_texts):
                prefix = "• "
                full_text = f"{prefix}{text}"
                draw.text((10, start_y + i * 25), full_text, fill=(0, 0, 0))
            
            # Show text count
            if len(extracted_texts) > max_display:
                draw.text((10, start_y + max_display * 25), 
                         f"... and {len(extracted_texts) - max_display} more texts extracted", 
                         fill=(100, 100, 100))
            
            return summary_img
        except Exception as e:
            logger.error(f"Error adding text extraction summary: {str(e)}")
            return pil_image
    
    def _add_legend(self, pil_image):
        """Add a legend to the visualization."""
        try:
            # Create a new image with space for the legend
            padding = 80
            new_width = pil_image.width + padding
            legend_img = Image.new('RGB', (new_width, pil_image.height), color=(255, 255, 255))
            legend_img.paste(pil_image, (0, 0))
            
            # Draw the legend
            draw = ImageDraw.Draw(legend_img)
            
            # Title
            draw.text((pil_image.width + 5, 10), "Legend", fill=(0, 0, 0))
            
            # Create legend entries
            entries = [
                {"label": "Room", "color": self.colors['space_living_room'][:3]},
                {"label": "Wall", "color": self.colors['wall'][:3]},
                {"label": "Door", "color": self.colors['door'][:3]},
                {"label": "Window", "color": self.colors['window'][:3]},
                {"label": "Dimension", "color": (255, 0, 255)},
                {"label": "Material", "color": (0, 150, 150)},
                {"label": "Text", "color": (0, 150, 255)}
            ]
            
            # Draw each entry
            for i, entry in enumerate(entries):
                y_pos = 40 + i * 25
                draw.rectangle(((pil_image.width + 5, y_pos), (pil_image.width + 20, y_pos + 15)), 
                              fill=entry['color'], outline=(0, 0, 0))
                draw.text((pil_image.width + 25, y_pos), entry['label'], fill=(0, 0, 0))
            
            return legend_img
        except Exception as e:
            logger.error(f"Error adding legend: {str(e)}")
            return pil_image
    
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
    """Render the OCR viewer page."""
    return render_template('ocr_viewer.html')
    
@ocr_viewer.route('/api/process_ocr', methods=['POST'])
def process_ocr():
    """Process an image with OCR to detect text and bounding boxes."""
    try:
        # Check if the post request has the file part
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'}), 400
        
        # Get filter preferences
        # Default to false for better text recognition
        filter_dimensions = request.form.get('filterDimensions', 'false').lower() == 'true'
        
        # Create a temporary file to save the image
        timestamp = int(time.time())
        filename = f"{timestamp}_{file.filename}"
        
        # Create temp directory if it doesn't exist
        temp_dir = os.path.join(os.getcwd(), 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save the uploaded file
        file_path = os.path.join(temp_dir, filename)
        file.save(file_path)
        
        # Process the image with OCR
        # Read the image
        img = cv2.imread(file_path)
        if img is None:
            return jsonify({'success': False, 'error': 'Failed to read image file'}), 400
        
        height, width, _ = img.shape
        
        # Use different config depending on filter preferences
        config = '--psm 11'  # Sparse text with OSD
        if filter_dimensions:
            # Focus on finding dimensions like 1000x500, 10'-6", etc.
            config += " --oem 3 -c tessedit_char_whitelist='0123456789.-\"\'x X@ abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'"
        
        # Get data including bounding boxes
        ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=config)
        
        # Process results into a structured format
        ocr_results = []
        n_boxes = len(ocr_data['level'])
        for i in range(n_boxes):
            # Filter out empty results and low confidence
            if int(ocr_data['conf'][i]) > 15 and ocr_data['text'][i].strip() != '':
                # Determine box type based on content
                box_type = 'text'
                text = ocr_data['text'][i]
                
                # Check for dimension patterns
                if re.search(r'\d+[\'\"\-\.]', text) or re.search(r'\d+\s*[xX]\s*\d+', text):
                    box_type = 'dimension'
                elif re.search(r'[Ww]ood|[Ss]teel|[Cc]oncrete|[Gg]lass|[Mm]etal|[Bb]rick', text):
                    box_type = 'material'
                
                ocr_result = {
                    'x': ocr_data['left'][i],
                    'y': ocr_data['top'][i],
                    'width': ocr_data['width'][i],
                    'height': ocr_data['height'][i],
                    'text': text,
                    'confidence': float(ocr_data['conf'][i]) / 100 if int(ocr_data['conf'][i]) > 0 else 0,
                    'type': box_type
                }
                
                ocr_results.append(ocr_result)
        
        # Now process the image through Roboflow models
        try:
            # Get the API key from config
            api_key = current_app.config.get('ROBOFLOW_API_KEY')
            if api_key:
                # Create OCR processor instance
                from app.models.ocr_processor import OCRProcessor
                processor = OCRProcessor(api_key)
                
                # Process with all models
                model_results = processor.process_image(file_path)
                
                # Add wall boxes
                for wall in model_results.get('walls', []):
                    bbox = wall['bbox']
                    ocr_results.append({
                        'x': bbox['x'],
                        'y': bbox['y'],
                        'width': bbox['width'],
                        'height': bbox['height'],
                        'text': 'Wall',
                        'confidence': wall['confidence'],
                        'type': 'wall'
                    })
                
                # Add floor element boxes (doors, windows)
                for element in model_results.get('floor_elements', []):
                    bbox = element['bbox']
                    ocr_results.append({
                        'x': bbox['x'],
                        'y': bbox['y'],
                        'width': bbox['width'],
                        'height': bbox['height'],
                        'text': element['class'].capitalize(),
                        'confidence': element['confidence'],
                        'type': element['class']
                    })
                
                # Add spaces
                for space in model_results.get('spaces', []):
                    bbox = space['bbox']
                    ocr_results.append({
                        'x': bbox['x'],
                        'y': bbox['y'],
                        'width': bbox['width'],
                        'height': bbox['height'],
                        'text': space['class'].replace('space_', 'Space: '),
                        'confidence': space['confidence'],
                        'type': 'room'
                    })
                
                # Add OCR elements from the three OCR models
                for element in model_results.get('ocr_elements', []):
                    bbox = element['bbox']
                    # Determine display type
                    element_type = element['class']
                    if element_type in ['door', 'window']:
                        display_type = element_type
                    elif element_type == 'wall':
                        display_type = 'wall'
                    else:
                        # Default for other detected elements
                        display_type = 'text'
                    
                    ocr_results.append({
                        'x': bbox['x'],
                        'y': bbox['y'],
                        'width': bbox['width'],
                        'height': bbox['height'],
                        'text': f"{element['class'].capitalize()} ({element['source']})",
                        'confidence': element['confidence'],
                        'type': display_type
                    })
        except Exception as e:
            logger.error(f"Error processing with Roboflow models: {str(e)}\n{traceback.format_exc()}")
            # Continue with just OCR results
        
        # Create a URL for the image
        image_url = f"/temp/{filename}"
        
        # Return the results
        return jsonify({
            'success': True,
            'imageUrl': image_url,
            'ocrResults': ocr_results
        })
            
    except Exception as e:
        logger.error(f"Error processing OCR: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': f'Internal server error: {str(e)}'}), 500

@ocr_viewer.route('/api/process_ocr_box', methods=['POST'])
def process_ocr_box():
    """Process a specific bounding box within an image for OCR."""
    try:
        data = request.json
        if not data or not all(k in data for k in ['x', 'y', 'width', 'height']):
            return jsonify({'success': False, 'error': 'Missing required box coordinates'}), 400
        
        # Get image path from a session identifier
        # This is a simplified version - in a real app you would use flask sessions
        # or cookies to store the current image path
        
        temp_dir = os.path.join(os.getcwd(), 'temp')
        files = os.listdir(temp_dir)
        
        # Find the most recent image file in the temp directory
        image_files = [f for f in files if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))]
        if not image_files:
            return jsonify({'success': False, 'error': 'No image found'}), 404
        
        # Sort by creation time (newest first)
        image_files.sort(key=lambda x: os.path.getctime(os.path.join(temp_dir, x)), reverse=True)
        image_path = os.path.join(temp_dir, image_files[0])
        
        # Extract the region and perform OCR
        img = cv2.imread(image_path)
        if img is None:
            return jsonify({'success': False, 'error': 'Failed to read image file'}), 400
            
        x, y, w, h = int(data['x']), int(data['y']), int(data['width']), int(data['height'])
        
        # Make sure coordinates are within image bounds
        height, width, _ = img.shape
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = max(1, min(w, width - x))
        h = max(1, min(h, height - y))
        
        # Extract the region
        roi = img[y:y+h, x:x+w]
        
        # Perform OCR on the region
        text = pytesseract.image_to_string(roi, config='--psm 7').strip()  # PSM 7 is for single line of text
        
        return jsonify({
            'success': True,
            'text': text,
            'box': {'x': x, 'y': y, 'width': w, 'height': h}
        })
        
    except Exception as e:
        logger.error(f"Error processing OCR box: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': f'Internal server error: {str(e)}'}), 500

@ocr_viewer.route('/api/save_ocr_annotations', methods=['POST'])
def save_ocr_annotations():
    """Save OCR annotations to a file."""
    try:
        data = request.json
        if not data or 'boxes' not in data:
            return jsonify({'success': False, 'error': 'Missing boxes data'}), 400
        
        boxes = data['boxes']
        
        # Find the current image being processed
        temp_dir = os.path.join(os.getcwd(), 'temp')
        files = os.listdir(temp_dir)
        
        # Find the most recent image file in the temp directory
        image_files = [f for f in files if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))]
        if not image_files:
            return jsonify({'success': False, 'error': 'No image found'}), 404
        
        # Sort by creation time (newest first)
        image_files.sort(key=lambda x: os.path.getctime(os.path.join(temp_dir, x)), reverse=True)
        image_path = os.path.join(temp_dir, image_files[0])
        image_filename = os.path.basename(image_path)
        
        # Create a results directory if it doesn't exist
        results_dir = os.path.join(current_app.config.get('RESULTS_FOLDER', 'app/results'))
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate a filename for the annotations
        base_name = os.path.splitext(image_filename)[0]
        annotations_filename = f"{base_name}_annotations.json"
        annotations_path = os.path.join(results_dir, annotations_filename)
        
        # Save annotations to file
        with open(annotations_path, 'w') as f:
            json.dump({
                'image': image_filename,
                'boxes': boxes,
                'timestamp': time.time()
            }, f, indent=4)
        
        # Create a visual representation of the annotations
        img = cv2.imread(image_path)
        result_img = img.copy()
        
        # Draw boxes on the image
        for box in boxes:
            x, y, w, h = box['x'], box['y'], box['width'], box['height']
            text = box.get('text', '')
            box_type = box.get('type', 'text')
            
            # Different colors for different types
            if box_type == 'dimension':
                color = (0, 150, 255)  # Orange for dimensions
            elif box_type == 'material':
                color = (255, 150, 0)  # Blue for materials
            else:
                color = (0, 255, 0)    # Green for other text
            
            # Draw the bounding box
            cv2.rectangle(result_img, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
            
            # Draw the text
            cv2.putText(result_img, text, (int(x), int(y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Save the result image
        result_filename = f"{base_name}_annotated.jpg"
        result_path = os.path.join(results_dir, result_filename)
        cv2.imwrite(result_path, result_img)
        
        return jsonify({
            'success': True,
            'message': 'Annotations saved successfully',
            'file': annotations_filename,
            'annotatedImage': f"/results/{result_filename}"
        })
            
    except Exception as e:
        logger.error(f"Error saving OCR annotations: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': f'Internal server error: {str(e)}'}), 500 