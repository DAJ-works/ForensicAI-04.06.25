import os
import numpy as np
import logging
import cv2
import torch
from typing import List, Dict, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ObjectDetector:
    """
    YOLOv8-based object detector.
    """
    
    def __init__(self, model_size: str = 'm', confidence_threshold: float = 0.25, device: Optional[str] = None):
        """
        Initialize the object detector.
        
        Parameters:
        -----------
        model_size : str
            Size of the YOLOv8 model ('n', 's', 'm', 'l', 'x')
        confidence_threshold : float
            Confidence threshold for detections
        device : str, optional
            Device to run the model on ('cpu', 'cuda', 'mps')
        """
        self.model_size = model_size
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.device = device
        
        # Set device automatically if not specified
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if self.device == 'cpu' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'  # Use MPS on M1/M2 Macs if available
        
        # Load model
        self._load_model()
    
    # Update the _load_model method to better utilize available hardware
    def _load_model(self):
        """Load YOLOv8 model with hardware acceleration."""
        try:
            from ultralytics import YOLO

            if torch.cuda.is_available():
                self.device = 'cuda'
                torch.backends.cudnn.benchmark = True  
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
                
            print(f"Loading yolov8{self.model_size} on {self.device}...")
            self.model = YOLO(f"yolov8s.pt")
            
            # Move model to device
            self.model.to(self.device)
            
            print(f"Model loaded with {len(self.model.names)} classes")
            
        except Exception as e:
            logger.error(f"Error loading YOLOv8 model: {e}")
            self.model = None
    
    def detect(self, image):
        """
        Detect objects in an image.
        
        Parameters:
        -----------
        image : np.ndarray
            Input image (BGR format)
            
        Returns:
        --------
        List[Dict]
            List of detection dictionaries
        """
        try:
            # Check if image is valid
            if image is None or image.size == 0:
                logger.warning("Invalid image provided to detector")
                return []
            
            # Check if model is loaded
            if self.model is None:
                logger.error("Model not loaded")
                self._load_model()
                if self.model is None:
                    return []
            
            # Convert image to RGB for YOLO
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Run inference
            results = self.model(image_rgb, verbose=False)
            
            # Process results
            detections = []
            
            # Extract results
            result = results[0]  # Get first result
            boxes = result.boxes
            
            # Process each detection
            for i, box in enumerate(boxes):
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Get confidence
                confidence = float(box.conf.cpu().numpy())
                
                # Get class
                cls_id = int(box.cls.cpu().numpy())
                class_name = result.names[cls_id]
                
                # Special handling for person class
                if class_name.lower() in ['person']:
                    # Significantly lower threshold for persons
                    if confidence < 0.10:  # Very low threshold just for people
                        continue
                elif confidence < self.confidence_threshold:
                    # Skip low confidence detections for other classes
                    continue
                
                # Create detection dictionary
                detection = {
                    "box": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(confidence),
                    "class_id": cls_id,
                    "class_name": class_name
                }
                
                # Add box center point
                box_center_x = (x1 + x2) / 2
                box_center_y = (y1 + y2) / 2
                detection["box_center"] = [float(box_center_x), float(box_center_y)]
                
                detections.append(detection)
            
            # Log detection count
            logger.debug(f"Detected {len(detections)} objects with confidence > {self.confidence_threshold}")
            
            return detections
            
        except Exception as e:
            logger.error(f"Error in object detection: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def detect_batch(self, images: List[np.ndarray]) -> List[List[Dict]]:
        """
        Detect objects in a batch of images.
        
        Parameters:
        -----------
        images : List[np.ndarray]
            List of input images (BGR format)
            
        Returns:
        --------
        List[List[Dict]]
            List of detection lists for each image
        """
        # Process each image individually
        batch_detections = []
        for image in images:
            detections = self.detect(image)
            batch_detections.append(detections)
        
        return batch_detections
    
    def visualize_detections(self, image: np.ndarray, detections: List[Dict], 
                           show_scores: bool = True, show_labels: bool = True,
                           thickness: int = 2, font_scale: float = 0.6) -> np.ndarray:
        """
        Visualize detections on the image.
        
        Parameters:
        -----------
        image : np.ndarray
            Input image (BGR format)
        detections : List[Dict]
            List of detections
        show_scores : bool
            Whether to show confidence scores
        show_labels : bool
            Whether to show class labels
        thickness : int
            Line thickness
        font_scale : float
            Font scale for text
            
        Returns:
        --------
        np.ndarray
            Image with visualized detections
        """
        image_copy = image.copy()
        
        for detection in detections:
            # Get box coordinates
            x1, y1, x2, y2 = [int(coord) for coord in detection['box']]
            
            # Get class name and confidence
            class_name = detection.get('class_name', 'Unknown')
            confidence = detection.get('confidence', 0.0)
            
            # Generate random color based on class name
            color_hash = sum(ord(c) for c in class_name) % 256
            color = [(color_hash * 17) % 256, (color_hash * 43) % 256, (color_hash * 71) % 256]
            
            # Draw box
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, thickness)
            
            # Prepare label text
            label_text = ''
            if show_labels:
                label_text += class_name
            if show_scores and confidence > 0:
                if label_text:
                    label_text += f' {confidence:.2f}'
                else:
                    label_text += f'{confidence:.2f}'
            
            # Draw label background
            if label_text:
                text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                cv2.rectangle(image_copy, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
                
                # Draw label text
                cv2.putText(image_copy, label_text, (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        return image_copy