import os
import cv2
import numpy as np
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union, Any

from .object_detector import ObjectDetector
from .object_tracker import ObjectTracker
from .vehicle_color_analyzer import VehicleColorAnalyzer
from .enhanced_interaction_detector import EnhancedInteractionDetector
from .weapon_detector import WeaponDetector


class EnhancedFilter:
    """
    Implements advanced filtering techniques to reduce false positives
    in object detection and tracking.
    """
    
    def __init__(
        self,
        class_confidence_thresholds: Optional[Dict[str, float]] = None,
        min_detection_size: Optional[Dict[str, Tuple[int, int]]] = None,
        max_detection_size: Optional[Dict[str, Tuple[int, int]]] = None,
        class_aspect_ratios: Optional[Dict[str, Tuple[float, float]]] = None,
        motion_threshold: float = 0.5,
        temporal_consistency_frames: int = 3,
        iou_threshold: float = 0.5,
        frame_height: int = 1080,
        frame_width: int = 1920
    ):
        """
        Initialize the enhanced filter.
        
        Parameters:
        -----------
        class_confidence_thresholds : Dict[str, float], optional
            Class-specific confidence thresholds
        min_detection_size : Dict[str, Tuple[int, int]], optional
            Minimum width and height for each class
        max_detection_size : Dict[str, Tuple[int, int]], optional
            Maximum width and height for each class
        class_aspect_ratios : Dict[str, Tuple[float, float]], optional
            Valid aspect ratio ranges (min, max) for each class
        motion_threshold : float
            Threshold for motion validation
        temporal_consistency_frames : int
            Number of frames to check for temporal consistency
        iou_threshold : float
            IoU threshold for non-maximum suppression
        frame_height : int
            Height of video frames
        frame_width : int
            Width of video frames
        """
        # Set default class-specific confidence thresholds
        self.class_confidence_thresholds = class_confidence_thresholds or {
            'person': 0.65,
            'car': 0.7,
            'truck': 0.7,
            'bicycle': 0.6,
            'motorcycle': 0.6,
            'bus': 0.7,
            'knife': 0.75,
            'gun': 0.8,
            'default': 0.6  # Default threshold for other classes
        }
        
        # Set default minimum detection sizes (width, height)
        self.min_detection_size = min_detection_size or {
            'person': (30, 60),
            'car': (50, 40),
            'truck': (80, 60),
            'bicycle': (30, 30),
            'motorcycle': (30, 30),
            'bus': (80, 60),
            'knife': (15, 5),
            'gun': (20, 10),
            'default': (20, 20)
        }
        
        # Set default maximum detection sizes (width, height)
        self.max_detection_size = max_detection_size or {
            'person': (300, 700),
            'car': (500, 300),
            'truck': (700, 500),
            'bicycle': (250, 200),
            'motorcycle': (250, 200),
            'bus': (800, 600),
            'knife': (150, 70),
            'gun': (200, 100),
            'default': (500, 500)
        }
        
        # Set default aspect ratio ranges (min, max) for each class
        self.class_aspect_ratios = class_aspect_ratios or {
            'person': (0.2, 0.7),   # Height typically greater than width
            'car': (1.2, 3.5),      # Width typically greater than height
            'truck': (1.2, 3.0),    # Width typically greater than height
            'bicycle': (0.8, 2.0),  # Various aspect ratios possible
            'motorcycle': (0.8, 2.0),
            'bus': (1.0, 2.5),
            'knife': (1.5, 6.0),    # Long and thin
            'gun': (1.5, 5.0),      # Longer than tall
            'default': (0.5, 2.0)   # Default aspect ratio range
        }
        
        # Other parameters
        self.motion_threshold = motion_threshold
        self.temporal_consistency_frames = temporal_consistency_frames
        self.iou_threshold = iou_threshold
        self.frame_height = frame_height
        self.frame_width = frame_width
        
        # Previous frame data for temporal consistency checks
        self.previous_detections = []  # List of previous detections by frame
        
        # Background subtractor for motion validation
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, 
            varThreshold=16, 
            detectShadows=False
        )
        
        # For moving average stabilization
        self.detection_history = {}  # object_id -> list of detections
        self.max_history = 5  # Number of frames to keep in history
    
    def filter_detections(
        self, 
        detections: List[Dict], 
        frame: np.ndarray,
        frame_idx: int
    ) -> List[Dict]:
        """
        Apply all filtering techniques to reduce false positives.
        
        Parameters:
        -----------
        detections : List[Dict]
            List of detection dictionaries
        frame : np.ndarray
            Current video frame
        frame_idx : int
            Current frame index
        
        Returns:
        --------
        List[Dict]
            Filtered detections
        """
        if not detections:
            return []
        
        # Update frame dimensions if needed
        if frame is not None:
            self.frame_height, self.frame_width = frame.shape[:2]
        
        # 1. Apply confidence thresholds
        detections = self._filter_by_confidence(detections)
        
        # 2. Apply size and aspect ratio filters
        detections = self._filter_by_size_and_ratio(detections)
        
        # 3. Apply non-maximum suppression
        detections = self._apply_nms(detections)
        
        # 4. Apply motion validation if frame is available
        if frame is not None:
            detections = self._validate_with_motion(detections, frame)
        
        # 5. Apply temporal consistency
        detections = self._check_temporal_consistency(detections, frame_idx)
        
        # 6. Stabilize detections using moving average
        detections = self._stabilize_detections(detections, frame_idx)
        
        # Update previous detections history
        self._update_detection_history(detections, frame_idx)
        
        return detections
    
    def _filter_by_confidence(self, detections: List[Dict]) -> List[Dict]:
        """Apply class-specific confidence thresholds."""
        filtered = []
        for det in detections:
            class_name = det['class_name']
            confidence = det['confidence']
            
            # Get threshold for this class, or use default
            threshold = self.class_confidence_thresholds.get(
                class_name, 
                self.class_confidence_thresholds['default']
            )
            
            # For weapon types, use higher thresholds
            if det.get('type') == 'weapon':
                threshold = max(threshold, 0.7)  # Minimum 0.7 for weapons
            
            if confidence >= threshold:
                filtered.append(det)
        
        return filtered
    
    def _filter_by_size_and_ratio(self, detections: List[Dict]) -> List[Dict]:
        """Filter by size and aspect ratio constraints."""
        filtered = []
        
        for det in detections:
            class_name = det['class_name']
            box = det['box']
            
            # Calculate width and height
            width = box[2] - box[0]
            height = box[3] - box[1]
            
            # Get constraints for this class
            min_size = self.min_detection_size.get(
                class_name, 
                self.min_detection_size['default']
            )
            max_size = self.max_detection_size.get(
                class_name, 
                self.max_detection_size['default']
            )
            aspect_range = self.class_aspect_ratios.get(
                class_name, 
                self.class_aspect_ratios['default']
            )
            
            # Size check - ensure object is within reasonable size limits
            size_ok = (
                width >= min_size[0] and 
                height >= min_size[1] and
                width <= max_size[0] and 
                height <= max_size[1]
            )
            
            # Aspect ratio check
            aspect_ratio = width / height if height > 0 else 0
            aspect_ok = aspect_range[0] <= aspect_ratio <= aspect_range[1]
            
            # Scale size constraints based on distance from camera (approximated by y position)
            # Objects higher in the frame (smaller y) should be smaller
            y_pos_factor = (box[3] / self.frame_height) * 0.8 + 0.2  # Scale factor 0.2-1.0
            
            min_size_adjusted = (min_size[0] * y_pos_factor, min_size[1] * y_pos_factor)
            size_ok_adjusted = (
                width >= min_size_adjusted[0] and 
                height >= min_size_adjusted[1]
            )
            
            if size_ok_adjusted and aspect_ok:
                filtered.append(det)
        
        return filtered
    
    def _apply_nms(self, detections: List[Dict]) -> List[Dict]:
        """Apply non-maximum suppression to reduce duplicate detections."""
        if not detections:
            return []
        
        # Group detections by class
        detections_by_class = {}
        for det in detections:
            class_name = det['class_name']
            if class_name not in detections_by_class:
                detections_by_class[class_name] = []
            detections_by_class[class_name].append(det)
        
        # Apply NMS for each class
        filtered_detections = []
        for class_name, class_dets in detections_by_class.items():
            # Extract boxes and scores
            boxes = [d['box'] for d in class_dets]
            scores = [d['confidence'] for d in class_dets]
            
            # Convert to numpy arrays
            boxes_np = np.array(boxes, dtype=np.float32)
            scores_np = np.array(scores, dtype=np.float32)
            
            # Apply NMS
            indices = self._nms(boxes_np, scores_np, self.iou_threshold)
            
            # Add filtered detections
            for i in indices:
                filtered_detections.append(class_dets[i])
        
        return filtered_detections
    
    def _nms(self, boxes: np.ndarray, scores: np.ndarray, threshold: float) -> List[int]:
        """Non-maximum suppression implementation."""
        # Convert boxes from [x1, y1, x2, y2] to [x, y, w, h]
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        
        # Calculate areas
        areas = w * h
        
        # Sort by confidence score
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            # Calculate intersection
            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])
            
            w_intersection = np.maximum(0.0, xx2 - xx1)
            h_intersection = np.maximum(0.0, yy2 - yy1)
            intersection = w_intersection * h_intersection
            
            # Calculate IoU
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)
            
            # Keep indices with IoU less than threshold
            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]
            
        return keep
    
    def _validate_with_motion(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """Validate detections using motion detection."""
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Basic morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Dilate to ensure moving objects are fully covered
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)
        
        validated_detections = []
        for det in detections:
            box = det['box']
            x1, y1, x2, y2 = map(int, box)
            
            # Ensure coordinates are within frame boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1] - 1, x2)
            y2 = min(frame.shape[0] - 1, y2)
            
            # Skip invalid boxes
            if x1 >= x2 or y1 >= y2:
                continue
            
            # Extract region of interest from motion mask
            roi = fg_mask[y1:y2, x1:x2]
            
            # Calculate the percentage of moving pixels in the detection box
            if roi.size > 0:  # Ensure ROI is not empty
                motion_percentage = np.count_nonzero(roi) / roi.size
            else:
                motion_percentage = 0
            
            # For static objects (parked cars, etc.), we still accept them
            is_static_object = det['class_name'] in ['car', 'truck', 'bus'] and motion_percentage < 0.1
            
            # Add motion percentage to detection for debugging
            det['motion_percentage'] = float(motion_percentage)
            
            # Accept detections with sufficient motion or static objects
            if motion_percentage >= self.motion_threshold or is_static_object:
                validated_detections.append(det)
                
            # For weapons, we use a much lower motion threshold as they might be static
            elif det.get('type') == 'weapon' and motion_percentage >= self.motion_threshold * 0.3:
                validated_detections.append(det)
                
            # For high confidence detections, still keep them even with less motion
            elif det['confidence'] > 0.85:
                validated_detections.append(det)
        
        return validated_detections
    
    def _check_temporal_consistency(
        self, 
        detections: List[Dict], 
        frame_idx: int
    ) -> List[Dict]:
        """Check temporal consistency of detections."""
        if not self.previous_detections:
            return detections
        
        consistent_detections = []
        
        for det in detections:
            # Always keep high-confidence detections
            if det['confidence'] > 0.85:
                consistent_detections.append(det)
                continue
            
            # For lower confidence detections, check temporal consistency
            box = np.array(det['box'])
            class_name = det['class_name']
            
            # Track consistency across previous frames
            consistency_count = 0
            
            # Check the last few frames
            for prev_frame_dets in self.previous_detections[-self.temporal_consistency_frames:]:
                # Look for overlapping detections of the same class
                for prev_det in prev_frame_dets:
                    if prev_det['class_name'] != class_name:
                        continue
                    
                    prev_box = np.array(prev_det['box'])
                    iou = self._calculate_iou(box, prev_box)
                    
                    if iou > 0.3:  # Lower threshold for temporal consistency
                        consistency_count += 1
                        break
            
            # Accept if found in multiple previous frames or high confidence
            temporal_threshold = min(len(self.previous_detections), self.temporal_consistency_frames) // 2
            if consistency_count >= temporal_threshold or det['confidence'] > 0.75:
                consistent_detections.append(det)
        
        return consistent_detections
    
    def _stabilize_detections(
        self, 
        detections: List[Dict], 
        frame_idx: int
    ) -> List[Dict]:
        """Stabilize detection boxes using moving average."""
        # Only process detections with object_id
        detections_with_id = [d for d in detections if 'object_id' in d]
        detections_without_id = [d for d in detections if 'object_id' not in d]
        
        stabilized_detections = []
        
        for det in detections_with_id:
            object_id = det['object_id']
            
            # Add to history
            if object_id not in self.detection_history:
                self.detection_history[object_id] = []
            
            history = self.detection_history[object_id]
            history.append(det)
            
            # Keep only recent history
            if len(history) > self.max_history:
                history = history[-self.max_history:]
                self.detection_history[object_id] = history
            
            # If we have enough history, smooth the bounding box
            if len(history) >= 2:
                # Create a smoothed detection
                smoothed_det = det.copy()
                
                # Apply exponential moving average for box
                box = np.array(det['box'])
                
                # Calculate weights (more recent = higher weight)
                weights = np.exp(np.linspace(0, 1, len(history)))
                weights = weights / weights.sum()
                
                # Calculate weighted average box
                avg_box = np.zeros(4)
                for i, h_det in enumerate(history):
                    avg_box += np.array(h_det['box']) * weights[i]
                
                # Update the detection with smoothed box
                smoothed_det['box'] = avg_box.tolist()
                smoothed_det['original_box'] = det['box']  # Keep original for reference
                
                stabilized_detections.append(smoothed_det)
            else:
                stabilized_detections.append(det)
        
        # Add back detections without ID
        stabilized_detections.extend(detections_without_id)
        
        return stabilized_detections
    
    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate IoU between two boxes."""
        # Box coordinates: [x1, y1, x2, y2]
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        iou = intersection / union if union > 0 else 0
        return float(iou)
    
    def _update_detection_history(self, detections: List[Dict], frame_idx: int) -> None:
        """Update the detection history for temporal consistency checks."""
        self.previous_detections.append(detections)
        
        # Keep only recent history
        if len(self.previous_detections) > self.temporal_consistency_frames * 2:
            self.previous_detections = self.previous_detections[-self.temporal_consistency_frames*2:]
    
    def get_context_aware_confidence(
        self, 
        detection: Dict, 
        all_detections: List[Dict]
    ) -> float:
        """
        Get context-aware confidence score for a detection.
        Increases confidence based on context (e.g., guns likely held by people).
        """
        class_name = detection['class_name']
        confidence = detection['confidence']
        box = detection['box']
        
        # Context adjustments
        if class_name in ['knife', 'gun'] or detection.get('type') == 'weapon':
            # Check if there's a person nearby
            for det in all_detections:
                if det['class_name'] == 'person':
                    person_box = det['box']
                    iou = self._calculate_iou(np.array(box), np.array(person_box))
                    
                    # If weapon is near person, increase confidence
                    if iou > 0.1:  # Even slight overlap or proximity
                        confidence = min(confidence * 1.2, 1.0)
                        break
        
        # For vehicles, check if they're on the ground
        if class_name in ['car', 'truck', 'bus', 'motorcycle']:
            # Bottom of box should be in lower half of frame
            bottom_y = box[3]
            if bottom_y > self.frame_height * 0.5:
                confidence = min(confidence * 1.1, 1.0)
            else:
                confidence = confidence * 0.9
        
        return confidence


class TwoStageDetector:
    """
    Implements a two-stage detection pipeline to reduce false positives.
    First stage uses the main detector, second stage validates with a different model or approach.
    """
    
    def __init__(
        self,
        primary_detector: Any,  # ObjectDetector instance
        secondary_model_path: Optional[str] = None,
        validation_threshold: float = 0.65,
        enable_second_stage: bool = True,
        class_validation_map: Optional[Dict[str, List[str]]] = None
    ):
        """
        Initialize the two-stage detector.
        
        Parameters:
        -----------
        primary_detector : ObjectDetector
            Main detector instance
        secondary_model_path : str, optional
            Path to secondary model (if None, uses a different approach)
        validation_threshold : float
            Minimum confidence for secondary validation
        enable_second_stage : bool
            Whether to enable second stage validation
        class_validation_map : Dict[str, List[str]], optional
            Map of which classes to validate with which models
        """
        self.primary_detector = primary_detector
        self.validation_threshold = validation_threshold
        self.enable_second_stage = enable_second_stage
        
        # Default validation map - which classes to validate with second stage
        self.class_validation_map = class_validation_map or {
            'person': ['person_validator'],
            'car': ['vehicle_validator'],
            'truck': ['vehicle_validator'],
            'bus': ['vehicle_validator'],
            'knife': ['weapon_validator'],
            'gun': ['weapon_validator']
        }
        
        # Load secondary models for validation
        self.secondary_models = {}
        if enable_second_stage and secondary_model_path:
            self._load_secondary_models(secondary_model_path)
        
        # Initialize feature extractor for image similarity
        self.feature_extractor = cv2.SIFT_create()
        
        # Cache for cropped detection images and their features
        self.exemplar_cache = {}  # class_name -> list of (features, image)
        self.max_exemplars_per_class = 20
        
        # Performance metrics
        self.validation_times = []
        self.validation_count = 0
    
    def _load_secondary_models(self, model_dir: str) -> None:
        """
        Load secondary models for validation.
        In a real implementation, you would load actual models here.
        """
        # This is a placeholder - in a real implementation you'd load models
        # For each class that needs validation
        try:
            self.secondary_models['person_validator'] = "person_model_placeholder"
            self.secondary_models['vehicle_validator'] = "vehicle_model_placeholder"
            self.secondary_models['weapon_validator'] = "weapon_model_placeholder"
            print(f"Loaded secondary validation models from {model_dir}")
        except Exception as e:
            print(f"Warning: Could not load secondary models: {e}")
            # Fall back to feature comparison method
    
    def detect(self, frame: np.ndarray) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Perform two-stage detection.
        
        Parameters:
        -----------
        frame : np.ndarray
            Input frame
        
        Returns:
        --------
        Tuple[List[Dict], Dict[str, Any]]
            List of validated detections and metadata
        """
        # First stage: Run primary detector
        detections, metadata = self.primary_detector.detect(frame)
        
        # If second stage is disabled or no detections, return first stage results
        if not self.enable_second_stage or not detections:
            return detections, metadata
        
        # Second stage: Validate detections
        validated_detections = []
        validation_metadata = {"validated": 0, "rejected": 0, "validation_time": 0}
        
        validation_start = time.time()
        
        for det in detections:
            class_name = det['class_name']
            confidence = det['confidence']
            
            # Skip validation for high-confidence detections or classes not in validation map
            if confidence > 0.85 or class_name not in self.class_validation_map:
                validated_detections.append(det)
                continue
            
            # Extract region for validation
            box = det['box']
            x1, y1, x2, y2 = map(int, box)
            
            # Ensure valid coordinates
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            # Skip boxes that are too small or invalid
            if x1 >= x2 or y1 >= y2 or (x2 - x1) < 10 or (y2 - y1) < 10:
                continue
            
            # Crop region of interest
            roi = frame[y1:y2, x1:x2]
            
            # Get validation result
            is_valid, validation_score = self._validate_detection(roi, class_name)
            
            # Update detection with validation score
            det['validation_score'] = validation_score
            
            # Keep valid detections
            if is_valid:
                # Blend original and validation scores
                det['confidence'] = (det['confidence'] * 0.7 + validation_score * 0.3)
                validated_detections.append(det)
                validation_metadata["validated"] += 1
            else:
                validation_metadata["rejected"] += 1
        
        validation_end = time.time()
        validation_time = validation_end - validation_start
        validation_metadata["validation_time"] = validation_time
        
        # Track performance metrics
        self.validation_times.append(validation_time)
        self.validation_count += 1
        
        # Update metadata with validation stats
        metadata.update(validation_metadata)
        
        return validated_detections, metadata
    
    def _validate_detection(self, roi: np.ndarray, class_name: str) -> Tuple[bool, float]:
        """
        Validate a detection using the second stage.
        
        Parameters:
        -----------
        roi : np.ndarray
            Region of interest (cropped detection)
        class_name : str
            Class name of the detection
        
        Returns:
        --------
        Tuple[bool, float]
            Whether the detection is valid and validation score
        """
        # Get validator models for this class
        validator_names = self.class_validation_map.get(class_name, [])
        
        # If no validators or roi is empty, return conservative result
        if not validator_names or roi.size == 0:
            return True, 0.5
        
        # Try using model-based validation if available
        for validator_name in validator_names:
            if validator_name in self.secondary_models:
                # In a real implementation, you would run the model here
                # For this example, we'll use a placeholder validation score
                validation_score = self._model_based_validation(roi, validator_name)
                is_valid = validation_score >= self.validation_threshold
                return is_valid, validation_score
        
        # Fall back to feature comparison if no models available
        return self._feature_based_validation(roi, class_name)
    
    def _model_based_validation(self, roi: np.ndarray, validator_name: str) -> float:
        """
        Validate using a model (placeholder implementation).
        In a real implementation, this would run the secondary model.
        """
        # This is a placeholder - in a real implementation you'd run the model
        # For demonstration, return a random score with a bias toward validation
        # Replace this with actual model inference
        base_score = 0.7  # Biased toward validation
        random_factor = np.random.uniform(-0.2, 0.2)  # Add some randomness
        return min(max(base_score + random_factor, 0.0), 1.0)
    
    def _feature_based_validation(self, roi: np.ndarray, class_name: str) -> Tuple[bool, float]:
        """
        Validate based on feature similarity to previous valid detections.
        """
        # Resize image for feature extraction
        resized_roi = cv2.resize(roi, (128, 128))
        
        # Convert to grayscale if needed
        if len(resized_roi.shape) == 3:
            gray_roi = cv2.cvtColor(resized_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray_roi = resized_roi
        
        # Extract SIFT features
        _, descriptors = self.feature_extractor.detectAndCompute(gray_roi, None)
        
        # If no features found, be conservative
        if descriptors is None:
            return True, 0.5
        
        # If no exemplars for this class yet, add this as first exemplar
        if class_name not in self.exemplar_cache or not self.exemplar_cache[class_name]:
            self.exemplar_cache[class_name] = [(descriptors, resized_roi)]
            return True, 0.6  # First example, be moderately confident
        
        # Compare with existing exemplars
        max_similarity = 0.0
        
        for exemplar_descriptors, _ in self.exemplar_cache[class_name]:
            # Skip if either descriptor is None
            if exemplar_descriptors is None:
                continue
                
            # Use FLANN matcher for efficient feature matching
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            
            try:
                matches = flann.knnMatch(descriptors, exemplar_descriptors, k=2)
            except:
                # If matching fails, try next exemplar
                continue
            
            # Apply ratio test
            good_matches = 0
            for i, (m, n) in enumerate(matches):
                if m.distance < 0.7 * n.distance:
                    good_matches += 1
            
            # Calculate similarity score based on good matches ratio
            if len(matches) > 0:
                similarity = good_matches / len(matches)
                max_similarity = max(max_similarity, similarity)
        
        # Add to exemplars if good confidence and we have space
        if max_similarity > 0.6 and len(self.exemplar_cache[class_name]) < self.max_exemplars_per_class:
            self.exemplar_cache[class_name].append((descriptors, resized_roi))
        
        # Determine validity based on similarity
        is_valid = max_similarity >= 0.4  # Lower threshold as feature matching is strict
        
        # Scale similarity to confidence range
        validation_score = max_similarity * 0.8 + 0.2  # Scale to 0.2-1.0 range
        
        return is_valid, validation_score
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for validation stage."""
        stats = {
            "validation_count": self.validation_count,
            "avg_validation_time": np.mean(self.validation_times) if self.validation_times else 0,
            "total_validation_time": np.sum(self.validation_times) if self.validation_times else 0
        }
        return stats


class VideoAnalyzer:
    """
    Analyzes videos using object detection and tracking.
    """
    
    def __init__(
        self,
        detector: Optional[ObjectDetector] = None,
        tracker: Optional[ObjectTracker] = None,
        weapon_detector: Optional[WeaponDetector] = None,
        output_dir: Optional[Union[str, Path]] = None,
        case_id: Optional[str] = None,
        enable_enhanced_filtering: bool = True,
        enable_two_stage_detection: bool = True
    ):
        """
        Initialize the video analyzer.
        
        Parameters:
        -----------
        detector : ObjectDetector, optional
            Detector instance. If None, creates a default detector.
        tracker : ObjectTracker, optional
            Tracker instance. If None, creates a default tracker.
        weapon_detector : WeaponDetector, optional
            Weapon detector instance. If None, creates a default detector.
        output_dir : str or Path, optional
            Directory to save results. If None, creates a timestamped directory.
        case_id : str, optional
            Unique identifier for this case/analysis session
        enable_enhanced_filtering : bool
            Whether to enable enhanced filtering to reduce false positives
        enable_two_stage_detection : bool
            Whether to enable two-stage detection validation
        """
        # Create detector if not provided
        if detector is None:
            self.detector = ObjectDetector(model_size='m', confidence_threshold=0.6)  # Increased default confidence
        else:
            self.detector = detector
        
        # Create tracker if not provided
        if tracker is None:
            self.tracker = ObjectTracker(max_age=20, min_hits=7)  # Stricter tracking parameters
        else:
            self.tracker = tracker
        
        # Create output directory if not provided
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(f"./data/analysis/video_{timestamp}")
        else:
            output_dir = Path(output_dir)
        
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate case ID if not provided
        self.case_id = case_id or f"case_{int(time.time())}"
        
        # Initialize analysis log
        self.analysis_log = []
        self._log_step("Initialized video analyzer")
        
        # Initialize enhanced components
        self.vehicle_color_analyzer = VehicleColorAnalyzer()
        
        # Initialize weapon detector if not provided
        if weapon_detector is None:
            self.weapon_detector = WeaponDetector()
        else:
            self.weapon_detector = weapon_detector
            
        # Initialize interaction detector
        self.interaction_detector = EnhancedInteractionDetector()
        
        # Initialize enhanced filtering components
        self.enable_enhanced_filtering = enable_enhanced_filtering
        if enable_enhanced_filtering:
            self.enhanced_filter = EnhancedFilter()
        
        # Initialize two-stage detection if enabled
        self.enable_two_stage_detection = enable_two_stage_detection
        if enable_two_stage_detection:
            self.two_stage_detector = TwoStageDetector(
                primary_detector=self.detector,
                secondary_model_path="weapon_detect.pt",
                validation_threshold=0.65,
                enable_second_stage=True
            )
        else:
            self.two_stage_detector = None
    
    def _log_step(self, step_name, details=None):
        """Log an analysis step with timestamp."""
        log_entry = {
            "step": step_name,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        self.analysis_log.append(log_entry)
        
        # Save log to file
        log_path = self.output_dir / "analysis_log.json"
        with open(log_path, "w") as f:
            json.dump(self.analysis_log, f, indent=4)
    
    def analyze_video(
        self,
        video_path: str,
        frame_interval: int = 1,
        save_video: bool = True,
        save_frames: bool = False,
        include_classes: Optional[List[str]] = None,
        draw_trajectories: bool = True,
        detect_weapons: bool = True,
        detect_interactions: bool = True,
        enable_enhanced_filtering: Optional[bool] = None,
        enable_two_stage_detection: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Analyze a video with object detection and tracking.
        
        Parameters:
        -----------
        video_path : str
            Path to the input video
        frame_interval : int
            Process every nth frame
        save_video : bool
            If True, save the annotated video
        save_frames : bool
            If True, save individual annotated frames
        include_classes : List[str], optional
            List of class names to include (None = all)
        draw_trajectories : bool
            If True, draw trajectory trails in the output video
        detect_weapons : bool
            If True, detect weapons in the video
        detect_interactions : bool
            If True, detect interactions between objects
        enable_enhanced_filtering : bool, optional
            Override default setting for enhanced filtering
        enable_two_stage_detection : bool, optional
            Override default setting for two-stage detection
            
        Returns:
        --------
        Dict
            Dictionary with analysis results and metadata
        """
        # Use provided values or fall back to instance defaults
        use_enhanced_filtering = enable_enhanced_filtering if enable_enhanced_filtering is not None else self.enable_enhanced_filtering
        use_two_stage_detection = enable_two_stage_detection if enable_two_stage_detection is not None else self.enable_two_stage_detection
        
        # Create results structure with enhanced parameters
        results = {
            "video_path": video_path,
            "case_id": self.case_id,
            "timestamp": datetime.now().isoformat(),
            "detector": self.detector.model_name,
            "parameters": {
                "confidence_threshold": self.detector.confidence_threshold,
                "frame_interval": frame_interval,
                "include_classes": include_classes,
                "detect_weapons": detect_weapons,
                "detect_interactions": detect_interactions,
                "enhanced_filtering": use_enhanced_filtering,
                "two_stage_detection": use_two_stage_detection
            },
            "frames": {},
            "tracks": [],
            "class_counts": {},
            "total_detections": 0,
            "valid_detections": 0,  # Count after filtering
            "filtered_detections": 0,  # Count of filtered out detections
            "total_unique_objects": 0
        }
        
        # Log enhanced parameters
        self._log_step("Starting video analysis", {
            "video_path": video_path,
            "frame_interval": frame_interval,
            "detect_weapons": detect_weapons,
            "detect_interactions": detect_interactions,
            "enhanced_filtering": use_enhanced_filtering,
            "two_stage_detection": use_two_stage_detection
        })
        
        # Reset tracker and interaction detector
        self.tracker.reset()
        
        # Reset filtering components if used
        if use_enhanced_filtering and hasattr(self, 'enhanced_filter'):
            # We'll update frame dimensions after opening the video
            pass
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Store video metadata
        results["video_metadata"] = {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration": frame_count / fps if fps > 0 else 0,
            "filename": os.path.basename(video_path)
        }
        
        # Update enhanced filter dimensions if enabled
        if use_enhanced_filtering and hasattr(self, 'enhanced_filter'):
            self.enhanced_filter.frame_width = width
            self.enhanced_filter.frame_height = height
        
        # Set up video writer if needed
        out = None
        if save_video:
            out_path = self.output_dir / f"{Path(video_path).stem}_analyzed.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
            results["output_video"] = str(out_path)
        
        # Set up frames directory if needed
        frames_dir = None
        if save_frames:
            frames_dir = self.output_dir / "frames"
            frames_dir.mkdir(exist_ok=True)
            results["frames_dir"] = str(frames_dir)
        
        # Process video
        frame_idx = 0
        processed_frames = 0
        
        # Dictionary to track object colors for visualization
        object_colors = {}
        
        # Store all trajectories
        all_trajectories = []
        
        # Initialize interactions list
        if detect_interactions:
            results["interactions"] = []
            results["weapon_alerts"] = []
        
        print(f"Analyzing video: {video_path}")
        print(f"Total frames: {frame_count}, Processing every {frame_interval} frame(s)")
        
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate timestamp
            timestamp = frame_idx / fps if fps > 0 else 0
            
            # Process at specified interval
            if frame_idx % frame_interval == 0:
                # Run detection - either two-stage or standard
                if use_two_stage_detection and hasattr(self, 'two_stage_detector'):
                    detections, detection_metadata = self.two_stage_detector.detect(frame)
                    results["two_stage_stats"] = detection_metadata
                else:
                    # Standard detection
                    detections, _ = self.detector.detect(frame)
                
                # Run weapon detection if enabled
                weapon_detections = []
                if detect_weapons:
                    weapon_detections = self.weapon_detector.detect(frame)
                
                # Combine all detections
                all_detections = detections + weapon_detections
                
                # Apply enhanced filtering if enabled
                pre_filter_count = len(all_detections)
                if use_enhanced_filtering and hasattr(self, 'enhanced_filter'):
                    all_detections = self.enhanced_filter.filter_detections(
                        all_detections, 
                        frame, 
                        frame_idx
                    )
                
                # Update filtering stats
                results["total_detections"] += pre_filter_count
                results["valid_detections"] += len(all_detections)
                results["filtered_detections"] += (pre_filter_count - len(all_detections))
                
                # Filter by class if needed
                if include_classes:
                    all_detections = [d for d in all_detections if d['class_name'] in include_classes]
                
                # For vehicle detections, analyze color
                for detection in all_detections:
                    if detection['class_name'] in ['car', 'truck', 'bus']:
                        # Extract vehicle image
                        box = detection.get('box', [0, 0, 0, 0])
                        x1, y1, x2, y2 = map(int, box)
                        
                        # Ensure valid box dimensions
                        if x1 < x2 and y1 < y2 and x1 >= 0 and y1 >= 0 and x2 < frame.shape[1] and y2 < frame.shape[0]:
                            vehicle_img = frame[y1:y2, x1:x2]
                            
                            # Analyze color if image is valid
                            if vehicle_img.size > 0:
                                detection['color'] = self.vehicle_color_analyzer.analyze_vehicle_color(vehicle_img)
                
                # Update tracker with all detections
                tracked_objects = self.tracker.update(all_detections, frame_idx, timestamp)
                
                # Add object IDs to detections
                for det in all_detections:
                    if det.get('object_id') is None:
                        # Find matching tracked object
                        matching_objects = [obj for obj in tracked_objects 
                                          if self._is_same_detection(det, obj)]
                        if matching_objects:
                            det['object_id'] = matching_objects[0]['object_id']
                
                # Update interaction detector if enabled
                if detect_interactions:
                    interactions = self.interaction_detector.update(
                        frame_idx, 
                        timestamp, 
                        tracked_objects,
                        frame
                    )
                    
                    # Add interactions to results
                    if interactions:
                        results["interactions"].extend(interactions)
                
                # Save frame results
                frame_key = f"frame_{frame_idx:06d}"
                results["frames"][frame_key] = {
                    "frame_number": frame_idx,
                    "timestamp": timestamp,
                    "detections": all_detections
                }
                
                # Update statistics
                for det in all_detections:
                    class_name = det['class_name']
                    if class_name not in results["class_counts"]:
                        results["class_counts"][class_name] = 0
                    results["class_counts"][class_name] += 1
                
                # Create annotated frame
                annotated_frame = self._draw_detections_and_tracks(
                    frame.copy(),
                    tracked_objects, 
                    object_colors,
                    draw_trajectories=draw_trajectories
                )
                
                # Draw high-severity alerts on frame if interaction detection is enabled
                if detect_interactions:
                    recent_alerts = [a for a in results["interactions"] 
                                    if a.get("severity", 0) >= 3 and 
                                    a.get("frame", 0) >= frame_idx - 10]
                    
                    if recent_alerts:
                        # Draw alert box at top of frame
                        cv2.rectangle(
                            annotated_frame,
                            (10, 100),
                            (width - 10, 150),
                            (0, 0, 255),
                            -1
                        )
                        
                        # Draw alert text
                        cv2.putText(
                            annotated_frame,
                            f"⚠️ ALERT: {recent_alerts[0]['type'].replace('_', ' ').upper()}",
                            (20, 135),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA
                        )
                
                # Add timestamp and frame info
                self._add_frame_info(
                    annotated_frame, 
                    frame_idx, 
                    timestamp, 
                    len(tracked_objects),
                    len(self.tracker.get_active_tracks())
                )
                
                # Save annotated frame if requested
                if save_frames and frames_dir:
                    frame_path = frames_dir / f"{frame_key}.jpg"
                    cv2.imwrite(str(frame_path), annotated_frame)
                
                # Add to output video if requested
                if save_video and out:
                    out.write(annotated_frame)
                
                processed_frames += 1
                
                # Show progress
                if processed_frames % 10 == 0:
                    elapsed = time.time() - start_time
                    fps_rate = processed_frames / elapsed if elapsed > 0 else 0
                    eta = (frame_count - frame_idx) / (fps_rate * frame_interval) if fps_rate > 0 else 0
                    print(f"Processed {frame_idx+1}/{frame_count} frames " 
                          f"({processed_frames} analyzed, {fps_rate:.1f} fps, ETA: {eta:.1f}s)")
            
            # Always write the frame to output video if not analyzed but video saving is enabled
            elif save_video and out:
                out.write(frame)
            
            frame_idx += 1
        
        # Get all trajectories from the tracker
        all_trajectories = self.tracker.get_trajectories()
        results["tracks"] = all_trajectories
        results["total_unique_objects"] = len(all_trajectories)
        
        # Generate security report if interaction detection was enabled
        if detect_interactions:
            # Store weapon alerts
            results["weapon_alerts"] = self.interaction_detector.get_weapon_alerts(min_severity=3)
            
            # Generate and save security report
            security_report = self.interaction_detector.generate_security_report()
            report_path = self.output_dir / f"{Path(video_path).stem}_security_report.md"
            with open(report_path, "w") as f:
                f.write(security_report)
            results["security_report"] = str(report_path)
            
            # Generate and save interaction report
            interaction_report = self.interaction_detector.generate_report()
            interaction_report_path = self.output_dir / f"{Path(video_path).stem}_interactions.md"
            with open(interaction_report_path, "w") as f:
                f.write(interaction_report)
            results["interaction_report"] = str(interaction_report_path)
        
        # Release resources
        cap.release()
        if out:
            out.release()
        
        # Add performance metrics
        elapsed_time = time.time() - start_time
        results["performance"] = {
            "elapsed_time": elapsed_time,
            "frames_per_second": processed_frames / elapsed_time if elapsed_time > 0 else 0,
            "processed_frames": processed_frames,
            "total_frames": frame_count
        }
        
        # If two-stage detection was used, add its performance stats
        if use_two_stage_detection and hasattr(self, 'two_stage_detector'):
            results["performance"]["two_stage_stats"] = self.two_stage_detector.get_performance_stats()
        
        # Save results to JSON
        results_path = self.output_dir / f"{Path(video_path).stem}_analysis.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)
        
        # Log completion with additional metrics
        self._log_step("Video analysis complete", {
            "total_frames": frame_count,
            "processed_frames": processed_frames,
            "total_detections": results["total_detections"],
            "valid_detections": results["valid_detections"],
            "filtered_detections": results["filtered_detections"],
            "total_unique_objects": results["total_unique_objects"],
            "elapsed_time": elapsed_time,
            "weapon_detections": len(results.get("weapon_alerts", [])) if detect_weapons else 0,
            "significant_interactions": len(results.get("interactions", [])) if detect_interactions else 0,
            "enhanced_filtering": use_enhanced_filtering,
            "two_stage_detection": use_two_stage_detection
        })
        
        # Print summary with enhanced information
        print(f"Analysis complete:")
        print(f"- Processed {processed_frames} frames in {elapsed_time:.2f} seconds")
        print(f"- Detected {results['total_detections']} objects")
        print(f"- Filtered out {results['filtered_detections']} false positives")
        print(f"- Valid detections: {results['valid_detections']}")
        print(f"- Tracked {results['total_unique_objects']} unique objects")
        
        if detect_weapons and "weapon_alerts" in results:
            print(f"- Detected {len(results['weapon_alerts'])} significant weapon alerts")
            
        if detect_interactions and "interactions" in results:
            print(f"- Identified {len(results['interactions'])} interactions")
            
        print(f"- Results saved to {results_path}")
        
        if detect_interactions:
            print(f"- Security report saved to {results.get('security_report', 'N/A')}")
            print(f"- Interaction report saved to {results.get('interaction_report', 'N/A')}")
        
        return results
    
    # The rest of the methods remain the same as in the original class
    def _is_same_detection(self, detection: Dict, tracked_object: Dict) -> bool:
        """Check if a detection matches a tracked object."""
        # Compare boxes
        det_box = np.array(detection['box'])
        track_box = np.array(tracked_object['box'])
        
        # Calculate IoU
        x_left = max(det_box[0], track_box[0])
        y_top = max(det_box[1], track_box[1])
        x_right = min(det_box[2], track_box[2])
        y_bottom = min(det_box[3], track_box[3])
        
        if x_right < x_left or y_bottom < y_top:
            return False
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        det_area = (det_box[2] - det_box[0]) * (det_box[3] - det_box[1])
        track_area = (track_box[2] - track_box[0]) * (track_box[3] - track_box[1])
        union = det_area + track_area - intersection
        
        iou = intersection / union if union > 0 else 0
        
        # Check class match
        class_match = detection['class_id'] == tracked_object['class_id']
        
        return iou > 0.5 and class_match
    
    def _get_object_color(self, object_id: int, object_colors: Dict) -> Tuple[int, int, int]:
        """Get consistent color for an object ID."""
        if object_id not in object_colors:
            # Generate a color based on object ID (for consistency)
            hue = (object_id * 43) % 360  # Use prime number to get good distribution
            sat = 0.8  # High saturation
            val = 0.9  # High value
            
            # Convert HSV to RGB
            c = val * sat
            x = c * (1 - abs((hue / 60) % 2 - 1))
            m = val - c
            
            if hue < 60:
                r, g, b = c, x, 0
            elif hue < 120:
                r, g, b = x, c, 0
            elif hue < 180:
                r, g, b = 0, c, x
            elif hue < 240:
                r, g, b = 0, x, c
            elif hue < 300:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x
            
            # Convert to 0-255 range for BGR
            r, g, b = int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)
            
            # Store in BGR format for OpenCV
            object_colors[object_id] = (b, g, r)
        
        return object_colors[object_id]
    
    def _draw_detections_and_tracks(
        self,
        frame: np.ndarray,
        tracked_objects: List[Dict],
        object_colors: Dict,
        draw_trajectories: bool = True
    ) -> np.ndarray:
        """Draw detections, tracks, and trajectories on a frame."""
        # Get trajectories for active tracks
        trajectories = self.tracker.get_trajectories()
        
        # Draw trajectories first (so they appear behind boxes)
        if draw_trajectories:
            for traj in trajectories:
                object_id = traj['object_id']
                # Get color for this object
                color = self._get_object_color(object_id, object_colors)
                
                # Draw trajectory line
                points = traj['trajectory']
                if len(points) > 1:
                    # Convert points to int tuples
                    pts = np.array([np.round(p).astype(int) for p in points])
                    # Draw polyline
                    cv2.polylines(
                        frame, 
                        [pts], 
                        False, 
                        color, 
                        2, 
                        cv2.LINE_AA
                    )
        
        # Draw boxes for tracked objects
        for obj in tracked_objects:
            object_id = obj['object_id']
            
            # Get color for this object
            color = self._get_object_color(object_id, object_colors)
            
            # Use red color for weapon objects
            if obj.get('type') == 'weapon':
                color = (0, 0, 255)  # Red for weapons
            
            # Get box coordinates
            box = obj['box']
            x1, y1, x2, y2 = map(int, box)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw ID and class
            label = f"ID:{object_id} {obj['class_name']} {obj['confidence']:.2f}"
            
            # Add color for vehicles
            if obj['class_name'] in ['car', 'truck', 'bus'] and 'color' in obj:
                label += f" {obj['color']}"
                
            # Add threat level for weapons
            if obj.get('type') == 'weapon' and 'threat_level' in obj:
                label += f" Threat:{obj['threat_level']}"
                
            # Add validation score if available
            if 'validation_score' in obj:
                label += f" Val:{obj['validation_score']:.2f}"
            
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # Draw label background
            cv2.rectangle(
                frame, 
                (x1, y1 - label_size[1] - 10), 
                (x1 + label_size[0], y1), 
                color, 
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame, 
                label, 
                (x1, y1 - 7), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 0, 0) if obj.get('type') != 'weapon' else (255, 255, 255), 
                1, 
                cv2.LINE_AA
            )
            
            # Draw velocity vector if available
            if 'velocity' in obj and (obj['velocity'][0] != 0 or obj['velocity'][1] != 0):
                center_x, center_y = map(int, obj['box_center'])
                vel_x, vel_y = obj['velocity']
                
                # Scale velocity for visibility
                scale = 5
                end_x = int(center_x + vel_x * scale)
                end_y = int(center_y + vel_y * scale)
                
                # Draw arrow
                cv2.arrowedLine(
                    frame, 
                    (center_x, center_y), 
                    (end_x, end_y), 
                    color, 
                    2, 
                    cv2.LINE_AA, 
                    tipLength=0.3
                )
        
        return frame
    
    def _add_frame_info(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp: float,
        detections_count: int,
        active_tracks_count: int
    ) -> np.ndarray:
        """Add frame information overlay."""
        # Get frame dimensions
        h, w = frame.shape[:2]
        
        # Create background for the text
        cv2.rectangle(frame, (10, 10), (300, 90), (0, 0, 0, 0.5), -1)
        
        # Add frame number and timestamp
        cv2.putText(
            frame, 
            f"Frame: {frame_idx}", 
            (20, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
                        (255, 255, 255), 
            1, 
            cv2.LINE_AA
        )
        
        cv2.putText(
            frame, 
            f"Objects: {detections_count} (Active: {active_tracks_count})", 
            (20, 70), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (255, 255, 255), 
            1, 
            cv2.LINE_AA
        )
        
        # Add filtering info if enhanced filtering is enabled
        if hasattr(self, 'enhanced_filter'):
            cv2.putText(
                frame, 
                f"Enhanced Filtering: ON", 
                (w - 200, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (100, 255, 100), 
                1, 
                cv2.LINE_AA
            )
        
        return frame
    
    def generate_analysis_report(self, results: Dict) -> Dict:
        """
        Generate a detailed analysis report from results.
        
        Parameters:
        -----------
        results : Dict
            Results from analyze_video method
            
        Returns:
        --------
        Dict
            Analysis report
        """
        # Create report structure
        report = {
            "case_id": self.case_id,
            "video_path": results["video_path"],
            "timestamp": datetime.now().isoformat(),
            "video_info": results["video_metadata"],
            "summary": {
                "total_frames": results["video_metadata"]["frame_count"],
                "frames_analyzed": results["performance"]["processed_frames"],
                "total_detections": results["total_detections"],
                "valid_detections": results.get("valid_detections", results["total_detections"]),
                "filtered_detections": results.get("filtered_detections", 0),
                "unique_objects": results["total_unique_objects"],
                "processing_time": results["performance"]["elapsed_time"],
                "frames_per_second": results["performance"]["frames_per_second"]
            },
            "object_stats": {
                "by_class": results["class_counts"],
                "tracks": []
            },
            "timeline": [],
            "filtering": {
                "enhanced_filtering": results["parameters"].get("enhanced_filtering", False),
                "two_stage_detection": results["parameters"].get("two_stage_detection", False),
                "filter_efficiency": results.get("filtered_detections", 0) / results["total_detections"] if results["total_detections"] > 0 else 0
            }
        }
        
        # Add two-stage detection stats if available
        if "two_stage_stats" in results:
            report["filtering"]["two_stage_stats"] = results["two_stage_stats"]
        
        # Add weapon and interaction data if available
        if "weapon_alerts" in results:
            report["security"] = {
                "weapon_alerts": len(results["weapon_alerts"]),
                "alert_details": results["weapon_alerts"]
            }
            
        if "interactions" in results:
            report["interactions"] = {
                "total": len(results["interactions"]),
                "by_type": {}
            }
            
            # Count interactions by type
            for interaction in results["interactions"]:
                int_type = interaction["type"]
                if int_type not in report["interactions"]["by_type"]:
                    report["interactions"]["by_type"][int_type] = 0
                report["interactions"]["by_type"][int_type] += 1
        
        # Process tracks for detailed stats
        for track in results["tracks"]:
            if len(track["trajectory"]) < 3:
                continue  # Skip very short tracks
                
            # Calculate track duration
            duration = track["timestamps"][-1] - track["timestamps"][0]
            
            # Calculate average speed
            total_distance = 0
            points = track["trajectory"]
            for i in range(1, len(points)):
                p1 = np.array(points[i-1])
                p2 = np.array(points[i])
                distance = np.linalg.norm(p2 - p1)
                total_distance += distance
            
            avg_speed = total_distance / duration if duration > 0 else 0
            
            # Calculate bounding box size changes
            boxes = track["boxes"]
            areas = []
            for box in boxes:
                width = box[2] - box[0]
                height = box[3] - box[1]
                areas.append(width * height)
            
            min_area = min(areas)
            max_area = max(areas)
            avg_area = sum(areas) / len(areas)
            
            # Store track stats
            track_stats = {
                "object_id": track["object_id"],
                "class": track["class_name"],
                "type": track.get("type", "normal"),
                "duration": duration,
                "frames": len(track["trajectory"]),
                "avg_speed": avg_speed,
                "min_area": min_area,
                "max_area": max_area,
                "avg_area": avg_area,
                "first_seen": track["timestamps"][0],
                "last_seen": track["timestamps"][-1]
            }
            
            # Add color for vehicles
            if track["class_name"] in ["car", "truck", "bus"] and "properties" in track:
                track_stats["color"] = track.get("properties", {}).get("color", "unknown")
            
            report["object_stats"]["tracks"].append(track_stats)
        
        # Generate timeline of significant events
        report["timeline"] = self._generate_timeline(results)
        
        # Save report to file
        report_path = self.output_dir / "analysis_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
            
        return report
    
    def _generate_timeline(self, results: Dict) -> List[Dict]:
        """Generate timeline of significant events from results."""
        timeline = []
        
        # Get frame indices and timestamps
        frames = results["frames"]
        video_fps = results["video_metadata"]["fps"]
        
        # Add filtering statistics to timeline
        if "filtered_detections" in results and results["filtered_detections"] > 0:
            timeline.append({
                "event_type": "filtering_summary",
                "description": f"Enhanced filtering removed {results['filtered_detections']} potential false positives",
                "timestamp": 0,
                "priority": "info"
            })
        
        # Add weapon detections and security alerts if available
        if "weapon_alerts" in results:
            for alert in results["weapon_alerts"]:
                timeline.append({
                    "event_type": "security_alert",
                    "alert_type": alert["type"],
                    "description": alert["description"],
                    "severity": alert.get("severity", 0),
                    "frame": alert["frame"],
                    "timestamp": alert["timestamp"]
                })
                
        # Add significant interactions if available
        if "interactions" in results:
            for interaction in results["interactions"]:
                # Only include significant interactions
                if interaction["type"] in ["person_exited_vehicle", "person_entered_vehicle"]:
                    timeline.append({
                        "event_type": "significant_interaction",
                        "interaction_type": interaction["type"],
                        "description": interaction["description"],
                        "frame": interaction.get("frame", 0),
                        "timestamp": interaction["timestamp"]
                    })
        
        # Process each frame for significant events
        for frame_key, frame_data in frames.items():
            frame_number = frame_data["frame_number"]
            timestamp = frame_data["timestamp"]
            detections = frame_data["detections"]
            
            # Group detections by class
            classes = {}
            for det in detections:
                class_name = det["class_name"]
                if class_name not in classes:
                    classes[class_name] = []
                classes[class_name].append(det)
            
            # Check for significant events
            
            # 1. First appearance of a class
            for class_name, dets in classes.items():
                # Check previous frames for this class
                class_seen_before = False
                for prev_key, prev_data in frames.items():
                    if prev_data["frame_number"] >= frame_number:
                        break
                    if any(d["class_name"] == class_name for d in prev_data["detections"]):
                        class_seen_before = True
                        break
                
                if not class_seen_before and len(dets) > 0:
                    # For weapons, mark as high priority
                    is_weapon = any(d.get("type") == "weapon" for d in dets)
                    
                    timeline.append({
                        "event_type": "first_appearance",
                        "class": class_name,
                        "count": len(dets),
                        "frame": frame_number,
                        "timestamp": timestamp,
                        "priority": "high" if is_weapon else "normal"
                    })
            
            # 2. Unusual number of objects (more than average)
            # This would require full processing of all frames first, so simplified here
            if len(detections) > 5:  # Arbitrary threshold
                timeline.append({
                    "event_type": "high_object_count",
                    "count": len(detections),
                    "frame": frame_number,
                    "timestamp": timestamp
                })
            
            # 3. Specific interactions (proximity between different classes)
            # For example, person near a vehicle
            person_detections = classes.get("person", [])
            vehicle_classes = ["car", "truck", "bus", "motorcycle"]
            vehicle_detections = []
            for v_class in vehicle_classes:
                vehicle_detections.extend(classes.get(v_class, []))
            
            if person_detections and vehicle_detections:
                # Check for proximity
                for person in person_detections:
                    p_center = np.array(person["box_center"])
                    
                    for vehicle in vehicle_detections:
                        v_center = np.array(vehicle["box_center"])
                        distance = np.linalg.norm(p_center - v_center)
                        
                        # Proximity threshold based on image dimensions
                        proximity_threshold = min(
                            results["video_metadata"]["width"],
                            results["video_metadata"]["height"]
                        ) * 0.2  # 20% of smaller dimension
                        
                        if distance < proximity_threshold:
                            vehicle_color = vehicle.get("color", "unknown")
                            timeline.append({
                                "event_type": "person_vehicle_proximity",
                                "person_id": person.get("object_id"),
                                "vehicle_class": vehicle["class_name"],
                                "vehicle_color": vehicle_color,
                                "vehicle_id": vehicle.get("object_id"),
                                "distance": float(distance),
                                "frame": frame_number,
                                "timestamp": timestamp,
                                "description": f"Person near {vehicle_color} {vehicle['class_name']}"
                            })
            
            # 4. Check for weapons near people (if not already in weapon alerts)
            weapon_detections = [d for d in detections if d.get("type") == "weapon"]
            if person_detections and weapon_detections:
                for person in person_detections:
                    p_center = np.array(person["box_center"])
                    
                    for weapon in weapon_detections:
                        w_center = np.array(weapon["box_center"])
                        distance = np.linalg.norm(p_center - w_center)
                        
                        # Proximity threshold for weapons
                        proximity_threshold = min(
                            results["video_metadata"]["width"],
                            results["video_metadata"]["height"]
                        ) * 0.15  # 15% of smaller dimension
                        
                        if distance < proximity_threshold:
                            timeline.append({
                                "event_type": "person_weapon_proximity",
                                "person_id": person.get("object_id"),
                                "weapon_class": weapon["class_name"],
                                "weapon_id": weapon.get("object_id"),
                                "distance": float(distance),
                                "frame": frame_number,
                                "timestamp": timestamp,
                                "priority": "critical",
                                "description": f"Person near {weapon['class_name']}"
                            })
        
        # Sort timeline by timestamp
        timeline.sort(key=lambda x: x["timestamp"])
        
        return timeline