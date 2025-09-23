import os
import cv2
import numpy as np
import json
import logging
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .weapon_detector import WeaponDetector
from .enhanced_interaction_detector import EnhancedInteractionDetector
from .enhanced_filter import EnhancedFilter

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
            'person': 0.25,         # Lower for person detection
            'car': 0.6,
            'truck': 0.6,
            'bicycle': 0.5,
            'motorcycle': 0.5,
            'bus': 0.6,
            'knife': 0.65,
            'gun': 0.7,
            'default': 0.5  # Default threshold for other classes
        }
        
        # Set default minimum detection sizes (width, height)
        self.min_detection_size = min_detection_size or {
            'person': (25, 50),     # Smaller min size for people
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
            'person': (0.2, 0.8),   # More lenient for persons
            'car': (1.1, 3.5),
            'truck': (1.1, 3.0),
            'bicycle': (0.7, 2.0),
            'motorcycle': (0.7, 2.0),
            'bus': (1.0, 2.5),
            'knife': (1.5, 6.0),
            'gun': (1.5, 5.0),
            'default': (0.4, 2.5)   # More lenient default
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
        
        logger.info(f"Initialized EnhancedFilter with motion_threshold={motion_threshold}, " 
                   f"temporal_frames={temporal_consistency_frames}")
    
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
        
        original_count = len(detections)
        
        # Update frame dimensions if needed
        if frame is not None:
            self.frame_height, self.frame_width = frame.shape[:2]
        
        # 1. Apply confidence thresholds
        detections = self._filter_by_confidence(detections)
        confidence_filtered = len(detections)
        
        # 2. Apply size and aspect ratio filters
        detections = self._filter_by_size_and_ratio(detections)
        size_filtered = len(detections)
        
        # 3. Apply non-maximum suppression
        detections = self._apply_nms(detections)
        nms_filtered = len(detections)
        
        # 4. Apply motion validation if frame is available
        if frame is not None:
            detections = self._validate_with_motion(detections, frame)
            motion_filtered = len(detections)
        else:
            motion_filtered = nms_filtered
        
        # 5. Apply temporal consistency
        detections = self._check_temporal_consistency(detections, frame_idx)
        temporal_filtered = len(detections)
        
        # 6. Stabilize detections using moving average
        detections = self._stabilize_detections(detections, frame_idx)
        
        # Update previous detections history
        self._update_detection_history(detections, frame_idx)
        
        # Log filtering results occasionally
        if frame_idx % 50 == 0:  # Log every 50 frames to avoid excessive logging
            logger.debug(f"Frame {frame_idx} filtering stats: "
                       f"Original: {original_count}, "
                       f"After confidence: {confidence_filtered}, "
                       f"After size: {size_filtered}, "
                       f"After NMS: {nms_filtered}, "
                       f"After motion: {motion_filtered}, "
                       f"After temporal: {temporal_filtered}")
        
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
                threshold = max(threshold, 0.6)  # Minimum 0.6 for weapons
            
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
            elif class_name == 'person' and width >= min_size_adjusted[0] * 0.8 and height >= min_size_adjusted[1] * 0.8:
                # More lenient for persons to ensure we don't miss people
                det['confidence'] *= 0.9  # Slightly reduce confidence for out-of-ratio persons
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
            
            # For persons, use a lower threshold to not miss stationary people
            is_person = det['class_name'] == 'person'
            
            # Add motion percentage to detection for debugging
            det['motion_percentage'] = float(motion_percentage)
            
            # Accept detections based on motion or object type
            if motion_percentage >= self.motion_threshold or is_static_object:
                validated_detections.append(det)
            elif is_person and motion_percentage >= self.motion_threshold * 0.3:
                # More lenient for persons
                validated_detections.append(det)
            # For weapons, we use a much lower motion threshold as they might be static
            elif det.get('type') == 'weapon' and motion_percentage >= self.motion_threshold * 0.3:
                validated_detections.append(det)
            # For high confidence detections, still keep them even with less motion
            elif det['confidence'] > 0.8:
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
            if det['confidence'] > 0.8:
                consistent_detections.append(det)
                continue
            
            # Always keep persons (more lenient)
            if det['class_name'] == 'person' and det['confidence'] > 0.4:
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
            if consistency_count >= temporal_threshold or det['confidence'] > 0.65:
                consistent_detections.append(det)
        
        return consistent_detections
    
    def _stabilize_detections(
        self, 
        detections: List[Dict], 
        frame_idx: int
    ) -> List[Dict]:
        """Stabilize detection boxes using moving average."""
        # Only process detections with object_id
        detections_with_id = [d for d in detections if 'id' in d]
        detections_without_id = [d for d in detections if 'id' not in d]
        
        stabilized_detections = []
        
        for det in detections_with_id:
            object_id = det['id']
            
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


class TwoStageDetector:
    """
    Implements a two-stage detection pipeline to reduce false positives.
    First stage uses the main detector, second stage validates with a different model or approach.
    """
    
    def __init__(
        self,
        primary_detector: Any,
        validation_threshold: float = 0.6,
        enable_second_stage: bool = True
    ):
        """
        Initialize the two-stage detector.
        
        Parameters:
        -----------
        primary_detector : ObjectDetector
            Main detector instance
        validation_threshold : float
            Minimum confidence for secondary validation
        enable_second_stage : bool
            Whether to enable second stage validation
        """
        self.primary_detector = primary_detector
        self.validation_threshold = validation_threshold
        self.enable_second_stage = enable_second_stage
        
        # Map classes that need validation
        self.validate_classes = {
            'person': True,
            'car': True,
            'truck': True,
            'bus': True,
            'knife': True,
            'gun': True
        }
        
        # Initialize feature extractor for image similarity
        self.feature_extractor = cv2.SIFT_create()
        
        # Cache for exemplars of each class
        self.exemplar_cache = {}  # class_name -> list of (features, image)
        self.max_exemplars_per_class = 20
        
        # Performance metrics
        self.validation_times = []
        self.validation_count = 0
        
        logger.info(f"Initialized TwoStageDetector with validation_threshold={validation_threshold}, " 
                   f"enable_second_stage={enable_second_stage}")
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Perform two-stage detection.
        
        Parameters:
        -----------
        frame : np.ndarray
            Input frame
        
        Returns:
        --------
        List[Dict]
            List of validated detections
        """
        # First stage: Run primary detector
        detections = self.primary_detector.detect(frame)
        
        # If second stage is disabled or no detections, return first stage results
        if not self.enable_second_stage or not detections:
            return detections
        
        # Second stage: Validate detections
        validated_detections = []
        
        validation_start = time.time()
        validated_count = 0
        rejected_count = 0
        
        for det in detections:
            class_name = det['class_name']
            confidence = det['confidence']
            
            # Skip validation for high-confidence detections or classes that don't need validation
            # Always keep persons with decent confidence to avoid missing people
            if confidence > 0.8 or class_name not in self.validate_classes or \
               (class_name == 'person' and confidence > 0.6):
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
            if x1 >= x2 or y1 >= y2 or (x2 - x1) < 8 or (y2 - y1) < 8:
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
                validated_count += 1
            else:
                rejected_count += 1
        
        validation_time = time.time() - validation_start
        
        # Track performance metrics
        self.validation_times.append(validation_time)
        self.validation_count += 1
        
        # Log validation results periodically
        if self.validation_count % 20 == 0:
            logger.debug(f"Validation stats: {validated_count} validated, "
                         f"{rejected_count} rejected in {validation_time:.3f}s")
        
        return validated_detections
    
    def _validate_detection(self, roi: np.ndarray, class_name: str) -> Tuple[bool, float]:
        """
        Validate a detection using the second stage.
        """
        try:
            # Resize image for feature extraction
            resized_roi = cv2.resize(roi, (128, 128))
            
            # Convert to grayscale if needed
            if len(resized_roi.shape) == 3:
                gray_roi = cv2.cvtColor(resized_roi, cv2.COLOR_BGR2GRAY)
            else:
                gray_roi = resized_roi
            
            # Extract SIFT features
            keypoints, descriptors = self.feature_extractor.detectAndCompute(gray_roi, None)
            
            # If no features found, be conservative
            if descriptors is None or len(descriptors) < 5:
                return True, 0.5
            
            # If no exemplars for this class yet, add this as first exemplar
            if class_name not in self.exemplar_cache or not self.exemplar_cache[class_name]:
                self.exemplar_cache[class_name] = [(descriptors, resized_roi)]
                return True, 0.6  # First example, be moderately confident
            
            # Compare with existing exemplars
            max_similarity = 0.0
            
            for exemplar_descriptors, _ in self.exemplar_cache[class_name]:
                # Skip if descriptor is None
                if exemplar_descriptors is None:
                    continue
                
                try:
                    # Use FLANN matcher for efficient feature matching
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                    search_params = dict(checks=50)
                    flann = cv2.FlannBasedMatcher(index_params, search_params)
                    
                    # Make sure descriptors are in the correct format
                    descriptors_float = np.float32(descriptors)
                    exemplar_descriptors_float = np.float32(exemplar_descriptors)
                    
                    matches = flann.knnMatch(descriptors_float, exemplar_descriptors_float, k=2)
                    
                    # Apply ratio test
                    good_matches = 0
                    total_matches = 0
                    
                    for match in matches:
                        if len(match) == 2:  # Need 2 matches for ratio test
                            m, n = match
                            if m.distance < 0.7 * n.distance:
                                good_matches += 1
                            total_matches += 1
                    
                    # Calculate similarity score
                    if total_matches > 0:
                        similarity = good_matches / total_matches
                        max_similarity = max(max_similarity, similarity)
                except Exception as e:
                    logger.warning(f"Error in feature matching: {str(e)}")
                    continue
            
            # Add to exemplars if good confidence and we have space
            if max_similarity > 0.6 and len(self.exemplar_cache[class_name]) < self.max_exemplars_per_class:
                self.exemplar_cache[class_name].append((descriptors, resized_roi))
            
            # Determine validity based on similarity
            is_valid = max_similarity >= 0.4  # Lower threshold as feature matching is strict
            
            # Scale similarity to confidence range
            validation_score = max_similarity * 0.8 + 0.2  # Scale to 0.2-1.0 range
            
            return is_valid, validation_score
        
        except Exception as e:
            logger.warning(f"Error in validation: {str(e)}")
            # On error, default to accepting
            return True, 0.5
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for validation stage."""
        stats = {
            "validation_count": self.validation_count,
            "avg_validation_time": np.mean(self.validation_times) if self.validation_times else 0,
            "total_validation_time": np.sum(self.validation_times) if self.validation_times else 0,
            "enabled": self.enable_second_stage
        }
        return stats


class VideoAnalyzerWithReID:
    """
    Video analyzer with person re-identification capabilities.
    Includes enhanced false-positive reduction features.
    """
    
    def __init__(
        self, 
        detector, 
        tracker, 
        reidentifier, 
        output_dir="output", 
        enable_reid=True,
        enable_enhanced_filtering=False,
        enable_two_stage_detection=False,
        enable_weapon_detection=False,
        enable_interaction_detection=False,
        weapon_model_path=None
    ):
        """
        Initialize the video analyzer.
        
        Parameters:
        -----------
        detector : ObjectDetector
            Object detector
        tracker : ObjectTracker
            Object tracker
        reidentifier : PersonReidentifier
            Person re-identification model
        output_dir : str
            Output directory for results
        enable_reid : bool
            Whether to use person re-identification
        enable_enhanced_filtering : bool
            Whether to use enhanced filtering to reduce false positives
        enable_two_stage_detection : bool
            Whether to use two-stage detection for validation
        """
        self.detector = detector
        self.tracker = tracker
        self.reidentifier = reidentifier
        self.output_dir = output_dir
        self.enable_reid = enable_reid
        
        # Enhanced filtering settings
        self.enable_enhanced_filtering = enable_enhanced_filtering
        if enable_enhanced_filtering:
            self.enhanced_filter = EnhancedFilter(
                motion_threshold=0.3,  # Lower threshold for static scenes
                temporal_consistency_frames=2  # Fewer frames for faster processing
            )
            logger.info("Enhanced filtering enabled")
        
        # Two-stage detection settings
        self.enable_two_stage_detection = enable_two_stage_detection
        if enable_two_stage_detection:
            self.two_stage_detector = TwoStageDetector(
                primary_detector=detector,
                validation_threshold=0.6,
                enable_second_stage=True
            )
            logger.info("Two-stage detection enabled")

        self.enable_weapon_detection = enable_weapon_detection
        if enable_weapon_detection:
            try:
                # Try different model paths
                model_paths = [
                    weapon_model_path,  # From parameter
                    "./backend/models/weapon_detect.pt",
                    "../backend/models/weapon_detect.pt",
                    "/backend/models/weapon_detect.pt",
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "weapon_detect.pt"),
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models/weapon_detect.pt")
                ]
                
                model_path = None
                for path in model_paths:
                    if path and os.path.exists(path):
                        model_path = path
                        print(f"✅ Found weapon model at: {path}")
                        break
                    elif path:
                        print(f"❌ Model not found at: {path}")
                
                if model_path:
                    from .weapon_detector import WeaponDetector
                    print(f"Initializing weapon detector with model path: {model_path}")
                    self.weapon_detector = WeaponDetector(
                        model_path=model_path,
                        confidence_threshold=0.35
                    )
                else:
                    print("⚠️ No valid model path found, using fallback detector...")
                
                if not hasattr(self.weapon_detector, 'model') or self.weapon_detector.model is None:
                    print("Weapon detector initialized but model failed to load")
                    self.enable_weapon_detection = False
                else:
                    print(f"Weapon detector initialized successfully with model: {type(self.weapon_detector.model)}")
            except Exception as e:
                print(f"Failed to initialize weapon detector: {e}")
                import traceback
                traceback.print_exc()
                self.weapon_detector = None
                self.enable_weapon_detection = False
        
        # Interaction detection
        self.enable_interaction_detection = enable_interaction_detection
        if enable_interaction_detection:
            from backend.models.enhanced_interaction_detector import EnhancedInteractionDetector
            self.interaction_detector = EnhancedInteractionDetector(weapon_proximity_threshold=0.8,weapon_alert_confidence=0.4)
            logger.info("Interaction detection enabled")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize statistics and results
        self.reset()
        
        logger.info(f"Initialized VideoAnalyzerWithReID with enhanced_filtering={enable_enhanced_filtering}, "
                   f"two_stage_detection={enable_two_stage_detection}, reid={enable_reid}")
    
    def reset(self):
        """Reset the analyzer state."""
        self.total_frames = 0
        self.total_detections = 0
        self.valid_detections = 0
        self.filtered_detections = 0
        self.class_counts = {}
        self.person_identities = []
        self.frames = {}
        self.tracks = {}
        self.timeline = []
    
    def analyze_video(
        self, 
        video_path, 
        frame_interval=1, 
        start_frame=0, 
        end_frame=None, 
        save_video=False, 
        output_fps=None,
        enable_enhanced_filtering=None,
        enable_two_stage_detection=None,
        enable_weapon_detection=None,
        enable_interaction_detection=None
    ):
        """
        Analyze a video file.
        
        Parameters:
        -----------
        video_path : str
            Path to video file
        frame_interval : int
            Process every n-th frame
        start_frame : int
            Start processing from this frame
        end_frame : int, optional
            Stop processing at this frame
        save_video : bool
            Whether to save output video
        output_fps : float, optional
            FPS for output video
        enable_enhanced_filtering : bool, optional
            Override default for enhanced filtering
        enable_two_stage_detection : bool, optional
            Override default for two-stage detection
            
        Returns:
        --------
        Dict
            Analysis results
        """
        # Use provided values or fall back to instance defaults
        use_enhanced_filtering = enable_enhanced_filtering if enable_enhanced_filtering is not None else self.enable_enhanced_filtering
        use_two_stage_detection = enable_two_stage_detection if enable_two_stage_detection is not None else self.enable_two_stage_detection
        use_weapon_detection = enable_weapon_detection if enable_weapon_detection is not None else self.enable_weapon_detection
        use_interaction_detection = enable_interaction_detection if enable_interaction_detection is not None else self.enable_interaction_detection
        
        # Check if video exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video info
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if end_frame is None:
            end_frame = total_frames
        
        # Set up output video writer if needed
        output_video = None
        if save_video:
            if output_fps is None:
                output_fps = fps
            
            output_video_path = os.path.join(self.output_dir, "output_video.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_video = cv2.VideoWriter(output_video_path, fourcc, output_fps, (width, height))
        
        # Reset results
        self.reset()
        
        # Store video metadata
        self.video_metadata = {
            "path": video_path,
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": total_frames,
            "processed_frames": 0
        }
        
        # Skip to start frame
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Process frames
        print(f"Analyzing video: {video_path}")
        print(f"Total frames: {total_frames}, Processing every {frame_interval} frame(s)")
        print(f"Using enhanced filtering: {use_enhanced_filtering}, two-stage detection: {use_two_stage_detection}")
        
        processed_frames = 0
        analyzed_frames = 0
        frame_number = start_frame
        
        while frame_number < end_frame:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame if it's on the interval
            if processed_frames % frame_interval == 0:
                # Get timestamp
                timestamp = frame_number / fps
                
                # Detect objects using two-stage detection if enabled
                if use_two_stage_detection and hasattr(self, 'two_stage_detector'):
                    detections = self.two_stage_detector.detect(frame)
                else:
                    # Standard detection
                    detections = self.detector.detect(frame)
                
                # Apply enhanced filtering if enabled
                pre_filter_count = len(detections)
                if use_enhanced_filtering and hasattr(self, 'enhanced_filter'):
                    detections = self.enhanced_filter.filter_detections(
                        detections, 
                        frame, 
                        frame_number
                    )


                weapon_detections = []
                # Add weapon detections
                if use_weapon_detection and hasattr(self, 'weapon_detector') and self.weapon_detector is not None:
                    try:
                        if self.weapon_detector.model is None:
                            print("ERROR: Weapon detector model failed to load")
                        else:
                            print(f"Weapon detector ready using device: {self.weapon_detector.device}")
                            # Verify model classes
                            model_classes = self.weapon_detector.model.names if hasattr(self.weapon_detector.model, 'names') else {}
                            print(f"Model classes: {model_classes}")
                            
                            # Now run detection
                            weapon_detections = self.weapon_detector.detect(frame)
                            
                            # Print what was found
                            if weapon_detections:
                                print(f"Found {len(weapon_detections)} weapon detections in this frame!")
                                for det in weapon_detections:
                                    print(f"  - {det['class_name']}: {det['confidence']:.2f}")
                            
                            detections.extend(weapon_detections)
                    except Exception as e:
                        print(f"Error in weapon detection: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    if use_weapon_detection:
                        print("Weapon detection enabled but detector not available")
                        
                        detections.extend(weapon_detections)
                    
                    for det in weapon_detections:
                        if det.get('type') == 'weapon':
                            self.class_counts['weapon'] = self.class_counts.get('weapon', 0) + 1

                # Track objects BEFORE interaction detection
                tracked_objects = self.tracker.update(detections)

                # Now do interaction detection with the tracked objects
                if use_interaction_detection and hasattr(self, 'interaction_detector'):
                    interactions = self.interaction_detector.update(
                        frame_number, 
                        timestamp,
                        tracked_objects,
                        frame
                    )
                    
                    for interaction in interactions:
                        interaction_type = interaction.get('type', 'interaction')
                        self.class_counts[interaction_type] = self.class_counts.get(interaction_type, 0) + 1
                
                self.total_detections += pre_filter_count
                self.valid_detections += len(detections)
                self.filtered_detections += (pre_filter_count - len(detections))
                
                # Update class counts
                for det in detections:
                    class_name = det.get('class_name', 'Unknown')
                    if class_name in self.class_counts:
                        self.class_counts[class_name] += 1
                    else:
                        self.class_counts[class_name] = 1
                
                # Filter person detections
                person_detections = [d for d in detections if d.get('class_name', '').lower() == 'person']
                
                # Perform re-identification if enabled
                person_ids = []
                if self.enable_reid and len(person_detections) > 0:
                    try:
                        # Update ReID database and get person IDs
                        person_ids = self.reidentifier.update(frame, person_detections)
                        
                        # Associate person IDs with detections
                        for i, det in enumerate(person_detections):
                            if i < len(person_ids):
                                det['person_id'] = person_ids[i]
                                
                                # Add to person identities if it's a new person
                                if person_ids[i] not in [p.get('id') for p in self.person_identities]:
                                    self.person_identities.append({
                                        'id': person_ids[i],
                                        'metadata': {
                                            'first_seen_frame': frame_number,
                                            'first_seen_time': timestamp,
                                            'last_seen_frame': frame_number,
                                            'last_seen_time': timestamp,
                                            'appearances': 1
                                        }
                                    })
                                else:
                                    # Update existing person metadata
                                    for person in self.person_identities:
                                        if person.get('id') == person_ids[i]:
                                            person['metadata']['last_seen_frame'] = frame_number
                                            person['metadata']['last_seen_time'] = timestamp
                                            person['metadata']['appearances'] = person['metadata'].get('appearances', 0) + 1
                                            break
                    except Exception as e:
                        logger.error(f"ReID error (continuing anyway): {e}")
                
                # Track objects
                tracked_objects = self.tracker.update(detections)
                
                # Store frame results
                frame_result = {
                    'frame_number': frame_number,
                    'timestamp': timestamp,
                    'detections': detections,
                    'tracks': tracked_objects
                }
                
                self.frames[frame_number] = frame_result
                
                # Update tracks
                for track in tracked_objects:
                    track_id = track.get('id')
                    if track_id not in self.tracks:
                        self.tracks[track_id] = {
                            'id': track_id,
                            'class_name': track.get('class_name', 'Unknown'),
                            'first_frame': frame_number,
                            'first_time': timestamp,
                            'last_frame': frame_number,
                            'last_time': timestamp,
                            'detections': [track]
                        }
                    else:
                        self.tracks[track_id]['last_frame'] = frame_number
                        self.tracks[track_id]['last_time'] = timestamp
                        self.tracks[track_id]['detections'].append(track)
                
                # Create annotated frame
                annotated_frame = self._draw_detections_and_tracks(
                    frame, detections, tracked_objects, use_enhanced_filtering)
                
                # Save to output video
                if output_video is not None:
                    output_video.write(annotated_frame)
                
                analyzed_frames += 1
                
                # Print progress
                if analyzed_frames % 10 == 0:
                    filter_rate = self.filtered_detections / self.total_detections if self.total_detections > 0 else 0
                    remaining = (end_frame - frame_number) / frame_interval
                    elapsed = frame_number - start_frame
                    if elapsed > 0:
                        frames_per_second = analyzed_frames / (elapsed / fps)
                        eta_seconds = remaining / frames_per_second
                        print(f"Processed {frame_number}/{total_frames} frames ({analyzed_frames} analyzed, {frames_per_second:.1f} fps, "
                             f"filtered: {filter_rate:.1%}, ETA: {eta_seconds:.1f}s)")
            
            frame_number += 1
            processed_frames += 1
        
        # Release resources
        cap.release()
        if output_video is not None:
            output_video.release()
        
        # Update video metadata
        self.video_metadata['processed_frames'] = processed_frames
        
        # Save ReID database and visualizations
        if self.enable_reid:
            try:
                # Save ReID database
                reid_dir = os.path.join(self.output_dir, "reid_data")
                os.makedirs(reid_dir, exist_ok=True)
                
                # Save person database
                self.reidentifier.save_database(os.path.join(reid_dir, "person_database.json"))
                
                # Create person image directory
                person_img_dir = os.path.join(reid_dir, "person_images")
                os.makedirs(person_img_dir, exist_ok=True)
                
                # Save person images
                for person in self.person_identities:
                    person_id = person.get('id')
                    self.reidentifier.save_person_image(person_id, None, person_img_dir)
                
                # Create person gallery visualization
                self.reidentifier.visualize_database(os.path.join(self.output_dir, "person_gallery.jpg"))
            except Exception as e:
                logger.error(f"Error saving ReID data: {e}")
        
        # Save analysis log
        analysis_log = {
            'timestamp': datetime.now().isoformat(),
            'video_path': video_path,
            'total_frames': total_frames,
            'processed_frames': processed_frames,
            'analyzed_frames': analyzed_frames,
            'total_detections': self.total_detections,
            'valid_detections': self.valid_detections,
            'filtered_detections': self.filtered_detections,
            'filter_rate': self.filtered_detections / self.total_detections if self.total_detections > 0 else 0,
            'person_identities': len(self.person_identities),
            'class_counts': self.class_counts,
            'enhanced_filtering': use_enhanced_filtering,
            'two_stage_detection': use_two_stage_detection
        }
        
        with open(os.path.join(self.output_dir, "analysis_log.json"), 'w') as f:
            json.dump(analysis_log, f, indent=2)
        
        # Prepare and return results
        results = {
            'video_path': video_path,
            'video_metadata': self.video_metadata,
            'total_detections': self.total_detections,
            'valid_detections': self.valid_detections,
            'filtered_detections': self.filtered_detections,
            'class_counts': self.class_counts,
            'total_unique_objects': len(self.tracks),
            'person_identities': self.person_identities,
            'tracks': self.tracks,
            'frames': self.frames,
            'timeline': self.timeline,
            'detector': f"yolov8{self.detector.model_size}",
            'enhanced_filtering': use_enhanced_filtering,
            'two_stage_detection': use_two_stage_detection
        }
        
        # Log summary stats
        logger.info(f"Analysis complete: {self.valid_detections} valid detections, "
                   f"{self.filtered_detections} filtered out ({filter_rate:.1%} reduction)")
        
        return results
    
    def generate_analysis_report(self, results):
        """
        Generate a report from analysis results.
        
        Parameters:
        -----------
        results : Dict
            Analysis results
            
        Returns:
        --------
        Dict
            Analysis report
        """
        # Ensure person identities exist
        if 'person_identities' not in results or not results['person_identities']:
            # Try to create person identities from person tracks
            person_identities = []
            person_id = 1
            for track_id, track in results.get('tracks', {}).items():
                if track.get('class_name', '').lower() == 'person':
                    person = {
                        'id': person_id,
                        'metadata': {
                            'appearances': len(track.get('detections', [])),
                            'first_seen_frame': track.get('first_frame', 0),
                            'last_seen_frame': track.get('last_frame', 0),
                            'first_seen_time': track.get('first_time', 0),
                            'last_seen_time': track.get('last_time', 0),
                            'created_from_track': True
                        }
                    }
                    person_identities.append(person)
                    person_id += 1
            
            # Update results with extracted persons if any found
            if person_identities:
                results['person_identities'] = person_identities
                print(f"Created {len(person_identities)} persons from tracks")

        # Create report structure
        report = {
            'video_path': results.get('video_path', ''),
            'timestamp': datetime.now().isoformat(),
            'video_metadata': results.get('video_metadata', {}),
            'total_detections': results.get('total_detections', 0),
            'valid_detections': results.get('valid_detections', 0),
            'filtered_detections': results.get('filtered_detections', 0),
            'total_unique_objects': results.get('total_unique_objects', 0),
            'class_counts': results.get('class_counts', {}),
            'person_identities': results.get('person_identities', []),
            'timeline': results.get('timeline', []),
            'false_positive_reduction': {
                'enabled': results.get('enhanced_filtering', False),
                'two_stage_detection': results.get('two_stage_detection', False),
                'filtered_count': results.get('filtered_detections', 0),
                'reduction_rate': results.get('filtered_detections', 0) / results.get('total_detections', 1) if results.get('total_detections', 0) > 0 else 0
            }
        }
        
        # Add detector info
        if 'detector' in results:
            report['detector'] = results['detector']
        
        # Save report
        report_path = os.path.join(self.output_dir, "analysis_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report
    
    def _draw_detections_and_tracks(self, frame, detections, tracks=None, show_filtering=False):
        """
        Draw detections and tracks on a frame.
        
        Parameters:
        -----------
        frame : np.ndarray
            Input frame
        detections : List[Dict]
            List of detections
        tracks : List[Dict], optional
            List of tracks
        show_filtering : bool
            Whether to show filtering info
            
        Returns:
        --------
        np.ndarray
            Frame with visualizations
        """
        # Make a copy of the frame
        vis_frame = frame.copy()
        
        # Draw detections
        for det in detections:
            # Get box coordinates
            box = det.get('box', [0, 0, 0, 0])
            x1, y1, x2, y2 = [int(c) for c in box]
            
            # Get class name and confidence
            class_name = det.get('class_name', 'Unknown')
            conf = det.get('confidence', 0.0)
            
            # Choose color based on class
            if class_name.lower() == 'person':
                color = (0, 255, 0)  # Green for persons
            else:
                color = (255, 0, 0)  # Red for other objects
            
            # Draw box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with confidence and validation score if available
            label = f"{class_name} {conf:.2f}"
            
            # Add validation score if available
            if 'validation_score' in det:
                label += f" (V:{det['validation_score']:.2f})"
                
            # Draw label
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(vis_frame, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
            cv2.putText(vis_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw tracks if provided
        if tracks:
            for track in tracks:
                # Get box
                box = track.get('box', [0, 0, 0, 0])
                x1, y1, x2, y2 = [int(c) for c in box]
                
                # Get ID
                track_id = track.get('id', -1)
                
                # Draw ID at the top of the box
                cv2.putText(vis_frame, f"ID:{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Draw box with unique color based on ID
                color_val = (track_id * 5) % 256
                color = ((color_val * 3) % 256, (color_val * 7) % 256, (color_val * 11) % 256)
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw trajectories
        trajectories = self.tracker.get_trajectories()
        for track_id, points in trajectories.items():
            if len(points) < 2:
                continue
                
            # Create unique color based on track ID
            color_val = (track_id * 5) % 256
            color = ((color_val * 3) % 256, (color_val * 7) % 256, (color_val * 11) % 256)
            
            # Draw trajectory line
            for i in range(1, len(points)):
                # Convert to int coordinates
                pt1 = (int(points[i-1][0]), int(points[i-1][1]))
                pt2 = (int(points[i][0]), int(points[i][1]))
                cv2.line(vis_frame, pt1, pt2, color, 2)
        
        # Draw filtering info if enabled
        if show_filtering:
            filter_rate = self.filtered_detections / self.total_detections if self.total_detections > 0 else 0
            info_text = f"Enhanced Filtering: {self.filtered_detections} filtered ({filter_rate:.1%})"
            cv2.putText(vis_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return vis_frame