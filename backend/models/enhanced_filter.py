from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

class EnhancedFilter:
    
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
        self.class_confidence_thresholds = class_confidence_thresholds or {
            'person': 0.65,
            'car': 0.7,
            'truck': 0.7,
            'bicycle': 0.6,
            'motorcycle': 0.6,
            'bus': 0.7,
            'knife': 0.75,
            'gun': 0.8,
            'default': 0.6
        }
        
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
        
        self.class_aspect_ratios = class_aspect_ratios or {
            'person': (0.2, 0.7),   # Height typically greater than width
            'car': (1.2, 3.5),      # Width typically greater than height
            'truck': (1.2, 3.0),    # Width typically greater than height
            'bicycle': (0.8, 2.0),  # Various aspect ratios possible
            'motorcycle': (0.8, 2.0),
            'bus': (1.0, 2.5),
            'knife': 0.4,  # Lower from 0.65
            'gun': 0.4,    # Lower from 0.7
            'pistol': 0.4, # Add this
            'rifle': 0.4,  # Add this
            'default': (0.5, 2.0)   # Default aspect ratio range
        }
        
        self.motion_threshold = motion_threshold
        self.temporal_consistency_frames = temporal_consistency_frames
        self.iou_threshold = iou_threshold
        self.frame_height = frame_height
        self.frame_width = frame_width
        
        self.previous_detections = []  # List of previous detections by frame
        
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, 
            varThreshold=16, 
            detectShadows=False
        )
        
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
        
        if frame is not None:
            self.frame_height, self.frame_width = frame.shape[:2]
        
        detections = self._filter_by_confidence(detections)
        confidence_filtered = len(detections)
        
        detections = self._filter_by_size_and_ratio(detections)
        size_filtered = len(detections)
        
        detections = self._apply_nms(detections)
        nms_filtered = len(detections)
        
        if frame is not None:
            detections = self._validate_with_motion(detections, frame)
            motion_filtered = len(detections)
        else:
            motion_filtered = nms_filtered
        
        detections = self._check_temporal_consistency(detections, frame_idx)
        temporal_filtered = len(detections)
        
        detections = self._stabilize_detections(detections, frame_idx)
        
        self._update_detection_history(detections, frame_idx)
        
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
        filtered = []
        for det in detections:
            class_name = det['class_name']
            confidence = det['confidence']
            
            threshold = self.class_confidence_thresholds.get(
                class_name, 
                self.class_confidence_thresholds['default']
            )
            
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
            
            size_ok = (
                width >= min_size[0] and 
                height >= min_size[1] and
                width <= max_size[0] and 
                height <= max_size[1]
            )
            
            aspect_ratio = width / height if height > 0 else 0
            aspect_ok = aspect_range[0] <= aspect_ratio <= aspect_range[1]
            
            y_pos_factor = (box[3] / self.frame_height) * 0.8 + 0.2  # Scale factor 0.2-1.0
            
            min_size_adjusted = (min_size[0] * y_pos_factor, min_size[1] * y_pos_factor)
            size_ok_adjusted = (
                width >= min_size_adjusted[0] and 
                height >= min_size_adjusted[1]
            )
            
            if size_ok_adjusted and aspect_ok:
                filtered.append(det)
            elif class_name == 'person' and width >= min_size_adjusted[0] * 0.8 and height >= min_size_adjusted[1] * 0.8:
                det['confidence'] *= 0.9  # Slightly reduce confidence for out-of-ratio persons
                filtered.append(det)
        
        return filtered
    
    def _apply_nms(self, detections: List[Dict]) -> List[Dict]:
        """Apply non-maximum suppression to reduce duplicate detections."""
        if not detections:
            return []
        
        detections_by_class = {}
        for det in detections:
            class_name = det['class_name']
            if class_name not in detections_by_class:
                detections_by_class[class_name] = []
            detections_by_class[class_name].append(det)
        
        filtered_detections = []
        for class_name, class_dets in detections_by_class.items():
            # Extract boxes and scores
            boxes = [d['box'] for d in class_dets]
            scores = [d['confidence'] for d in class_dets]
            
            boxes_np = np.array(boxes, dtype=np.float32)
            scores_np = np.array(scores, dtype=np.float32)
            
            indices = self._nms(boxes_np, scores_np, self.iou_threshold)
            
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
        
        areas = w * h
        
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])
            
            w_intersection = np.maximum(0.0, xx2 - xx1)
            h_intersection = np.maximum(0.0, yy2 - yy1)
            intersection = w_intersection * h_intersection
            
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)
            
            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]
            
        return keep
    
    def _validate_with_motion(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        fg_mask = self.bg_subtractor.apply(frame)
        
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)
        
        validated_detections = []
        for det in detections:
            box = det['box']
            x1, y1, x2, y2 = map(int, box)
            
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1] - 1, x2)
            y2 = min(frame.shape[0] - 1, y2)
            
            if x1 >= x2 or y1 >= y2:
                continue
            
            roi = fg_mask[y1:y2, x1:x2]
            
            if roi.size > 0:  # Ensure ROI is not empty
                motion_percentage = np.count_nonzero(roi) / roi.size
            else:
                motion_percentage = 0
            
            is_static_object = det['class_name'] in ['car', 'truck', 'bus'] and motion_percentage < 0.1
            
            is_person = det['class_name'] == 'person'
            
            det['motion_percentage'] = float(motion_percentage)
            
            if motion_percentage >= self.motion_threshold or is_static_object:
                validated_detections.append(det)
            elif is_person and motion_percentage >= self.motion_threshold * 0.4:
                validated_detections.append(det)
            elif det.get('type') == 'weapon' and motion_percentage >= self.motion_threshold * 0.3:
                validated_detections.append(det)
            elif det['confidence'] > 0.85:
                validated_detections.append(det)
        
        return validated_detections
    
    def _check_temporal_consistency(
        self, 
        detections: List[Dict], 
        frame_idx: int
    ) -> List[Dict]:
        if not self.previous_detections:
            return detections
        
        consistent_detections = []
        
        for det in detections:
            # Always keep high-confidence detections
            if det['confidence'] > 0.85:
                consistent_detections.append(det)
                continue
            
            # Always keep persons (more lenient)
            if det['class_name'] == 'person' and det['confidence'] > 0.5:
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
