from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import cv2
import time
import logging

logger = logging.getLogger(__name__)

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
        
        logger.info(f"Initialized TwoStageDetector with validation_threshold={validation_threshold}, " 
                   f"enable_second_stage={enable_second_stage}")
    
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
            logger.info(f"Loaded secondary validation models from {model_dir}")
        except Exception as e:
            logger.warning(f"Could not load secondary models: {e}")
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
            # Always keep persons with decent confidence to avoid missing people
            if confidence > 0.85 or class_name not in self.class_validation_map or \
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
        
        # Log validation results periodically
        if self.validation_count % 20 == 0:
            logger.debug(f"Validation stats: {validation_metadata['validated']} validated, "
                       f"{validation_metadata['rejected']} rejected in {validation_time:.3f}s")
        
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
        try:
            resized_roi = cv2.resize(roi, (128, 128))
            
            # Convert to grayscale if needed
            if len(resized_roi.shape) == 3:
                gray_roi = cv2.cvtColor(resized_roi, cv2.COLOR_BGR2GRAY)
            else:
                gray_roi = resized_roi
            
            # Extract SIFT features
            keypoints, descriptors = self.feature_extractor.detectAndCompute(gray_roi, None)
            
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
                
                try:
                    # Use FLANN matcher for efficient feature matching
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                    search_params = dict(checks=50)
                    flann = cv2.FlannBasedMatcher(index_params, search_params)
                    
                    # Make sure descriptors are in the correct format (CV_32F)
                    if descriptors.dtype != np.float32:
                        descriptors = np.float32(descriptors)
                    if exemplar_descriptors.dtype != np.float32:
                        exemplar_descriptors = np.float32(exemplar_descriptors)
                    
                    matches = flann.knnMatch(descriptors, exemplar_descriptors, k=2)
                    
                    # Apply ratio test
                    good_matches = 0
                    total_matches = 0
                    for m_n in matches:
                        if len(m_n) == 2:  # Ensure we have 2 matches for ratio test
                            m, n = m_n
                            if m.distance < 0.7 * n.distance:
                                good_matches += 1
                            total_matches += 1
                    
                    # Calculate similarity score based on good matches ratio
                    if total_matches > 0:
                        similarity = good_matches / total_matches
                        max_similarity = max(max_similarity, similarity)
                except Exception as e:
                    logger.warning(f"Error in feature matching: {e}")
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
            logger.warning(f"Error in feature-based validation: {e}")
            # On error, return conservative result
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