import numpy as np
from enum import Enum
import logging
from typing import List, Dict, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrackState(Enum):
    """
    Track states in the tracking process.
    """
    NEW = 0
    TRACKED = 1
    LOST = 2
    CONFIRMED = 3
    DELETED = 4

class Track:
    """
    Track class for a single object.
    """
    
    def __init__(self, detection: Dict, track_id: int):
        """
        Initialize a track from a detection.
        
        Parameters:
        -----------
        detection : Dict
            Detection dictionary
        track_id : int
            Unique track ID
        """
        self.track_id = track_id
        self.box = detection.get('box', [0, 0, 0, 0])
        self.box_center = detection.get('box_center', [0, 0])
        self.confidence = detection.get('confidence', 0.0)
        self.class_id = detection.get('class_id', -1)
        self.class_name = detection.get('class_name', '')
        self.state = TrackState.NEW
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.history = []
        self.features = []
        self.detections = [detection]
        
        # Copy any other attributes from the detection
        for key, value in detection.items():
            if key not in ['box', 'box_center', 'confidence', 'class_id', 'class_name']:
                setattr(self, key, value)
    
    def update(self, detection: Dict):
        """
        Update the track with a new detection.
        
        Parameters:
        -----------
        detection : Dict
            Detection dictionary
        """
        self.box = detection.get('box', self.box)
        self.box_center = detection.get('box_center', self.box_center)
        self.confidence = detection.get('confidence', self.confidence)
        self.hits += 1
        self.time_since_update = 0
        self.state = TrackState.TRACKED
        self.history.clear()  # Clear history when updated
        self.detections.append(detection)
        
        # Update any other attributes from the detection
        for key, value in detection.items():
            if key not in ['box', 'box_center', 'confidence', 'class_id', 'class_name']:
                setattr(self, key, value)
    
    def mark_missed(self):
        """Mark the track as missed (not detected)."""
        if self.state == TrackState.TRACKED:
            self.state = TrackState.LOST
        
        self.time_since_update += 1
    
    def predict(self):
        """Predict the next state of the track (for future implementations)."""
        # For now, just add the current box to history
        self.history.append(self.box.copy())
    
    def is_confirmed(self) -> bool:
        """Check if the track is confirmed."""
        return self.state == TrackState.CONFIRMED
    
    def is_tracked(self) -> bool:
        """Check if the track is being actively tracked."""
        return self.state == TrackState.TRACKED
    
    def is_deleted(self) -> bool:
        """Check if the track is deleted."""
        return self.state == TrackState.DELETED
    
    def is_lost(self) -> bool:
        """Check if the track is lost."""
        return self.state == TrackState.LOST

class ObjectTracker:
    """
    Simple object tracker using IoU matching.
    """
    
    def __init__(self, max_age: int = 60, min_hits: int = 1, iou_threshold: float = 0.2):
        """
        Initialize the tracker.
        
        Parameters:
        -----------
        max_age : int
            Maximum number of frames to keep a track alive without matching
        min_hits : int
            Minimum number of detection matches needed to confirm a track
        iou_threshold : float
            IoU threshold for matching detections to tracks
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.track_id = 0
    
    def _iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Calculate IoU between two bounding boxes.
        
        Parameters:
        -----------
        box1, box2 : np.ndarray
            Bounding boxes in format [x1, y1, x2, y2]
            
        Returns:
        --------
        float
            IoU value
        """
        # Get the coordinates of the intersection rectangle
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # Calculate area of intersection
        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Calculate area of both bounding boxes
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Calculate IoU
        iou = intersection_area / float(box1_area + box2_area - intersection_area + 1e-6)
        
        return iou
    
    def _calculate_cost_matrix(self, detections: List[Dict]) -> np.ndarray:
        """
        Calculate cost matrix for assignment problem.
        
        Parameters:
        -----------
        detections : List[Dict]
            List of detections
            
        Returns:
        --------
        np.ndarray
            Cost matrix where cost[i, j] = 1 - IoU between detection i and track j
        """
        if not self.tracks or not detections:
            return np.zeros((len(detections), len(self.tracks)))
        
        # Initialize cost matrix
        cost_matrix = np.zeros((len(detections), len(self.tracks)))
        
        # Calculate IoU for each detection-track pair
        for i, detection in enumerate(detections):
            det_box = np.array(detection.get('box', [0, 0, 0, 0]))
            det_class = detection.get('class_id', -1)
            
            for j, track in enumerate(self.tracks):
                # Only match detections and tracks of the same class
                if track.class_id != det_class:
                    cost_matrix[i, j] = float('inf')
                    continue
                
                track_box = np.array(track.box)
                iou = self._iou(det_box, track_box)
                
                # Cost is 1 - IoU (lower cost = better match)
                cost_matrix[i, j] = 1 - iou
        
        return cost_matrix
    
    def _solve_assignment_problem(self, cost_matrix, max_distance):
        """
        Solve the linear assignment problem.
        
        Parameters:
        -----------
        cost_matrix : np.ndarray
            Cost matrix of shape (num_detections, num_tracks)
        max_distance : float
            Maximum allowed distance
            
        Returns:
        --------
        matches : list of tuple
            Matched indices (detection_idx, track_idx)
        unmatched_detections : list of int
            Indices of unmatched detections
        unmatched_tracks : list of int
            Indices of unmatched tracks
        """
        try:
            if cost_matrix.size == 0:
                return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
                
            # Mask out assignments with distances exceeding max_distance
            masked_cost_matrix = np.copy(cost_matrix)
            masked_cost_matrix[cost_matrix > max_distance] = 1e6  # Very high cost instead of masking
            
            # Use scipy's linear_sum_assignment for increased robustness
            from scipy.optimize import linear_sum_assignment
            detection_indices, track_indices = linear_sum_assignment(masked_cost_matrix)
            
            # Filter out assignments with high cost
            matches = []
            unmatched_detections = list(range(cost_matrix.shape[0]))
            unmatched_tracks = list(range(cost_matrix.shape[1]))
            
            for d_idx, t_idx in zip(detection_indices, track_indices):
                if cost_matrix[d_idx, t_idx] <= max_distance:
                    matches.append((d_idx, t_idx))
                    if d_idx in unmatched_detections:
                        unmatched_detections.remove(d_idx)
                    if t_idx in unmatched_tracks:
                        unmatched_tracks.remove(t_idx)
            
            return matches, unmatched_detections, unmatched_tracks
        except Exception as e:
            import traceback
            logger.error(f"Assignment problem error: {e}")
            logger.error(traceback.format_exc())
            # Return a safe fallback that won't crash but won't perform well
            return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
    
    def update(self, detections: List[Dict], frame=None, frame_number=None, timestamp=None) -> List[Dict]:
        """
        Update tracks with new detections.
        
        Parameters:
        -----------
        detections : List[Dict]
            List of detections
            
        Returns:
        --------
        List[Dict]
            List of track dictionaries
        """
        # Predict new locations of tracks
        for track in self.tracks:
            track.predict()
        
        # Calculate cost matrix
        cost_matrix = self._calculate_cost_matrix(detections)
        
        # Solve assignment problem
        matches, unmatched_det_indices, unmatched_track_indices = self._solve_assignment_problem(
            cost_matrix, 1 - self.iou_threshold)
        
        # Update matched tracks
        for det_idx, track_idx in matches:
            self.tracks[track_idx].update(detections[det_idx])
        
        # Mark unmatched tracks as missed
        for track_idx in unmatched_track_indices:
            self.tracks[track_idx].mark_missed()
        
        # Initialize new tracks
        for det_idx in unmatched_det_indices:
            # Create a new track
            new_track = Track(detections[det_idx], self.track_id)
            new_track.state = TrackState.CONFIRMED
            self.tracks.append(new_track)
            self.track_id += 1
        
        # Remove old tracks
        self.tracks = [track for track in self.tracks if track.time_since_update <= self.max_age]
        
        # Update track states
        for track in self.tracks:
            if track.hits >= self.min_hits and track.state != TrackState.CONFIRMED:
                track.state = TrackState.CONFIRMED
        
        # Return active tracks as dictionaries
        return self.get_tracks()
    
    def get_tracks(self) -> List[Dict]:
        """
        Get current tracks as a list of dictionaries.
        
        Returns:
        --------
        List[Dict]
            List of track dictionaries
        """
        tracks = []
        
        for track in self.tracks:
            # Only include active tracks
            if track.state in [TrackState.TRACKED, TrackState.CONFIRMED]:
                track_dict = {
                    'id': track.track_id,
                    'box': track.box,
                    'box_center': track.box_center,
                    'confidence': track.confidence,
                    'class_id': track.class_id,
                    'class_name': track.class_name,
                    'state': track.state.name,
                    'age': track.age,
                    'hits': track.hits,
                    'time_since_update': track.time_since_update
                }
                tracks.append(track_dict)
        
        return tracks
    
    def reset(self):
        """Reset the tracker."""
        self.tracks = []
        self.track_id = 0

    # Add this method to your ObjectTracker class:

    def get_trajectories(self):
        """
        Get trajectories for all tracks.
        
        Returns:
        --------
        Dict[int, List[List[float]]]
            Dictionary mapping track_id to list of box_center positions
        """
        trajectories = {}
        
        for track in self.tracks:
            if track.state in [TrackState.TRACKED, TrackState.CONFIRMED]:
                # Create trajectory if it doesn't exist
                if track.track_id not in trajectories:
                    trajectories[track.track_id] = []
                
                # Add current position
                trajectories[track.track_id].append(track.box_center)
                
                # Add history positions if available
                for box in track.history:
                    # Calculate center from box
                    box_center_x = (box[0] + box[2]) / 2
                    box_center_y = (box[1] + box[3]) / 2
                    center = [float(box_center_x), float(box_center_y)]
                    
                    # Add to trajectory if not already present
                    if center not in trajectories[track.track_id]:
                        trajectories[track.track_id].insert(0, center)
        
        return trajectories