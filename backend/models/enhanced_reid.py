import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
import json
import time
from typing import List, Dict, Tuple, Optional, Union, Any
from datetime import datetime
import matplotlib.pyplot as plt
import faiss
import logging

from .person_reid import PersonFeatureExtractor, PersonDatabase
from .reid_models import OSNet, MGNModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedFeatureExtractor(PersonFeatureExtractor):
    """
    Enhanced feature extractor that supports fine-tuned models.
    """
    
    def __init__(
        self,
        model_type: str = 'osnet',
        model_path: Optional[str] = None,
        feature_dim: int = 512,
        num_classes: int = 751,  # Default for Market-1501 dataset
        use_gpu: bool = None
    ):
        """
        Initialize the enhanced feature extractor.
        
        Parameters:
        -----------
        model_type : str
            Model architecture ('osnet', 'mgn', 'resnet50')
        model_path : str, optional
            Path to fine-tuned model checkpoint
        feature_dim : int
            Dimension of output features
        num_classes : int
            Number of identity classes (for model initialization)
        use_gpu : bool, optional
            Whether to use GPU. If None, automatically detects.
        """
        # Determine device
        if use_gpu is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        
        # Store config
        self.model_type = model_type.lower()
        self.feature_dim = feature_dim
        
        # Image preprocessing
        self.transform = T.Compose([
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load model based on type
        if self.model_type == 'osnet':
            self.model = OSNet(
                num_classes=num_classes,
                feature_dim=feature_dim,
                blocks=[2, 2, 2],
                channels=[64, 256, 384, 512],
                use_attention=True
            )
            logger.info(f"Created OSNet model with feature dimension {feature_dim}")
        elif self.model_type == 'mgn':
            self.model = MGNModel(
                num_classes=num_classes,
                feature_dim=feature_dim // 6,  # MGN produces 6 feature blocks
                pretrained=True
            )
            logger.info(f"Created MGN model with feature dimension {feature_dim}")
        elif self.model_type == 'resnet50':
            # For compatibility with base class
            super().__init__(
                model_name='resnet50',
                feature_dim=feature_dim,
                pretrained=True,
                use_gpu=use_gpu
            )
            return
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load fine-tuned weights if provided
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, path: str) -> None:
        """Load model weights from checkpoint."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model checkpoint not found: {path}")
        
        # Load checkpoint
        checkpoint = torch.load(path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            # Our trainer format
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            # Common format
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            # Direct state dict
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        logger.info(f"Loaded model weights from {path}")
    
    def extract_features(self, img: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Extract features from a person image.
        
        Parameters:
        -----------
        img : np.ndarray or PIL.Image
            Input image (OpenCV BGR or PIL RGB format)
            
        Returns:
        --------
        np.ndarray
            Feature vector
        """
        # Skip if we're using the base class implementation
        if hasattr(self, 'model_name') and self.model_type == 'resnet50':
            return super().extract_features(img)
        
        # Convert OpenCV image to PIL if necessary
        if isinstance(img, np.ndarray):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
        
        # Preprocess image
        img_tensor = self.transform(img).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        
        # Extract features
        with torch.no_grad():
            if self.model_type == 'osnet':
                features = self.model(img_tensor)
            elif self.model_type == 'mgn':
                features = self.model(img_tensor)
            
            # Convert to numpy and normalize
            features = features.cpu().numpy()
            features = features.reshape(-1)  # Flatten
            
            # Normalize
            features = features / np.linalg.norm(features)
        
        return features
    
    def batch_extract_features(self, images: List[Union[np.ndarray, Image.Image]]) -> np.ndarray:
        """
        Extract features from multiple person images.
        
        Parameters:
        -----------
        images : List[np.ndarray or PIL.Image]
            List of input images
            
        Returns:
        --------
        np.ndarray
            Array of feature vectors (n_images, feature_dim)
        """
        # Skip if we're using the base class implementation
        if hasattr(self, 'model_name') and self.model_type == 'resnet50':
            return super().batch_extract_features(images)
        
        if not images:
            return np.array([])
        
        # Process batch of images
        batch_tensors = []
        for img in images:
            # Convert OpenCV image to PIL if necessary
            if isinstance(img, np.ndarray):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
            
            # Preprocess image
            tensor = self.transform(img).unsqueeze(0)
            batch_tensors.append(tensor)
        
        # Concatenate tensors
        batch_tensor = torch.cat(batch_tensors, dim=0)
        batch_tensor = batch_tensor.to(self.device)
        
        # Extract features
        with torch.no_grad():
            if self.model_type == 'osnet':
                features = self.model(batch_tensor)
            elif self.model_type == 'mgn':
                features = self.model(batch_tensor)
            
            # Convert to numpy
            features = features.cpu().numpy()
            
            # Normalize each feature vector
            for i in range(features.shape[0]):
                features[i] = features[i] / np.linalg.norm(features[i])
        
        return features
    
    def predict_attributes(self, img: Union[np.ndarray, Image.Image]) -> Dict[str, np.ndarray]:
        """
        Predict attributes for a person image (if available).
        
        Parameters:
        -----------
        img : np.ndarray or PIL.Image
            Input image
            
        Returns:
        --------
        Dict[str, np.ndarray]
            Dictionary of attribute predictions or empty dict if not available
        """
        # Check if we have an attribute predictor
        if not hasattr(self, 'attribute_predictor') or self.attribute_predictor is None:
            return {}
        
        # Convert OpenCV image to PIL if necessary
        if isinstance(img, np.ndarray):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
        
        # Preprocess image
        img_tensor = self.transform(img).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        
        # Extract features
        with torch.no_grad():
            if self.model_type == 'osnet':
                features = self.model(img_tensor)
            elif self.model_type == 'mgn':
                features = self.model(img_tensor)
            
            # Predict attributes
            attr_predictions = self.attribute_predictor(features)
            
            # Convert to numpy
            result = {}
            for attr_name, pred in attr_predictions.items():
                if pred.size(1) == 2:  # Binary attribute
                    result[attr_name] = torch.sigmoid(pred).cpu().numpy()
                else:  # Multi-class attribute
                    result[attr_name] = torch.softmax(pred, dim=1).cpu().numpy()
            
            return result


class EnhancedPersonReidentifier:
    """
    Enhanced person re-identification system with support for:
    - Fine-tuned models
    - Attribute-based search
    - Clothing color matching
    - Improved multi-camera tracking
    """
    
    def __init__(
        self,
        model_type: str = 'osnet',
        model_path: Optional[str] = None,
        feature_dim: int = 512,
        similarity_threshold: float = 0.6,
        use_attributes: bool = False,
        attribute_model_path: Optional[str] = None,
        use_gpu: bool = None
    ):
        """
        Initialize the enhanced re-identification system.
        
        Parameters:
        -----------
        model_type : str
            Model architecture ('osnet', 'mgn', 'resnet50')
        model_path : str, optional
            Path to fine-tuned model checkpoint
        feature_dim : int
            Dimension of feature vectors
        similarity_threshold : float
            Threshold for considering two features as matching (0-1)
        use_attributes : bool
            Whether to use attribute-based matching
        attribute_model_path : str, optional
            Path to attribute model checkpoint
        use_gpu : bool, optional
            Whether to use GPU
        """
        # Create feature extractor
        self.feature_extractor = EnhancedFeatureExtractor(
            model_type=model_type,
            model_path=model_path,
            feature_dim=feature_dim,
            use_gpu=use_gpu
        )
        
        # Create person database
        self.database = PersonDatabase(
            feature_dim=feature_dim,
            similarity_threshold=similarity_threshold,
            use_gpu=use_gpu
        )
        
        # Configuration
        self.similarity_threshold = similarity_threshold
        self.use_attributes = use_attributes
        self.model_type = model_type
        
        # Storage for unmatched detections that might form a new identity
        self.pending_identities = {}
        self.min_detections_for_new_id = 3
        
        # For tracking processing statistics
        self.stats = {
            'processed_detections': 0,
            'matched_identities': 0,
            'new_identities': 0,
            'processing_time': 0
        }
        
        # Load attribute model if specified
        self.attribute_predictor = None
        if use_attributes and attribute_model_path:
            self._load_attribute_model(attribute_model_path)
    
    def _load_attribute_model(self, path: str):
        """Load attribute prediction model."""
        pass  # Implement if your attribute model is separate
    
    def process_detections(
        self,
        detections: List[Dict],
        frame: np.ndarray,
        frame_idx: int,
        timestamp: float
    ) -> List[Dict]:
        """
        Process person detections to assign identities.
        
        Parameters:
        -----------
        detections : List[Dict]
            List of person detections
        frame : np.ndarray
            Full video frame
        frame_idx : int
            Current frame index
        timestamp : float
            Current timestamp
            
        Returns:
        --------
        List[Dict]
            Detections with assigned identities
        """
        start_time = time.time()
        
        # Filter only person detections
        person_detections = [d for d in detections if d.get('class_name') == 'person']
        
        if not person_detections:
            return detections  # No persons to process
        
        # Extract crops and features
        crops = []
        for det in person_detections:
            box = det['box']
            x1, y1, x2, y2 = map(int, box)
            
            # Ensure coordinates are within image boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            # Extract person crop
            crop = frame[y1:y2, x1:x2]
            
            # Skip if crop is too small
            if crop.shape[0] < 10 or crop.shape[1] < 10:
                crops.append(None)
                continue
                
            crops.append(crop)
        
        # Extract features for all valid crops
        valid_crops = [crop for crop in crops if crop is not None]
        valid_indices = [i for i, crop in enumerate(crops) if crop is not None]
        
        if not valid_crops:
            return detections  # No valid crops
            
        features = self.feature_extractor.batch_extract_features(valid_crops)
        
        # Extract attributes if enabled
        attributes = {}
        if self.use_attributes:
            for i, crop in enumerate(valid_crops):
                attr = self.feature_extractor.predict_attributes(crop)
                if attr:
                    attributes[valid_indices[i]] = attr
        
        # Assign features back to detections
        for idx, feat_idx in enumerate(valid_indices):
            person_detections[feat_idx]['reid_features'] = features[idx]
            
            # Add attributes if available
            if feat_idx in attributes:
                person_detections[feat_idx]['attributes'] = attributes[feat_idx]
        
        # Match with database
        for i, det in enumerate(person_detections):
            # Skip detections without features
            if 'reid_features' not in det:
                continue
                
            # Try to match with existing identities
            matches = self.database.find_matches(det['reid_features'])
            
            if matches:
                # Found a match
                best_match = matches[0]
                det['person_id'] = best_match['id']
                det['reid_score'] = best_match['similarity']
                
                # Get additional information from database
                person = self.database.get_person(best_match['id'])
                if person and 'metadata' in person:
                    det['person_metadata'] = person['metadata']
                
                # Update metadata
                self.database.update_metadata(
                    best_match['id'],
                    {
                        'last_seen_frame': frame_idx,
                        'last_seen_time': timestamp,
                        'appearances': self.database.get_person(best_match['id'])
                                          .get('metadata', {})
                                          .get('appearances', 0) + 1
                    }
                )
                
                # Update appearance attributes if available
                if 'attributes' in det:
                    # Store averaged attributes
                    current_attrs = person.get('metadata', {}).get('attributes', {})
                    for attr_name, attr_val in det['attributes'].items():
                        if attr_name not in current_attrs:
                            current_attrs[attr_name] = attr_val
                        else:
                            # Running average (simple for now)
                            current_attrs[attr_name] = 0.7 * current_attrs[attr_name] + 0.3 * attr_val
                    
                    # Update metadata
                    self.database.update_metadata(
                        best_match['id'],
                        {'attributes': current_attrs}
                    )
                
                self.stats['matched_identities'] += 1
            else:
                # No match found, handle as potential new identity
                self._handle_unmatched_detection(det, crops[i], frame_idx, timestamp)
        
        # Update statistics
        self.stats['processed_detections'] += len(person_detections)
        self.stats['processing_time'] += time.time() - start_time
        
        return detections
    
    def _handle_unmatched_detection(
        self,
        detection: Dict,
        crop: np.ndarray,
        frame_idx: int,
        timestamp: float
    ) -> None:
        """
        Handle an unmatched detection - may become a new identity.
        
        Parameters:
        -----------
        detection : Dict
            Detection information
        crop : np.ndarray
            Image crop of the person
        frame_idx : int
            Current frame index
        timestamp : float
            Current timestamp
        """
        # Use tracking ID as temporary identifier
        track_id = detection.get('object_id')
        
        if track_id is None:
            return  # Can't track without ID
        
        # Get or create pending identity entry
        if track_id not in self.pending_identities:
            self.pending_identities[track_id] = {
                'detections': [],
                'features': [],
                'crops': [],
                'attributes': [],
                'first_seen': frame_idx,
                'timestamp': timestamp
            }
        
        # Add to pending identity
        self.pending_identities[track_id]['detections'].append(detection)
        self.pending_identities[track_id]['features'].append(detection['reid_features'])
        self.pending_identities[track_id]['crops'].append(crop)
        
        # Store attributes if available
        if 'attributes' in detection:
            self.pending_identities[track_id]['attributes'].append(detection['attributes'])
        
        # Check if we have enough detections to create a new identity
        if len(self.pending_identities[track_id]['detections']) >= self.min_detections_for_new_id:
            # Average the features
            features = np.array(self.pending_identities[track_id]['features'])
            avg_feature = np.mean(features, axis=0)
            avg_feature = avg_feature / np.linalg.norm(avg_feature)
            
            # Get best crop (middle detection)
            best_crop_idx = len(self.pending_identities[track_id]['crops']) // 2
            best_crop = self.pending_identities[track_id]['crops'][best_crop_idx]
            
            # Prepare metadata with attributes if available
            metadata = {
                'first_seen_frame': self.pending_identities[track_id]['first_seen'],
                'first_seen_time': self.pending_identities[track_id]['timestamp'],
                'last_seen_frame': frame_idx,
                'last_seen_time': timestamp,
                'track_id': track_id,
                'appearances': len(self.pending_identities[track_id]['detections'])
            }
            
            # Add average attributes if available
            if self.pending_identities[track_id]['attributes']:
                # Compute average of each attribute
                all_attrs = self.pending_identities[track_id]['attributes']
                avg_attrs = {}
                
                # For each attribute type
                for attr_dict in all_attrs:
                    for attr_name, attr_val in attr_dict.items():
                        if attr_name not in avg_attrs:
                            avg_attrs[attr_name] = attr_val
                        else:
                            # Update running average
                            avg_attrs[attr_name] = (avg_attrs[attr_name] + attr_val) / 2
                
                metadata['attributes'] = avg_attrs
            
            # Create new identity
            person_id = self.database.add_person(
                features=avg_feature,
                image=best_crop,
                metadata=metadata
            )
            
            # Update all detections with this ID
            for det in self.pending_identities[track_id]['detections']:
                det['person_id'] = person_id
                det['reid_score'] = 1.0  # Perfect match for own detections
            
            # Remove from pending
            del self.pending_identities[track_id]
            
            self.stats['new_identities'] += 1
    
    def cleanup_pending_identities(self, max_age: int = 50) -> None:
        """
        Clean up old pending identities.
        
        Parameters:
        -----------
        max_age : int
            Maximum age (in frames) for pending identities
        """
        to_remove = []
        for track_id, data in self.pending_identities.items():
            age = data['detections'][-1].get('frame_index', 0) - data['first_seen']
            if age > max_age:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.pending_identities[track_id]
    
    def query_by_attributes(
        self,
        attribute_query: Dict[str, Union[int, float, List]],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Query database by attribute values.
        
        Parameters:
        -----------
        attribute_query : Dict
            Attributes to search for (e.g., {'gender': 1, 'upper_color': [2, 3]})
        top_k : int
            Number of top matches to return
            
        Returns:
        --------
        List[Dict]
            List of matched persons
        """
        if not self.use_attributes:
            logger.warning("Attribute-based search is not enabled")
            return []
        
        matches = []
        
        # Search through all persons in database
        for person_id, person_data in self.database.persons.items():
            # Skip if no attributes stored
            if 'metadata' not in person_data or 'attributes' not in person_data['metadata']:
                continue
            
            person_attrs = person_data['metadata']['attributes']
            match_score = 0
            max_score = 0
            
            # Calculate match score for each attribute
            for attr_name, query_val in attribute_query.items():
                if attr_name not in person_attrs:
                    continue
                    
                max_score += 1  # Count this attribute in normalization
                
                # Handle different query formats
                if isinstance(query_val, (int, float)):
                    # Exact match (for categorical attributes)
                    if isinstance(person_attrs[attr_name], (np.ndarray, list)):
                        # Softmax/sigmoid output: get highest class
                        if np.argmax(person_attrs[attr_name]) == query_val:
                            match_score += 1
                    else:
                        # Direct value
                        if person_attrs[attr_name] == query_val:
                            match_score += 1
                            
                elif isinstance(query_val, list):
                    # Multiple acceptable values
                    if isinstance(person_attrs[attr_name], (np.ndarray, list)):
                        # Check if highest probability class is in the list
                        if np.argmax(person_attrs[attr_name]) in query_val:
                            match_score += 1
                    else:
                        # Direct value
                        if person_attrs[attr_name] in query_val:
                            match_score += 1
            
            # Calculate normalized score
            if max_score > 0:
                normalized_score = match_score / max_score
                
                # Add to matches if score is above threshold
                if normalized_score > 0.5:  # Use 0.5 as minimum threshold for attributes
                    matches.append({
                        'id': person_id,
                        'similarity': normalized_score,
                        'metadata': person_data.get('metadata', {})
                    })
        
        # Sort by similarity
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Return top-k
        return matches[:top_k]
    
    def query_by_appearance(
        self,
        query_image: Union[np.ndarray, str],
        top_k: int = 5,
        use_attributes: bool = True
    ) -> List[Dict]:
        """
        Query database with an appearance image.
        
        Parameters:
        -----------
        query_image : np.ndarray or str
            Query image or path to image
        top_k : int
            Number of top matches to return
        use_attributes : bool
            Whether to use attributes in ranking
            
        Returns:
        --------
        List[Dict]
            List of matched persons
        """
        # Handle image path
        if isinstance(query_image, str):
            query_image = cv2.imread(query_image)
            
        if query_image is None or query_image.size == 0:
            return []
            
        # Extract features
        features = self.feature_extractor.extract_features(query_image)
        
        # Get initial matches based on features
        initial_matches = self.database.find_matches(features, top_k=top_k*2)  # Get more for reranking
        
        # If not using attributes or no attributes available, return direct matches
        if not use_attributes or not self.use_attributes or not initial_matches:
            return initial_matches[:top_k]
        
        # Extract attributes from query image
        query_attrs = self.feature_extractor.predict_attributes(query_image)
        
        # Re-rank matches using attribute similarity
        for match in initial_matches:
            person_id = match['id']
            person = self.database.get_person(person_id)
            
            if not person or 'metadata' not in person or 'attributes' not in person['metadata']:
                continue
                
            person_attrs = person['metadata']['attributes']
            
            # Calculate attribute similarity
            attr_sim = self._calculate_attribute_similarity(query_attrs, person_attrs)
            
            # Combine with feature similarity (0.7 feature + 0.3 attributes)
            combined_sim = 0.7 * match['similarity'] + 0.3 * attr_sim
            
            # Update similarity score
            match['similarity'] = combined_sim
            match['feature_similarity'] = match['similarity']  # Store original for reference
            match['attribute_similarity'] = attr_sim
        
        # Re-sort based on combined similarity
        initial_matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        return initial_matches[:top_k]
    
    def _calculate_attribute_similarity(self, query_attrs, person_attrs):
        """Calculate similarity between two sets of attributes."""
        if not query_attrs or not person_attrs:
            return 0.0
            
        similarity = 0.0
        count = 0
        
        for attr_name, query_val in query_attrs.items():
            if attr_name not in person_attrs:
                continue
                
            person_val = person_attrs[attr_name]
            
            # Handle different types of attributes
            if isinstance(query_val, (np.ndarray, list)) and isinstance(person_val, (np.ndarray, list)):
                # Distribution similarity (softmax/sigmoid outputs)
                if len(query_val) == len(person_val):
                    # Calculate cosine similarity
                    dot_product = np.sum(query_val * person_val)
                    norm_product = np.linalg.norm(query_val) * np.linalg.norm(person_val)
                    if norm_product > 0:
                        attr_sim = dot_product / norm_product
                        similarity += attr_sim
                        count += 1
            elif isinstance(query_val, (int, float)) and isinstance(person_val, (int, float)):
                # Direct value comparison
                if query_val == person_val:
                    similarity += 1.0
                    count += 1
        
        # Normalize
        if count > 0:
            return similarity / count
        else:
            return 0.0
    
    def save(self, base_path: str) -> None:
        """
        Save the reidentifier state.
        
        Parameters:
        -----------
        base_path : str
            Base path for saving files
        """
        # Create directory if not exists
        base_dir = Path(base_path)
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # Save feature extractor
        if hasattr(self.feature_extractor, 'model'):
            model_path = base_dir / "feature_extractor.pth"
            torch.save(self.feature_extractor.model.state_dict(), str(model_path))
        
        # Save database
        db_path = base_dir / "person_database.json"
        self.database.save_database(str(db_path))
        
        # Save configuration
        config = {
            'model_type': self.model_type,
            'similarity_threshold': self.similarity_threshold,
            'use_attributes': self.use_attributes,
            'min_detections_for_new_id': self.min_detections_for_new_id,
            'stats': self.stats,
            'timestamp': datetime.now().isoformat()
        }
        
        config_path = base_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        logger.info(f"Enhanced ReID system saved to {base_path}")
    
    def load(self, base_path: str) -> None:
        """
        Load the reidentifier state.
        
        Parameters:
        -----------
        base_path : str
            Base path for loading files
        """
        base_dir = Path(base_path)
        
        # Load feature extractor
        model_path = base_dir / "feature_extractor.pth"
        if model_path.exists() and hasattr(self.feature_extractor, 'model'):
            self.feature_extractor.model.load_state_dict(
                torch.load(str(model_path), map_location=self.feature_extractor.device)
            )
            self.feature_extractor.model.eval()
        
        # Load database
        db_path = base_dir / "person_database.json"
        if db_path.exists():
            self.database.load_database(str(db_path))
        
        # Load configuration
        config_path = base_dir / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            self.similarity_threshold = config.get('similarity_threshold', self.similarity_threshold)
            self.use_attributes = config.get('use_attributes', self.use_attributes)
            self.min_detections_for_new_id = config.get('min_detections_for_new_id', self.min_detections_for_new_id)
            self.stats = config.get('stats', self.stats)
            self.model_type = config.get('model_type', self.model_type)
            
        logger.info(f"Enhanced ReID system loaded from {base_path}")