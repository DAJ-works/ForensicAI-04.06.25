import os
import numpy as np
import cv2
import torch
import json
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersonReidentifier:
    """
    Person re-identification model using ResNet50 features.
    """
    
    def __init__(self, match_threshold=0.75, max_features_per_person=10, device=None):
        """
        Initialize the person re-identification model.
        
        Parameters:
        -----------
        match_threshold : float
            Similarity threshold for matching (increased from 0.5 to 0.75)
        max_features_per_person : int
            Maximum number of feature vectors to store per person
        device : str, optional
            Device to run the model on ('cpu', 'cuda', 'mps')
        """
        self.match_threshold = match_threshold  # Increased from 0.5 to 0.75
        self.max_features_per_person = max_features_per_person  # New parameter
        self.model = None
        self.device = device
        self.person_db = {}
        self.next_person_id = 1
        self.person_images = {}
        
        # Set device automatically if not specified
        if self.device is None:
            self.device = 'cpu'  # Default to CPU for compatibility
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'  # M1/M2 Macs
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load ResNet50 model for feature extraction."""
        try:
            from torchvision import models
            
            print(f"Initialized resnet50 feature extractor on {self.device}")
            self.model = models.resnet50(pretrained=True)
            
            # Remove the final classification layer
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
            self.model.to(self.device)
            self.model.eval()
            
            # Initialize transformations
            from torchvision import transforms
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
        except Exception as e:
            logger.error(f"Error loading ReID model: {e}")
            self.model = None
    
    def extract_features(self, image):
        """
        Extract features from an image.
        
        Parameters:
        -----------
        image : np.ndarray
            Input image (BGR format)
            
        Returns:
        --------
        np.ndarray
            Feature vector
        """
        if self.model is None:
            return None
        
        # Convert to RGB and preprocess
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        tensor = self.transform(image_rgb)
        tensor = tensor.unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(tensor)
            features = features.view(features.size(0), -1)
            
        # Convert to numpy and normalize
        features = features.cpu().numpy()[0]
        features = features / np.linalg.norm(features)
        
        return features
    
    def update(self, frame, person_detections):
        """
        Update the person database with new detections.
        
        Parameters:
        -----------
        frame : np.ndarray
            Current frame
        person_detections : List[Dict]
            List of person detections
            
        Returns:
        --------
        List[int]
            List of person IDs corresponding to the detections
        """
        if self.model is None:
            logger.warning("ReID model not loaded")
            return [-1] * len(person_detections)
        
        person_ids = []
        
        # New: Process each detection once and store their features
        detection_features = []
        valid_detections = []
        
        for detection in person_detections:
            # Extract person from frame using bounding box
            box = detection.get('box', [0, 0, 0, 0])
            x1, y1, x2, y2 = map(int, box)
            
            # Ensure valid box dimensions
            if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                detection_features.append(None)
                valid_detections.append(False)
                continue
            
            # Extract person image
            person_img = frame[y1:y2, x1:x2]
            
            # Skip if image is too small
            if person_img.size == 0 or person_img.shape[0] < 20 or person_img.shape[1] < 20:
                detection_features.append(None)
                valid_detections.append(False)
                continue
            
            # Extract features
            features = self.extract_features(person_img)
            detection_features.append(features)
            valid_detections.append(features is not None)
        
        # Process each detection in a separate pass to avoid ID conflicts
        # This helps ensure we don't match multiple detections to the same ID
        matched_ids = set()
        
        for idx, (detection, features, is_valid) in enumerate(zip(person_detections, detection_features, valid_detections)):
            if not is_valid:
                person_ids.append(-1)
                continue
                
            # Match with existing persons
            best_id, best_similarity = self._match_person(features)
            
            # Check if this ID is already matched in this frame and if the match meets threshold
            if best_id != -1 and best_similarity >= self.match_threshold and best_id not in matched_ids:
                # Update existing person with new features, keeping only the most recent ones
                self.person_db[best_id]['features'].append(features)
                if len(self.person_db[best_id]['features']) > self.max_features_per_person:
                    # Remove the oldest feature
                    self.person_db[best_id]['features'] = self.person_db[best_id]['features'][-self.max_features_per_person:]
                
                self.person_db[best_id]['detections'] += 1
                
                # Store latest image if higher quality (larger)
                box = detection.get('box', [0, 0, 0, 0])
                x1, y1, x2, y2 = map(int, box)
                person_img = frame[y1:y2, x1:x2]
                
                current_img = self.person_images.get(best_id)
                if current_img is None or person_img.size > current_img.size:
                    self.person_images[best_id] = person_img
                
                matched_ids.add(best_id)
                person_ids.append(best_id)
            else:
                # Create new person
                new_id = self.next_person_id
                self.next_person_id += 1
                
                self.person_db[new_id] = {
                    'id': new_id,
                    'features': [features],
                    'detections': 1
                }
                
                # Store image
                box = detection.get('box', [0, 0, 0, 0])
                x1, y1, x2, y2 = map(int, box)
                self.person_images[new_id] = frame[y1:y2, x1:x2]
                
                person_ids.append(new_id)
        
        # Debug info
        logger.info(f"Frame processed: {len(person_ids)} persons identified, {len(set(person_ids))} unique")
        
        return person_ids
    
    def _match_person(self, features):
        """
        Match features with existing persons.
        
        Parameters:
        -----------
        features : np.ndarray
            Feature vector
            
        Returns:
        --------
        Tuple[int, float]
            Matched person ID and similarity (-1 if no match)
        """
        best_match = -1
        best_similarity = 0.0
        
        for person_id, person_data in self.person_db.items():
            # Calculate average similarity across all features for this person
            similarities = []
            for person_features in person_data['features']:
                similarity = np.dot(features, person_features)
                similarities.append(similarity)
            
            # Use the average similarity for more stable matching
            if similarities:
                avg_similarity = np.mean(similarities)
                
                # Also check max similarity to ensure at least one strong match
                max_similarity = np.max(similarities)
                
                # Consider both average and max in deciding the match
                effective_similarity = 0.7 * max_similarity + 0.3 * avg_similarity
                
                if effective_similarity > best_similarity:
                    best_similarity = effective_similarity
                    best_match = person_id
        
        return best_match, best_similarity
    
    def get_person_count(self):
        """
        Get the number of identified persons.
        
        Returns:
        --------
        int
            Number of persons
        """
        return len(self.person_db)
    
    def save_database(self, output_path):
        """
        Save the person database to a file.
        
        Parameters:
        -----------
        output_path : str
            Output path for the database file
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Prepare serializable database
        serializable_db = {}
        for person_id, person_data in self.person_db.items():
            serializable_db[str(person_id)] = {
                'id': person_data['id'],
                'detections': person_data['detections'],
                'features_shape': person_data['features'][0].shape
            }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(serializable_db, f, indent=2)
    
    def save_person_image(self, person_id, image, output_dir):
        """
        Save a person image to a file.
        
        Parameters:
        -----------
        person_id : int
            Person ID
        image : np.ndarray, optional
            Person image (uses stored image if None)
        output_dir : str
            Output directory for the image
        """
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Use provided image or stored image
        if image is None:
            image = self.person_images.get(person_id)
        
        # Skip if no image available
        if image is None:
            logger.warning(f"No image available for person {person_id}")
            return
        
        # Save image
        output_path = os.path.join(output_dir, f"person_{person_id}.jpg")
        try:
            cv2.imwrite(output_path, image)
        except Exception as e:
            logger.error(f"Error saving person image: {e}")
    
    def visualize_database(self, output_path, max_persons=20, grid_size=None):
        """
        Create a visualization of all identified persons.
        
        Parameters:
        -----------
        output_path : str
            Output path for the visualization
        max_persons : int
            Maximum number of persons to include
        grid_size : tuple, optional
            Grid size (rows, cols)
        """
        try:
            if not self.person_images:
                logger.warning("No person images to visualize")
                return
            
            # Limit the number of persons
            person_ids = list(self.person_images.keys())[:max_persons]
            num_persons = len(person_ids)
            
            if num_persons == 0:
                return
            
            # Determine grid size if not provided
            if grid_size is None:
                grid_cols = min(5, num_persons)
                grid_rows = (num_persons + grid_cols - 1) // grid_cols
            else:
                grid_rows, grid_cols = grid_size
            
            # Resize images to a common size
            target_height = 200
            images = []
            for person_id in person_ids:
                image = self.person_images.get(person_id)
                if image is not None:
                    # Resize maintaining aspect ratio
                    aspect_ratio = image.shape[1] / image.shape[0]
                    target_width = int(target_height * aspect_ratio)
                    resized = cv2.resize(image, (target_width, target_height))
                    
                    # Add person ID
                    cv2.putText(resized, f"ID: {person_id}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    
                    images.append(resized)
            
            # Create a blank canvas
            max_width = max(img.shape[1] for img in images)
            canvas = np.zeros((grid_rows * target_height, grid_cols * max_width, 3), dtype=np.uint8)
            
            # Place images on the canvas
            for i, image in enumerate(images):
                row = i // grid_cols
                col = i % grid_cols
                y1 = row * target_height
                y2 = y1 + image.shape[0]
                x1 = col * max_width
                x2 = x1 + image.shape[1]
                canvas[y1:y2, x1:x2] = image
            
            # Save the visualization
            cv2.imwrite(output_path, canvas)
            logger.info(f"Person gallery saved to {output_path}")
                
        except Exception as e:
            logger.error(f"Error creating person gallery: {e}")