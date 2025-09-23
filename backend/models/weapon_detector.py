import cv2
import numpy as np
import torch
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

class WeaponDetector:
    """Specialized detector for weapons and suspicious items."""
    
    def __init__(
        self, 
        model_path="./backend/models/weapon_detect.pt",  # Updated absolute path
        confidence_threshold=0.45,  # Higher threshold to reduce false positives
        device=None
    ):
        """
        Initialize the weapon detector.
        
        Parameters:
        -----------
        model_path : str
            Path to the weapon detection model weights
        confidence_threshold : float
            Confidence threshold for detection (higher for weapons to reduce false positives)
        device : str, optional
            Device to run the model on ('cpu', 'cuda', 'mps')
        """
        self.confidence_threshold = confidence_threshold
        self.device = device or ('cuda' if torch.cuda.is_available() else 
                                'mps' if hasattr(torch.backends, 'mps') and 
                                torch.backends.mps.is_available() else 'cpu')
        
       
        import os
        os.environ['TORCH_HOME'] = '/tmp/torch_home_no_download'
        os.environ['ULTRALYTICS_CONFIG_DIR'] = '/tmp/torch_no_download_config'
       
        # Load weapon detection model
        # Check if file exists
        import os
        if not os.path.exists(model_path):
            print(f"ERROR: Weapon model file not found at {model_path}")
            self.model = None
            return
        else:
            file_size_mb = os.path.getsize(model_path)/1024/1024
            print(f"Found weapon model at {model_path} - size: {file_size_mb:.1f} MB")
            # Very small files are likely corrupted
            if file_size_mb < 1.0:
                print(f"WARNING: Model file is suspiciously small ({file_size_mb:.1f} MB)")
        
        # Load weapon detection model
        try:
            print(f"Attempting to load weapon model using YOLOv5 from {model_path}")
            
            # Set cache directory to prevent downloads
            torch.hub.set_dir('/tmp/torch_no_download_hub')
            
            # Try direct loading first (safer)
            try:
                import sys, subprocess
                
                # Install dependencies only if needed
                try:
                    import yolov5
                except ImportError:
                    print("YOLOv5 package not found. Will use torch.hub...")
                    
                # Try loading with torch.hub with safety measures
                print("Loading with torch.hub...")
                self.model = torch.hub.load(
                    'ultralytics/yolov5', 
                    'custom', 
                    path=model_path, 
                    trust_repo=True,
                    force_reload=True,
                    skip_validation=True,
                    verbose=True  # Show detailed output
                )
                
                # Configure model
                self.model.conf = self.confidence_threshold  # Set confidence threshold
                self.model.to(self.device)  # Move to correct device
                
                # Verify model loaded correctly
                if hasattr(self.model, 'names'):
                    print(f"Model loaded successfully with classes: {self.model.names}")
                else:
                    print("WARNING: Model loaded but no class names found")
                    
            except Exception as e1:
                print(f"First loading method failed: {e1}")
                
                # Try alternative loading method
                print("Trying alternative loading method...")
                try:
                    # Load as a local PyTorch model
                    self.model = torch.load(model_path, map_location=self.device)
                    print("Model loaded directly with torch.load()")
                except Exception as e2:
                    print(f"Second loading method failed: {e2}")
                    raise  # Re-raise to outer exception handler
                
            # Set class names if the model doesn't define them
            # Modified code that handles dictionary models
            if isinstance(self.model, dict):
                print("Model loaded as dictionary, creating wrapper")
                # Store the dict model
                self.model_dict = self.model
                # Create a simple wrapper class to hold the model dict and names
                class ModelWrapper:
                    def __init__(self, model_dict):
                        self.model_dict = model_dict
                        self.names = {
                            0: "pistol", 1: "rifle", 2: "shotgun", 3: "knife", 
                            4: "machete", 5: "bat", 6: "suspicious_package"
                        }
                
                # Replace dict with wrapper object
                self.model = ModelWrapper(self.model_dict)
                print(f"Created model wrapper with classes: {self.model.names}")
            elif not hasattr(self.model, 'names') or not self.model.names:
                print("Setting default class names for model")
                self.model.names = {
                    0: "pistol", 1: "rifle", 2: "shotgun", 3: "knife", 
                    4: "machete", 5: "bat", 6: "suspicious_package"
                }
                
            logger.info(f"Weapon detector initialized on {self.device}")
            
        except Exception as e:
            print(f"ERROR: All weapon model loading methods failed: {e}")
            print("Detailed error information:")
            import traceback
            traceback.print_exc()
            self.model = None
            logger.error(f"Failed to load weapon model: {e}")
    
    # Update the detect method in weapon_detector.py
    def detect(self, frame):
        """Detect weapons in a frame."""
        if self.model is None:
            print("WARNING: Weapon detection model not loaded")
            return []
        
        # Print a debug message
        print(f"Running weapon detection on frame with shape {frame.shape}")
        
        # Run detection
        try:
            results = self.model(frame)
            
            # Process results
            detections = []
            
            # Get detections
            result_data = results.pandas().xyxy[0]
            
            # Check if we found anything
            if len(result_data) > 0:
                print(f"Weapon detector found {len(result_data)} potential objects")
            
            # Process each detection
            for idx, row in result_data.iterrows():
                # Get class name and ensure lowercase for consistency
                class_name = str(row["name"]).lower() if "name" in row else f"class_{int(row['class'])}"
                confidence = float(row["confidence"])
                
                # Only include weapon-like objects
                is_weapon = any(weapon in class_name for weapon in 
                            ["gun", "pistol", "rifle", "weapon", "knife", "shotgun", "machete"])
                
                # Skip low-confidence or non-weapon detections
                if confidence < self.confidence_threshold or not is_weapon:
                    continue
                    
                print(f"  ✅ Detected weapon: {class_name} with confidence {confidence:.2f}")
                
                # Create detection object
                detection = {
                    "class_id": int(row["class"]),
                    "class_name": class_name,
                    "confidence": confidence,
                    "box": [
                        float(row["xmin"]), 
                        float(row["ymin"]), 
                        float(row["xmax"]), 
                        float(row["ymax"])
                    ],
                    "box_center": [
                        (float(row["xmin"]) + float(row["xmax"])) / 2,
                        (float(row["ymin"]) + float(row["ymax"])) / 2
                    ],
                    "type": "weapon",
                    "threat_level": self._determine_threat_level(class_name, confidence)
                }
                
                detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"ERROR during weapon detection: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _determine_threat_level(self, class_name, confidence):
        """Determine threat level based on weapon type and confidence."""
        # Higher threat level for firearms
        if class_name in ["pistol", "rifle", "shotgun"]:
            base_level = 3  # High
        elif class_name in ["knife", "machete"]:
            base_level = 2  # Medium
        else:
            base_level = 1  # Low
            
        # Adjust by confidence
        if confidence > 0.85:
            return base_level
        elif confidence > 0.75:
            return max(1, base_level - 1)  # Reduce by 1 but minimum 1
        else:
            return max(1, base_level - 2)  # Reduce by 2 but minimum 1
            
    def is_weapon(self, class_name):
        """Check if a class is a weapon."""
        return class_name in self.weapon_classes.values()