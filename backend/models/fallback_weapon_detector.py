import os
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class BasicWeaponDetector:
    """A simplified weapon detector for fallback use"""
    
    def __init__(self, model_path=None, confidence_threshold=0.45):
        """Initialize a basic weapon detector"""
        self.confidence_threshold = confidence_threshold
        self.model = True  # Dummy model but this ensures the attribute exists
        self.model_path = model_path
        
        # Set class names
        self.model_names = {
            0: "pistol", 1: "rifle", 2: "shotgun", 3: "knife", 
            4: "machete", 5: "bat", 6: "suspicious_package"
        }
        
        logger.info(f"BasicWeaponDetector initialized (fallback mode)")
        print("BasicWeaponDetector initialized in fallback mode")
        
    def detect(self, frame):
        """Placeholder detection function"""
        print("Running BasicWeaponDetector.detect (fallback mode)")
        # This detector doesn't do actual detection, just serves as a placeholder
        # that doesn't crash when called
        return []