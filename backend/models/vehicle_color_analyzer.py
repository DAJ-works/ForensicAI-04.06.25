import cv2
import numpy as np
from sklearn.cluster import KMeans

class VehicleColorAnalyzer:
    """Analyze and determine the dominant color of vehicles."""
    
    def __init__(self):
        # Define color ranges and names
        self.color_ranges = {
            'black': ([0, 0, 0], [180, 255, 50]),
            'white': ([0, 0, 200], [180, 30, 255]),
            'red': ([0, 50, 50], [10, 255, 255]),
            'blue': ([100, 50, 50], [130, 255, 255]),
            'green': ([40, 50, 50], [80, 255, 255]),
            'yellow': ([20, 100, 100], [35, 255, 255]),
            'orange': ([10, 100, 100], [20, 255, 255]),
            'silver': ([0, 0, 100], [180, 30, 200]),
        }
    
    def analyze_vehicle_color(self, vehicle_img):
        """
        Determine the dominant color of a vehicle.
        
        Parameters:
        -----------
        vehicle_img : np.ndarray
            Vehicle image in BGR format
            
        Returns:
        --------
        str
            Dominant color name
        """
        # Convert to HSV for better color analysis
        hsv_img = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2HSV)
        
        # Mask out shadows and reflections
        mask = cv2.inRange(hsv_img, (0, 30, 30), (180, 255, 255))
        masked_img = cv2.bitwise_and(hsv_img, hsv_img, mask=mask)
        
        # Reshape for KMeans
        pixels = masked_img.reshape(-1, 3)
        pixels = pixels[~np.all(pixels == 0, axis=1)]  # Remove black pixels from mask
        
        if len(pixels) == 0:
            return "unknown"
            
        # Use KMeans to find dominant colors
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(pixels)
        
        # Get the dominant color cluster
        dominant_cluster = np.argmax(np.bincount(kmeans.labels_))
        dominant_color = kmeans.cluster_centers_[dominant_cluster]
        
        # Match to known colors
        return self._match_color(dominant_color)
    
    def _match_color(self, hsv_color):
        """Match HSV color to predefined color names."""
        best_match = "unknown"
        max_match_count = 0
        
        # Check if the color falls within any of our defined ranges
        for color_name, (lower, upper) in self.color_ranges.items():
            lower = np.array(lower)
            upper = np.array(upper)
            
            if np.all(hsv_color >= lower) and np.all(hsv_color <= upper):
                # Count number of parameters that match well
                match_count = np.sum((hsv_color >= lower) & (hsv_color <= upper))
                if match_count > max_match_count:
                    max_match_count = match_count
                    best_match = color_name
        
        return best_match