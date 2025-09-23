import numpy as np
from typing import List, Dict, Tuple
import logging
import time

logger = logging.getLogger(__name__)

class EnhancedInteractionDetector:
    """Detect interactions with enhanced weapon and suspicious behavior detection."""
    
    def __init__(
        self, 
        proximity_threshold=0.3, 
        temporal_window=30,
        weapon_proximity_threshold=0.5,  # Threshold for person-weapon proximity
        weapon_alert_confidence=0.7      # Min confidence for weapon alerts
    ):
        """
        Initialize the enhanced interaction detector.
        
        Parameters:
        -----------
        proximity_threshold : float
            Spatial threshold for interaction (as a fraction of frame size)
        temporal_window : int
            Number of frames to analyze for temporal interactions
        weapon_proximity_threshold : float
            Threshold for weapon-person proximity (larger than regular proximity)
        weapon_alert_confidence : float
            Minimum confidence for generating weapon alerts
        """
        self.proximity_threshold = proximity_threshold
        self.temporal_window = temporal_window
        self.weapon_proximity_threshold = weapon_proximity_threshold
        self.weapon_alert_confidence = weapon_alert_confidence
        
        self.object_history = {}  # Track objects across frames
        self.interaction_events = []
        self.frame_history = []  # Store recent frames for analysis
        self.weapon_alerts = []  # Track weapon-related alerts
        
        # Suspicious behavior patterns
        self.suspicious_patterns = {
            "person_weapon_proximity": {
                "description": "Person close to weapon",
                "severity": 3
            },
            "concealed_weapon": {
                "description": "Possible concealed weapon",
                "severity": 4
            },
            "weapon_handoff": {
                "description": "Possible weapon handoff between individuals",
                "severity": 5
            },
            "weapon_brandishing": {
                "description": "Person potentially brandishing weapon",
                "severity": 5
            }
        }
    
    def update(self, frame_idx, timestamp, tracked_objects, frame=None):
        """
        Update the interaction detector with new frame data.
        
        Parameters:
        -----------
        frame_idx : int
            Current frame index
        timestamp : float
            Current frame timestamp
        tracked_objects : List[Dict]
            List of tracked objects with their properties
        frame : np.ndarray, optional
            Current frame image for visual analysis
            
        Returns:
        --------
        List[Dict]
            Newly detected interaction events
        """
        current_frame_data = {
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "objects": tracked_objects
        }
        
        # Add frame to history
        self.frame_history.append(current_frame_data)
        if len(self.frame_history) > self.temporal_window:
            self.frame_history.pop(0)
        
        # Update object history
        self._update_object_history(tracked_objects, frame_idx, timestamp)
        
        # Detect spatial interactions (objects close to each other)
        spatial_interactions = self._detect_spatial_interactions(tracked_objects)
        
        # Detect temporal interactions (changes over time)
        temporal_interactions = self._detect_temporal_interactions()
        
        # Detect weapon-specific interactions
        weapon_interactions = self._detect_weapon_interactions(tracked_objects)
        
        # Generate events from interactions
        new_events = self._generate_events(
            spatial_interactions, 
            temporal_interactions,
            weapon_interactions,
            frame_idx,
            timestamp
        )
        
        # Add new events to history
        self.interaction_events.extend(new_events)
        
        return new_events
    
    def _update_object_history(self, objects, frame_idx, timestamp):
        """Update the history of tracked objects."""
        for obj in objects:
            obj_id = obj.get("object_id", -1)
            if obj_id == -1:
                continue
                
            if obj_id not in self.object_history:
                self.object_history[obj_id] = {
                    "class_name": obj["class_name"],
                    "class_id": obj["class_id"],
                    "type": obj.get("type", "normal"),  # Track if this is a weapon
                    "first_seen": frame_idx,
                    "last_seen": frame_idx,
                    "positions": [],
                    "properties": {},
                    "interactions": [],
                    "threat_level": obj.get("threat_level", 0)
                }
                
                # Store color info for vehicles
                if obj["class_name"] in ["car", "truck", "bus"] and "color" in obj:
                    self.object_history[obj_id]["properties"]["color"] = obj["color"]
                
                # Store confidence for weapons
                if obj.get("type") == "weapon":
                    self.object_history[obj_id]["properties"]["initial_confidence"] = obj["confidence"]
            
            # Update last seen
            history = self.object_history[obj_id]
            history["last_seen"] = frame_idx
            
            # Update threat level if it increased
            if "threat_level" in obj and obj["threat_level"] > history.get("threat_level", 0):
                history["threat_level"] = obj["threat_level"]
            
            # Add position
            center = obj.get("box_center", [(obj["box"][0] + obj["box"][2])/2, 
                                           (obj["box"][1] + obj["box"][3])/2])
            history["positions"].append({
                "frame": frame_idx,
                "timestamp": timestamp,
                "center": center,
                "box": obj["box"],
                "confidence": obj.get("confidence", 1.0)
            })
            
            # Limit history length
            if len(history["positions"]) > self.temporal_window:
                history["positions"].pop(0)
    
    def _detect_spatial_interactions(self, objects):
        """Detect objects that are close to each other in the current frame."""
        interactions = []
        
        # Check each pair of objects
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects[i+1:], i+1):
                # Skip if invalid IDs
                if (obj1.get("object_id", -1) == -1 or obj2.get("object_id", -1) == -1):
                    continue
                
                # Calculate centers
                center1 = obj1.get("box_center", [(obj1["box"][0] + obj1["box"][2])/2, 
                                                (obj1["box"][1] + obj1["box"][3])/2])
                center2 = obj2.get("box_center", [(obj2["box"][0] + obj2["box"][2])/2, 
                                                (obj2["box"][1] + obj2["box"][3])/2])
                
                # Calculate distance
                distance = np.linalg.norm(np.array(center1) - np.array(center2))
                
                # Normalize by object sizes
                obj1_size = max(obj1["box"][2] - obj1["box"][0], obj1["box"][3] - obj1["box"][1])
                obj2_size = max(obj2["box"][2] - obj2["box"][0], obj2["box"][3] - obj2["box"][1])
                avg_size = (obj1_size + obj2_size) / 2
                
                normalized_distance = distance / avg_size
                
                # Set appropriate threshold based on object types
                threshold = self.proximity_threshold
                
                # Use weapon proximity threshold if one object is a weapon
                if (obj1.get("type") == "weapon" or obj2.get("type") == "weapon"):
                    threshold = self.weapon_proximity_threshold
                
                # Check if objects are close
                if normalized_distance < threshold:
                    interactions.append({
                        "type": "proximity",
                        "objects": [obj1["object_id"], obj2["object_id"]],
                        "classes": [obj1["class_name"], obj2["class_name"]],
                        "object_types": [obj1.get("type", "normal"), obj2.get("type", "normal")],
                        "distance": distance,
                        "normalized_distance": normalized_distance
                    })
        
        return interactions

    def _detect_weapon_interactions(self, objects):
        """Detect weapon-specific interactions and suspicious behaviors."""
        interactions = []
        
        # Get all weapons in the current frame
        weapons = [obj for obj in objects if obj.get("type") == "weapon"]
        
        # Get all people in the current frame
        people = [obj for obj in objects if obj["class_name"] == "person"]
        
        # Check interactions between people and weapons
        for weapon in weapons:
            weapon_id = weapon.get("object_id", -1)
            if weapon_id == -1:
                continue
                
            weapon_center = np.array(weapon.get("box_center", 
                                    [(weapon["box"][0] + weapon["box"][2])/2, 
                                     (weapon["box"][1] + weapon["box"][3])/2]))
            
            # Check weapon confidence
            weapon_confidence = weapon.get("confidence", 0)
            
            # Only create alerts for weapons with sufficient confidence
            if weapon_confidence < self.weapon_alert_confidence:
                continue
            
            # Check for nearby people
            for person in people:
                person_id = person.get("object_id", -1)
                if person_id == -1:
                    continue
                
                person_center = np.array(person.get("box_center", 
                                       [(person["box"][0] + person["box"][2])/2, 
                                        (person["box"][1] + person["box"][3])/2]))
                
                # Calculate distance
                distance = np.linalg.norm(weapon_center - person_center)
                
                # Normalize by object sizes
                weapon_size = max(weapon["box"][2] - weapon["box"][0], 
                                 weapon["box"][3] - weapon["box"][1])
                person_size = max(person["box"][2] - person["box"][0], 
                                 person["box"][3] - person["box"][1])
                avg_size = (weapon_size + person_size) / 2
                
                normalized_distance = distance / avg_size
                
                # Check for weapon-person proximity
                if normalized_distance < self.weapon_proximity_threshold:
                    interactions.append({
                        "type": "person_weapon_proximity",
                        "weapon_id": weapon_id,
                        "person_id": person_id,
                        "weapon_class": weapon["class_name"],
                        "distance": distance,
                        "normalized_distance": normalized_distance,
                        "weapon_confidence": weapon_confidence,
                        "severity": self.suspicious_patterns["person_weapon_proximity"]["severity"]
                    })
                
                # Check for weapon inside person bounding box (potential concealment)
                person_box = person["box"]
                weapon_box = weapon["box"]
                
                # Check if weapon is mostly inside person bounding box
                overlap_area = self._calculate_box_overlap(person_box, weapon_box)
                weapon_area = (weapon_box[2] - weapon_box[0]) * (weapon_box[3] - weapon_box[1])
                
                if overlap_area > 0.7 * weapon_area:  # 70% of weapon inside person box
                    interactions.append({
                        "type": "concealed_weapon",
                        "weapon_id": weapon_id,
                        "person_id": person_id,
                        "weapon_class": weapon["class_name"],
                        "overlap_ratio": overlap_area / weapon_area,
                        "weapon_confidence": weapon_confidence,
                        "severity": self.suspicious_patterns["concealed_weapon"]["severity"]
                    })
        
        # Look for potential weapon handoffs (weapons moving between people)
        weapon_handoffs = self._detect_weapon_handoffs()
        if weapon_handoffs:
            interactions.extend(weapon_handoffs)
        
        # Look for sudden weapon appearances (potential brandishing)
        brandishing_events = self._detect_weapon_brandishing(objects)
        if brandishing_events:
            interactions.extend(brandishing_events)
        
        return interactions
    
    def _calculate_box_overlap(self, box1, box2):
        """Calculate the overlap area between two bounding boxes."""
        # Extract coordinates
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection
        x_min = max(x1_min, x2_min)
        y_min = max(y1_min, y2_min)
        x_max = min(x1_max, x2_max)
        y_max = min(y1_max, y2_max)
        
        # Check if there is an intersection
        if x_max < x_min or y_max < y_min:
            return 0
            
        return (x_max - x_min) * (y_max - y_min)
    
    def _detect_weapon_handoffs(self):
        """Detect potential weapon handoffs between people."""
        handoffs = []
        
        # This requires tracking a weapon that was close to one person
        # and then becomes close to another person
        
        # Get all weapons that have sufficient history
        for obj_id, history in self.object_history.items():
            if history.get("type") != "weapon" or len(history["positions"]) < 3:
                continue
                
            weapon_class = history["class_name"]
            
            # Look for nearby people at each position
            prev_person = None
            
            for pos_idx, position in enumerate(history["positions"]):
                # Skip if final position (need next position to compare)
                if pos_idx == len(history["positions"]) - 1:
                    continue
                    
                frame_idx = position["frame"]
                
                # Find people close to this weapon in this frame
                close_people = []
                
                for other_id, other_history in self.object_history.items():
                    if other_history["class_name"] != "person":
                        continue
                        
                    # Find person's position at this frame
                    person_positions = [p for p in other_history["positions"] if p["frame"] == frame_idx]
                    if not person_positions:
                        continue
                        
                    person_position = person_positions[0]
                    
                    # Calculate distance
                    weapon_center = np.array(position["center"])
                    person_center = np.array(person_position["center"])
                    distance = np.linalg.norm(weapon_center - person_center)
                    
                    # Check proximity
                    if distance < 150:  # Arbitrary pixel distance
                        close_people.append({
                            "person_id": other_id,
                            "distance": distance
                        })
                
                # Sort by distance
                close_people.sort(key=lambda x: x["distance"])
                
                # Get closest person
                current_person = close_people[0]["person_id"] if close_people else None
                
                # Check for handoff (different person than before)
                if prev_person is not None and current_person is not None and current_person != prev_person:
                    # This could be a handoff
                    handoffs.append({
                        "type": "weapon_handoff",
                        "weapon_id": obj_id,
                        "weapon_class": weapon_class,
                        "from_person": prev_person,
                        "to_person": current_person,
                        "frame": frame_idx,
                        "timestamp": position["timestamp"],
                        "severity": self.suspicious_patterns["weapon_handoff"]["severity"]
                    })
                
                # Update previous person
                if current_person is not None:
                    prev_person = current_person
        
        return handoffs
    
    def _detect_weapon_brandishing(self, objects):
        """Detect potential weapon brandishing (sudden appearance, rapid movements)."""
        brandishing_events = []
        
        # Get all weapons in current frame
        current_weapons = [obj for obj in objects if obj.get("type") == "weapon"]
        
        for weapon in current_weapons:
            weapon_id = weapon.get("object_id", -1)
            if weapon_id == -1:
                continue
                
            # Get weapon history
            history = self.object_history.get(weapon_id)
            if not history or len(history["positions"]) < 2:
                continue
                
            # Check if this is a recently appeared weapon
            if len(history["positions"]) < 5 and len(self.frame_history) > 10:
                # Weapon appeared recently in a video that's been running
                
                # Check if it's near a person
                for obj in objects:
                    if obj["class_name"] == "person":
                        person_id = obj.get("object_id", -1)
                        if person_id == -1:
                            continue
                            
                        person_center = np.array(obj.get("box_center",
                                               [(obj["box"][0] + obj["box"][2])/2,
                                                (obj["box"][1] + obj["box"][3])/2]))
                        weapon_center = np.array(weapon.get("box_center",
                                               [(weapon["box"][0] + weapon["box"][2])/2,
                                                (weapon["box"][1] + weapon["box"][3])/2]))
                        
                        distance = np.linalg.norm(person_center - weapon_center)
                        
                        if distance < 150:  # Arbitrary pixel distance
                            brandishing_events.append({
                                "type": "weapon_brandishing",
                                "weapon_id": weapon_id,
                                "weapon_class": weapon["class_name"],
                                "person_id": person_id,
                                "confidence": weapon.get("confidence", 0),
                                "frame": self.frame_history[-1]["frame_idx"],
                                "timestamp": self.frame_history[-1]["timestamp"],
                                "severity": self.suspicious_patterns["weapon_brandishing"]["severity"]
                            })
                            break  # Only report one brandishing event per weapon
            
            # Check for rapid movement of weapon (another form of brandishing)
            if len(history["positions"]) >= 3:
                recent_positions = history["positions"][-3:]
                
                # Calculate movement speed
                total_distance = 0
                for i in range(1, len(recent_positions)):
                    pos1 = np.array(recent_positions[i-1]["center"])
                    pos2 = np.array(recent_positions[i]["center"])
                    distance = np.linalg.norm(pos2 - pos1)
                    total_distance += distance
                
                # If significant movement detected
                if total_distance > 100:  # Arbitrary threshold
                    # Check if near a person
                    for obj in objects:
                        if obj["class_name"] == "person":
                            person_id = obj.get("object_id", -1)
                            if person_id == -1:
                                continue
                                
                            person_center = np.array(obj.get("box_center",
                                                   [(obj["box"][0] + obj["box"][2])/2,
                                                    (obj["box"][1] + obj["box"][3])/2]))
                            weapon_center = np.array(weapon.get("box_center",
                                                   [(weapon["box"][0] + weapon["box"][2])/2,
                                                    (weapon["box"][1] + weapon["box"][3])/2]))
                            
                            distance = np.linalg.norm(person_center - weapon_center)
                            
                            if distance < 150:  # Arbitrary pixel distance
                                brandishing_events.append({
                                    "type": "weapon_brandishing",
                                    "subtype": "rapid_movement",
                                    "weapon_id": weapon_id,
                                    "weapon_class": weapon["class_name"],
                                    "person_id": person_id,
                                    "movement": total_distance,
                                    "confidence": weapon.get("confidence", 0),
                                    "frame": self.frame_history[-1]["frame_idx"],
                                    "timestamp": self.frame_history[-1]["timestamp"],
                                    "severity": self.suspicious_patterns["weapon_brandishing"]["severity"]
                                })
                                break  # Only report one brandishing event per weapon
        
        return brandishing_events
    
    def _detect_temporal_interactions(self):
        """Detect interactions that occur over time (e.g., person exiting car)."""
        interactions = []
        
        # For each object pair, check for specific temporal patterns
        for obj_id, history in self.object_history.items():
            # Skip if not enough history
            if len(history["positions"]) < 2:
                continue
                
            # Look for other objects that were near this object
            for other_id, other_history in self.object_history.items():
                # Skip if same object or not enough history
                if obj_id == other_id or len(other_history["positions"]) < 2:
                    continue
                
                # Check for person exiting vehicle
                if (history["class_name"] == "person" and 
                    other_history["class_name"] in ["car", "truck", "bus"]):
                    # Check if person was near vehicle at some point
                    possible_exit = self._check_person_vehicle_exit(
                        history["positions"], 
                        other_history["positions"]
                    )
                    
                    if possible_exit:
                        interactions.append({
                            "type": "person_exited_vehicle",
                            "person_id": obj_id,
                            "vehicle_id": other_id,
                            "vehicle_class": other_history["class_name"],
                            "vehicle_color": other_history.get("properties", {}).get("color", "unknown"),
                            "frame": possible_exit["frame"],
                            "timestamp": possible_exit["timestamp"]
                        })
                        
                # Check for person entering vehicle
                if (history["class_name"] == "person" and 
                    other_history["class_name"] in ["car", "truck", "bus"]):
                    # Check if person was near vehicle at some point
                    possible_entry = self._check_person_vehicle_entry(
                        history["positions"], 
                        other_history["positions"]
                    )
                    
                    if possible_entry:
                        interactions.append({
                            "type": "person_entered_vehicle",
                            "person_id": obj_id,
                            "vehicle_id": other_id,
                            "vehicle_class": other_history["class_name"],
                            "vehicle_color": other_history.get("properties", {}).get("color", "unknown"),
                            "frame": possible_entry["frame"],
                            "timestamp": possible_entry["timestamp"]
                        })
        
        return interactions
    
    def _check_person_vehicle_exit(self, person_positions, vehicle_positions):
        """Check if a person exited a vehicle based on position histories."""
        # Implementation remains the same as in the original class
        # For brevity, assuming the original implementation works
        return None  # Placeholder
    
    def _check_person_vehicle_entry(self, person_positions, vehicle_positions):
        """Check if a person entered a vehicle based on position histories."""
        # Implementation would be similar to exit but reversed
        return None  # Placeholder
    
    def _generate_events(self, spatial_interactions, temporal_interactions, 
                        weapon_interactions, frame_idx, timestamp):
        """Generate event descriptions from detected interactions."""
        events = []
        
        # Process spatial interactions
        for interaction in spatial_interactions:
            if interaction["type"] == "proximity":
                obj1_id, obj2_id = interaction["objects"]
                class1, class2 = interaction["classes"]
                
                # Get object details
                obj1 = self.object_history.get(obj1_id, {})
                obj2 = self.object_history.get(obj2_id, {})
                
                # Generate proximity event
                events.append({
                    "type": "proximity",
                    "description": f"{class1} (ID: {obj1_id}) is near {class2} (ID: {obj2_id})",
                    "objects": [obj1_id, obj2_id],
                    "classes": [class1, class2],
                    "frame": frame_idx,
                    "timestamp": timestamp,
                    "details": {
                        "distance": interaction["distance"],
                        "normalized_distance": interaction["normalized_distance"]
                    }
                })
        
        # Process temporal interactions
        for interaction in temporal_interactions:
            if interaction["type"] == "person_exited_vehicle":
                person_id = interaction["person_id"]
                vehicle_id = interaction["vehicle_id"]
                vehicle_class = interaction["vehicle_class"]
                vehicle_color = interaction["vehicle_color"]
                
                # Generate car exit event
                events.append({
                    "type": "person_exited_vehicle",
                    "description": f"Person (ID: {person_id}) exited {vehicle_color} {vehicle_class} (ID: {vehicle_id})",
                    "objects": [person_id, vehicle_id],
                    "classes": ["person", vehicle_class],
                    "frame": interaction["frame"],
                    "timestamp": interaction["timestamp"],
                    "details": {
                        "vehicle_color": vehicle_color
                    }
                })
            
            elif interaction["type"] == "person_entered_vehicle":
                person_id = interaction["person_id"]
                vehicle_id = interaction["vehicle_id"]
                vehicle_class = interaction["vehicle_class"]
                vehicle_color = interaction["vehicle_color"]
                
                # Generate car entry event
                events.append({
                    "type": "person_entered_vehicle",
                    "description": f"Person (ID: {person_id}) entered {vehicle_color} {vehicle_class} (ID: {vehicle_id})",
                    "objects": [person_id, vehicle_id],
                    "classes": ["person", vehicle_class],
                    "frame": interaction["frame"],
                    "timestamp": interaction["timestamp"],
                    "details": {
                        "vehicle_color": vehicle_color
                    }
                })
        
        # Process weapon interactions (these are high priority)
        for interaction in weapon_interactions:
            if interaction["type"] == "person_weapon_proximity":
                weapon_id = interaction["weapon_id"]
                person_id = interaction["person_id"]
                weapon_class = interaction["weapon_class"]
                
                # Generate weapon proximity alert
                alert = {
                    "type": "person_weapon_proximity",
                    "description": f"ALERT: Person (ID: {person_id}) is close to a {weapon_class} (ID: {weapon_id})",
                    "objects": [person_id, weapon_id],
                    "classes": ["person", weapon_class],
                    "frame": frame_idx,
                    "timestamp": timestamp,
                    "severity": interaction["severity"],
                    "details": {
                        "weapon_confidence": interaction["weapon_confidence"],
                        "distance": interaction["distance"]
                    }
                }
                events.append(alert)
                self.weapon_alerts.append(alert)
                
            elif interaction["type"] == "concealed_weapon":
                weapon_id = interaction["weapon_id"]
                person_id = interaction["person_id"]
                weapon_class = interaction["weapon_class"]
                
                # Generate concealed weapon alert
                alert = {
                    "type": "concealed_weapon",
                    "description": f"ALERT: Person (ID: {person_id}) may have concealed {weapon_class} (ID: {weapon_id})",
                    "objects": [person_id, weapon_id],
                    "classes": ["person", weapon_class],
                    "frame": frame_idx,
                    "timestamp": timestamp,
                    "severity": interaction["severity"],
                    "details": {
                        "weapon_confidence": interaction["weapon_confidence"],
                        "overlap_ratio": interaction["overlap_ratio"]
                    }
                }
                events.append(alert)
                self.weapon_alerts.append(alert)
                
            elif interaction["type"] == "weapon_handoff":
                weapon_id = interaction["weapon_id"]
                from_person = interaction["from_person"]
                to_person = interaction["to_person"]
                weapon_class = interaction["weapon_class"]
                
                # Generate weapon handoff alert
                alert = {
                    "type": "weapon_handoff",
                    "description": f"ALERT: Possible {weapon_class} (ID: {weapon_id}) handoff from person {from_person} to person {to_person}",
                    "objects": [from_person, to_person, weapon_id],
                    "classes": ["person", "person", weapon_class],
                    "frame": interaction["frame"],
                    "timestamp": interaction["timestamp"],
                    "severity": interaction["severity"]
                }
                events.append(alert)
                self.weapon_alerts.append(alert)
                
            elif interaction["type"] == "weapon_brandishing":
                weapon_id = interaction["weapon_id"]
                person_id = interaction["person_id"]
                weapon_class = interaction["weapon_class"]
                
                # Generate brandishing alert
                alert = {
                    "type": "weapon_brandishing",
                    "description": f"ALERT: Person (ID: {person_id}) may be brandishing a {weapon_class} (ID: {weapon_id})",
                    "objects": [person_id, weapon_id],
                    "classes": ["person", weapon_class],
                    "frame": interaction["frame"],
                    "timestamp": interaction["timestamp"],
                    "severity": interaction["severity"],
                    "details": {
                        "confidence": interaction["confidence"],
                        "movement": interaction.get("movement")
                    }
                }
                events.append(alert)
                self.weapon_alerts.append(alert)
        
        return events
    
    def get_all_events(self):
        """Get all interaction events detected so far."""
        return self.interaction_events
    
    def get_weapon_alerts(self, min_severity=0):
        """Get all weapon-related alerts with minimum severity."""
        if min_severity > 0:
            return [alert for alert in self.weapon_alerts 
                    if alert.get("severity", 0) >= min_severity]
        return self.weapon_alerts
    
    def generate_report(self, include_severity=True):
        """Generate a textual report of all significant interactions."""
        report = []
        report.append("# Interaction Report")
        report.append("")
        
        # First report weapon alerts (high priority)
        if self.weapon_alerts:
            report.append("## ⚠️ WEAPON ALERTS ⚠️")
            report.append("")
            
            # Sort by severity
            sorted_alerts = sorted(self.weapon_alerts, 
                                 key=lambda x: x.get("severity", 0), reverse=True)
            
            for i, alert in enumerate(sorted_alerts):
                severity = alert.get("severity", 0)
                severity_stars = "⚠️" * severity if include_severity else ""
                time_str = f"{alert['timestamp']:.2f}s"
                report.append(f"{i+1}. [{time_str}] {severity_stars} {alert['description']}")
            
            report.append("")
        
        # Group other events by type
        event_types = {}
        for event in self.interaction_events:
            # Skip weapon alerts as they're already reported
            if event["type"] in ["person_weapon_proximity", "concealed_weapon", 
                               "weapon_handoff", "weapon_brandishing"]:
                continue
                
            event_type = event["type"]
            if event_type not in event_types:
                event_types[event_type] = []
            event_types[event_type].append(event)
        
        # Report on each type
        for event_type, events in event_types.items():
            report.append(f"## {event_type.replace('_', ' ').title()}")
            report.append(f"Total: {len(events)}")
            report.append("")
            
            # List each event
            for i, event in enumerate(sorted(events, key=lambda e: e["timestamp"])):
                time_str = f"{event['timestamp']:.2f}s"
                report.append(f"{i+1}. [{time_str}] {event['description']}")
            
            report.append("")
        
        return "\n".join(report)
    
    def generate_security_report(self):
        """Generate a security-focused report highlighting suspicious activities."""
        report = []
        report.append("# Security Analysis Report")
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Add summary
        total_alerts = len(self.weapon_alerts)
        max_severity = max([alert.get("severity", 0) for alert in self.weapon_alerts]) if self.weapon_alerts else 0
        
        report.append("## Summary")
        report.append(f"Total Security Alerts: {total_alerts}")
        report.append(f"Maximum Alert Severity: {max_severity}/5")
        report.append("")
        
        # Add severity explanation
        report.append("## Severity Levels")
        report.append("1 ⚠️ - Low concern")
        report.append("2 ⚠️⚠️ - Moderate concern")
        report.append("3 ⚠️⚠️⚠️ - High concern")
        report.append("4 ⚠️⚠️⚠️⚠️ - Very high concern")
        report.append("5 ⚠️⚠️⚠️⚠️⚠️ - Critical concern")
        report.append("")
        
        # Group alerts by type
        alert_types = {}
        for alert in self.weapon_alerts:
            alert_type = alert["type"]
            if alert_type not in alert_types:
                alert_types[alert_type] = []
            alert_types[alert_type].append(alert)
        
        # Report on each type of alert
        for alert_type, alerts in alert_types.items():
            report.append(f"## {alert_type.replace('_', ' ').title()}")
            report.append(f"Total Incidents: {len(alerts)}")
            report.append("")
            
            # Sort by severity
            sorted_alerts = sorted(alerts, key=lambda x: x.get("severity", 0), reverse=True)
            
            # List each alert
            for i, alert in enumerate(sorted_alerts):
                severity = alert.get("severity", 0)
                severity_stars = "⚠️" * severity
                time_str = f"{alert['timestamp']:.2f}s"
                report.append(f"{i+1}. [{time_str}] {severity_stars} {alert['description']}")
                
                # Add details
                if "details" in alert:
                    report.append("   Details:")
                    for key, value in alert["details"].items():
                        if isinstance(value, float):
                            report.append(f"   - {key}: {value:.2f}")
                        else:
                            report.append(f"   - {key}: {value}")
            
            report.append("")
        
        return "\n".join(report)