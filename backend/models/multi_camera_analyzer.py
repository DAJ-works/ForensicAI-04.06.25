import os
import cv2
import numpy as np
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt

from .video_analyzer_reid import VideoAnalyzerWithReID
from .person_reid import PersonReidentifier

class MultiCameraAnalyzer:
    """
    Analyzes multiple camera feeds and correlates identities across cameras.
    """
    
    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        case_id: Optional[str] = None,
        reidentifier: Optional[PersonReidentifier] = None
    ):
        """
        Initialize the multi-camera analyzer.
        
        Parameters:
        -----------
        output_dir : str or Path, optional
            Directory to save results. If None, creates a timestamped directory.
        case_id : str, optional
            Unique identifier for this case/analysis session
        reidentifier : PersonReidentifier, optional
            Person re-identification system. If None, creates a new one.
        """
        # Create output directory if not provided
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(f"./data/multi_camera/{timestamp}")
        else:
            output_dir = Path(output_dir)
        
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate case ID if not provided
        self.case_id = case_id or f"case_{int(time.time())}"
        
        # Initialize re-identifier (shared across all cameras)
        self.reidentifier = reidentifier or PersonReidentifier()
        
        # Storage for camera analyzers
        self.camera_analyzers = {}
        
        # Storage for analysis results
        self.results = {
            "case_id": self.case_id,
            "timestamp": datetime.now().isoformat(),
            "cameras": {},
            "cross_camera_identities": {},
            "timeline": []
        }
        
        # Initialize log
        self.analysis_log = []
        self._log_step("Initialized multi-camera analyzer")
    
    def _log_step(self, step_name, details=None):
        """Log an analysis step with timestamp."""
        log_entry = {
            "step": step_name,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        self.analysis_log.append(log_entry)
        
        # Save log to file
        log_path = self.output_dir / "analysis_log.json"
        with open(log_path, "w") as f:
            json.dump(self.analysis_log, f, indent=4)
    
    def add_camera(
        self,
        camera_id: str,
        video_path: str,
        timestamp_offset: float = 0.0,
        description: str = ""
    ) -> None:
        """
        Add a camera feed to the multi-camera system.
        
        Parameters:
        -----------
        camera_id : str
            Unique identifier for this camera
        video_path : str
            Path to the video file
        timestamp_offset : float
            Time offset in seconds (for synchronization)
        description : str
            Camera description (e.g., "Front Entrance")
        """
        # Create camera directory
        camera_dir = self.output_dir / camera_id
        camera_dir.mkdir(exist_ok=True)
        
        # Create camera analyzer with shared reidentifier
        camera_analyzer = VideoAnalyzerWithReID(
            output_dir=camera_dir,
            case_id=f"{self.case_id}_{camera_id}",
            reidentifier=self.reidentifier
        )
        
        # Store in dictionary
        self.camera_analyzers[camera_id] = camera_analyzer
        
        # Add to results
        self.results["cameras"][camera_id] = {
            "video_path": video_path,
            "timestamp_offset": timestamp_offset,
            "description": description,
            "analysis_path": None
        }
        
        self._log_step(f"Added camera {camera_id}", {
            "video_path": video_path,
            "timestamp_offset": timestamp_offset
        })
    
    def analyze_all_cameras(
        self,
        frame_interval: int = 1,
        save_video: bool = True,
        include_classes: Optional[List[str]] = None
    ) -> Dict:
        """
        Analyze all camera feeds.
        
        Parameters:
        -----------
        frame_interval : int
            Process every nth frame
        save_video : bool
            If True, save the annotated videos
        include_classes : List[str], optional
            List of class names to include (None = all)
            
        Returns:
        --------
        Dict
            Combined analysis results
        """
        # Process each camera
        for camera_id, analyzer in self.camera_analyzers.items():
            print(f"Processing camera: {camera_id}")
            
            video_path = self.results["cameras"][camera_id]["video_path"]
            
            # Run analysis
            camera_results = analyzer.analyze_video(
                video_path=video_path,
                frame_interval=frame_interval,
                save_video=save_video,
                include_classes=include_classes,
                draw_trajectories=True,
                visualize_reid=True
            )
            
            # Store results path
            results_path = Path(analyzer.output_dir) / f"{Path(video_path).stem}_analysis.json"
            self.results["cameras"][camera_id]["analysis_path"] = str(results_path)
        
        # Build cross-camera identity map
        self._build_identity_map()
        
        # Create unified timeline
        self._create_unified_timeline()
        
        # Generate cross-camera report
        self._generate_cross_camera_report()
        
        # Save final results
        results_path = self.output_dir / "multi_camera_analysis.json"
        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=4)
        
        return self.results
    
    def _build_identity_map(self) -> None:
        """Build a map of identities across cameras."""
        # PersonReidentifier already maintains a global identity database
        # We just need to organize the data by camera
        
        # Get all person IDs from the reidentifier database
        person_ids = set()
        for person_id in self.reidentifier.database.persons:
            person_ids.add(person_id)
        
        # For each ID, find appearances in each camera
        for person_id in person_ids:
            person_data = self.reidentifier.database.get_person(person_id)
            if not person_data:
                continue
                
            # Create entry for this person
            self.results["cross_camera_identities"][person_id] = {
                "id": person_id,
                "camera_appearances": {},
                "first_seen": None,
                "last_seen": None,
                "total_appearances": 0
            }
            
            # Collect appearances from each camera
            for camera_id, camera_info in self.results["cameras"].items():
                analysis_path = camera_info["analysis_path"]
                if not analysis_path or not os.path.exists(analysis_path):
                    continue
                    
                # Load camera analysis
                with open(analysis_path, "r") as f:
                    camera_data = json.load(f)
                
                # Check each frame for this person
                appearances = []
                for frame_key, frame_data in camera_data.get("frames", {}).items():
                    for det in frame_data.get("detections", []):
                        if det.get("person_id") == person_id:
                            appearances.append({
                                "frame": frame_data["frame_number"],
                                "timestamp": frame_data["timestamp"] + camera_info["timestamp_offset"],
                                "box": det["box"],
                                "confidence": det.get("reid_score", det.get("confidence", 0))
                            })
                
                # If found appearances, add to results
                if appearances:
                    # Sort by timestamp
                    appearances.sort(key=lambda x: x["timestamp"])
                    
                    # Add to person data
                    self.results["cross_camera_identities"][person_id]["camera_appearances"][camera_id] = {
                        "first_seen": appearances[0]["timestamp"],
                        "last_seen": appearances[-1]["timestamp"],
                        "appearances": len(appearances),
                        "samples": appearances[:5]  # Store just a few samples
                    }
                    
                    # Update overall first/last seen
                    current_first = self.results["cross_camera_identities"][person_id]["first_seen"]
                    current_last = self.results["cross_camera_identities"][person_id]["last_seen"]
                    
                    if current_first is None or appearances[0]["timestamp"] < current_first:
                        self.results["cross_camera_identities"][person_id]["first_seen"] = appearances[0]["timestamp"]
                        
                    if current_last is None or appearances[-1]["timestamp"] > current_last:
                        self.results["cross_camera_identities"][person_id]["last_seen"] = appearances[-1]["timestamp"]
                    
                    # Update total appearances
                    self.results["cross_camera_identities"][person_id]["total_appearances"] += len(appearances)
        
        # Remove any identities with no camera appearances
        to_remove = []
        for person_id, data in self.results["cross_camera_identities"].items():
            if not data["camera_appearances"]:
                to_remove.append(person_id)
                
        for person_id in to_remove:
            del self.results["cross_camera_identities"][person_id]
        
        self._log_step("Built identity map", {
            "total_identities": len(self.results["cross_camera_identities"])
        })
    
    def _create_unified_timeline(self) -> None:
        """Create a unified timeline of events across all cameras."""
        timeline_events = []
        
        # 1. Person appearances in each camera
        for person_id, person_data in self.results["cross_camera_identities"].items():
            for camera_id, camera_data in person_data["camera_appearances"].items():
                # Add first appearance
                timeline_events.append({
                    "event_type": "person_appearance",
                    "person_id": person_id,
                    "camera_id": camera_id,
                    "timestamp": camera_data["first_seen"],
                    "description": f"Person #{person_id} first appeared on camera {camera_id}"
                })
                
                # Add last appearance
                timeline_events.append({
                    "event_type": "person_disappearance",
                    "person_id": person_id,
                    "camera_id": camera_id,
                    "timestamp": camera_data["last_seen"],
                    "description": f"Person #{person_id} last seen on camera {camera_id}"
                })
        
        # 2. Add camera transitions
        for person_id, person_data in self.results["cross_camera_identities"].items():
            # Only look at people seen in multiple cameras
            if len(person_data["camera_appearances"]) < 2:
                continue
                
            # Get all appearances across cameras
            camera_events = []
            for camera_id, camera_data in person_data["camera_appearances"].items():
                camera_events.append({
                    "camera_id": camera_id,
                    "timestamp": camera_data["first_seen"],
                    "event_type": "first_seen"
                })
                camera_events.append({
                    "camera_id": camera_id,
                    "timestamp": camera_data["last_seen"],
                    "event_type": "last_seen"
                })
            
            # Sort by timestamp
            camera_events.sort(key=lambda x: x["timestamp"])
            
            # Find transitions
            for i in range(1, len(camera_events)):
                current = camera_events[i]
                previous = camera_events[i-1]
                
                # If this is a first_seen and the previous was a last_seen on a different camera
                if (current["event_type"] == "first_seen" and 
                    previous["event_type"] == "last_seen" and
                    current["camera_id"] != previous["camera_id"]):
                    
                    # Calculate time gap
                    time_gap = current["timestamp"] - previous["timestamp"]
                    
                    # Only add if the gap is reasonably small
                    if 0 < time_gap < 300:  # 5 minutes max
                        timeline_events.append({
                            "event_type": "camera_transition",
                            "person_id": person_id,
                            "from_camera": previous["camera_id"],
                            "to_camera": current["camera_id"],
                            "timestamp": previous["timestamp"],
                            "transition_time": time_gap,
                            "description": f"Person #{person_id} moved from camera {previous['camera_id']} to {current['camera_id']} in {time_gap:.1f} seconds"
                        })
        
        # Sort all events by timestamp
        timeline_events.sort(key=lambda x: x["timestamp"])
        
        # Add to results
        self.results["timeline"] = timeline_events
        
        self._log_step("Created unified timeline", {
            "total_events": len(timeline_events)
        })
    
    def _generate_cross_camera_report(self) -> None:
        """Generate a report on cross-camera movements and identity matches."""
        report = {
            "case_id": self.case_id,
            "timestamp": datetime.now().isoformat(),
            "camera_count": len(self.camera_analyzers),
            "person_count": len(self.results["cross_camera_identities"]),
            "multi_camera_persons": 0,
            "camera_transitions": [],
            "person_summaries": []
        }
        
        # Count people seen in multiple cameras
        for person_id, person_data in self.results["cross_camera_identities"].items():
            if len(person_data["camera_appearances"]) > 1:
                report["multi_camera_persons"] += 1
        
        # Extract camera transitions
        for event in self.results["timeline"]:
            if event["event_type"] == "camera_transition":
                report["camera_transitions"].append(event)
        
        # Create person summaries
        for person_id, person_data in self.results["cross_camera_identities"].items():
            cameras_seen = list(person_data["camera_appearances"].keys())
            
            # Get total time visible
            total_visible_time = 0
            for camera_id, camera_data in person_data["camera_appearances"].items():
                visible_time = camera_data["last_seen"] - camera_data["first_seen"]
                total_visible_time += visible_time
            
            # Create summary
            summary = {
                "person_id": person_id,
                "cameras_seen": cameras_seen,
                "camera_count": len(cameras_seen),
                "first_seen": person_data["first_seen"],
                "last_seen": person_data["last_seen"],
                "total_time_visible": total_visible_time,
                "total_appearances": person_data["total_appearances"]
            }
            
            report["person_summaries"].append(summary)
        
        # Sort person summaries by total appearances
        report["person_summaries"].sort(key=lambda x: x["total_appearances"], reverse=True)
        
        # Save report
        report_path = self.output_dir / "cross_camera_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
        
        # Create visualizations
        self._create_cross_camera_visualizations(report)
        
        self._log_step("Generated cross-camera report", {
            "multi_camera_persons": report["multi_camera_persons"],
            "transitions": len(report["camera_transitions"])
        })
    
    def _create_cross_camera_visualizations(self, report: Dict) -> None:
        """Create visualizations for the cross-camera report."""
        vis_dir = self.output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        
        # 1. Person flow between cameras
        self._visualize_camera_flow(report, str(vis_dir / "camera_flow.png"))
        
        # 2. Person timeline across cameras
        self._visualize_cross_camera_timeline(
            self.results["cross_camera_identities"],
            str(vis_dir / "cross_camera_timeline.png")
        )
        
        # 3. Camera occupancy over time
        self._visualize_camera_occupancy(
            self.results["cross_camera_identities"],
            str(vis_dir / "camera_occupancy.png")
        )
    
    def _visualize_camera_flow(self, report: Dict, output_path: str):
        """Visualize flow of people between cameras."""
        # Count transitions between each camera pair
        transitions = {}
        for transition in report["camera_transitions"]:
            from_camera = transition["from_camera"]
            to_camera = transition["to_camera"]
            
            key = (from_camera, to_camera)
            if key not in transitions:
                transitions[key] = 0
            transitions[key] += 1
        
        # If no transitions, skip
        if not transitions:
            return
        
        # Get list of all cameras
        all_cameras = set()
        for (from_camera, to_camera) in transitions.keys():
            all_cameras.add(from_camera)
            all_cameras.add(to_camera)
        
        # Sort cameras for consistent ordering
        sorted_cameras = sorted(all_cameras)
        
        # Create directed graph
        import networkx as nx
        G = nx.DiGraph()
        
        # Add nodes
        for camera in sorted_cameras:
            G.add_node(camera)
        
        # Add edges with weights
        for (from_camera, to_camera), count in transitions.items():
            G.add_edge(from_camera, to_camera, weight=count)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Set up positions
        pos = nx.spring_layout(G, seed=42)
        
        # Get edge weights for width and color
        weights = [G[u][v]['weight'] for u, v in G.edges()]
        max_weight = max(weights) if weights else 1
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue')
        nx.draw_networkx_labels(G, pos, font_size=12)
        
        # Draw edges with varying width and color
        for (u, v, w) in G.edges(data=True):
            width = w['weight'] / max_weight * 5.0  # Scale edge width
            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v)], 
                width=width,
                alpha=0.7,
                edge_color='darkblue',
                arrowsize=20
            )
            
            # Add edge label for count
            edge_label = {(u, v): str(w['weight'])}
            nx.draw_networkx_edge_labels(
                G, pos,
                edge_labels=edge_label,
                font_size=10,
                label_pos=0.3
            )
        
        plt.title("Person Flow Between Cameras")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    
    def _visualize_cross_camera_timeline(
        self,
        identities: Dict,
        output_path: str
    ):
        """Visualize person presence across multiple cameras."""
        # Only include people seen in multiple cameras
        multi_camera_persons = {}
        for person_id, person_data in identities.items():
            if len(person_data["camera_appearances"]) > 1:
                multi_camera_persons[person_id] = person_data
        
        if not multi_camera_persons:
            return
        
        # Sort by first seen time
        sorted_persons = sorted(
            multi_camera_persons.items(),
            key=lambda x: x[1]["first_seen"]
        )
        
        # Get all cameras
        all_cameras = set()
        for _, person_data in sorted_persons:
            for camera_id in person_data["camera_appearances"].keys():
                all_cameras.add(camera_id)
        
        # Sort cameras
        sorted_cameras = sorted(all_cameras)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, len(sorted_persons) * 1.5))
        
        # Get global min and max times
        min_time = min(p[1]["first_seen"] for p in sorted_persons)
        max_time = max(p[1]["last_seen"] for p in sorted_persons)
        
        # Add buffer
        time_range = max_time - min_time
        min_time -= time_range * 0.02
        max_time += time_range * 0.02
        
        # Create a distinct color for each camera
        camera_colors = {}
        for i, camera_id in enumerate(sorted_cameras):
            hue = i / len(sorted_cameras)
            camera_colors[camera_id] = plt.cm.hsv(hue)
        
        # Plot each person's appearances
        for i, (person_id, person_data) in enumerate(sorted_persons):
            y_pos = i
            
            # Plot line for person
            ax.plot([min_time, max_time], [y_pos, y_pos], '-', color='lightgray', alpha=0.5)
            
            # Plot appearances in each camera
            for camera_id, camera_data in person_data["camera_appearances"].items():
                first_seen = camera_data["first_seen"]
                last_seen = camera_data["last_seen"]
                
                # Get camera color
                color = camera_colors[camera_id]
                
                # Plot appearance segment
                ax.plot(
                    [first_seen, last_seen],
                    [y_pos, y_pos],
                    '-',
                    linewidth=8,
                    color=color,
                    solid_capstyle="butt",
                    label=camera_id
                )
                
                # Mark start and end
                ax.plot(first_seen, y_pos, 'o', color=color, markersize=6)
                ax.plot(last_seen, y_pos, 'o', color=color, markersize=6)
        
        # Add camera legend
        handles = []
        labels = []
        for camera_id, color in camera_colors.items():
            handles.append(plt.Line2D([0], [0], color=color, lw=4))
            labels.append(f"Camera {camera_id}")
        
        ax.legend(handles, labels, loc='upper right')
        
        # Set labels and title
        ax.set_xlabel("Time (seconds)")
        ax.set_yticks(range(len(sorted_persons)))
        ax.set_yticklabels([f"Person #{p[0]}" for p in sorted_persons])
        ax.set_title("Cross-Camera Person Timeline")
        
        # Set time limits
        ax.set_xlim(min_time, max_time)
        
        # Add grid
        ax.grid(True, axis='x', linestyle='--', alpha=0.6)
        
        # Tight layout
        plt.tight_layout()
        
        # Save
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    
    def _visualize_camera_occupancy(
        self,
        identities: Dict,
        output_path: str
    ):
        """Visualize camera occupancy over time (how many people in each camera)."""
        # Get all cameras
        all_cameras = set()
        for _, person_data in identities.items():
            for camera_id in person_data["camera_appearances"].keys():
                all_cameras.add(camera_id)
        
        if not all_cameras:
            return
        
        # Sort cameras
        sorted_cameras = sorted(all_cameras)
        
        # Collect presence events for each camera
        camera_events = {camera_id: [] for camera_id in sorted_cameras}
        
        for person_id, person_data in identities.items():
            for camera_id, camera_data in person_data["camera_appearances"].items():
                first_seen = camera_data["first_seen"]
                last_seen = camera_data["last_seen"]
                
                # Add entry and exit events
                camera_events[camera_id].append(("enter", first_seen))
                camera_events[camera_id].append(("exit", last_seen))
        
        # Get global min and max times
        all_times = []
        for events in camera_events.values():
            all_times.extend([t for _, t in events])
        
        min_time = min(all_times)
        max_time = max(all_times)
        
        # Add buffer
        time_range = max_time - min_time
        min_time -= time_range * 0.02
        max_time += time_range * 0.02
        
        # Create time points for evaluation
        time_points = np.linspace(min_time, max_time, 1000)
        
        # Create occupancy data for each camera
        occupancy_data = {}
        for camera_id, events in camera_events.items():
            # Sort events by time
            sorted_events = sorted(events, key=lambda x: x[1])
            
            # Calculate occupancy at each time point
            occupancy = []
            for t in time_points:
                count = 0
                for event_type, event_time in sorted_events:
                    if event_time <= t:
                        if event_type == "enter":
                            count += 1
                        else:  # exit
                            count -= 1
                    else:
                        break
                        
                occupancy.append(max(0, count))  # Ensure non-negative
            
            occupancy_data[camera_id] = occupancy
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot occupancy for each camera
        for camera_id, occupancy in occupancy_data.items():
            plt.plot(time_points, occupancy, label=f"Camera {camera_id}", linewidth=2)
        
        # Set labels and title
        plt.xlabel("Time (seconds)")
        plt.ylabel("Number of People")
        plt.title("Camera Occupancy Over Time")
        
        # Add legend
        plt.legend(loc='upper right')
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Set limits
        plt.xlim(min_time, max_time)
        plt.ylim(bottom=0)
        
        # Tight layout
        plt.tight_layout()
        
        # Save
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()