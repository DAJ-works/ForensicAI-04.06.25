import cv2
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import hsv_to_rgb
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any

class ResultsVisualizer:
    """
    Visualization utilities for detection and tracking results.
    """
    
    def __init__(self, results_path: str, output_dir: Optional[str] = None):
        """
        Initialize the visualizer.
        
        Parameters:
        -----------
        results_path : str
            Path to the analysis results JSON file
        output_dir : str, optional
            Directory to save visualizations. If None, uses the same directory as results.
        """
        # Load results
        with open(results_path, 'r') as f:
            self.results = json.load(f)
        
        # Set output directory
        if output_dir is None:
            self.output_dir = Path(results_path).parent
        else:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract key data
        self.video_path = self.results.get('video_path')
        self.tracks = self.results.get('tracks', [])
        self.frames = self.results.get('frames', {})
        self.metadata = self.results.get('video_metadata', {})
        
        # Set up color mapping
        self.class_colors = {}
    
    def _get_class_color(self, class_name: str) -> Tuple[float, float, float]:
        """Get a consistent color for a class."""
        if class_name not in self.class_colors:
            # Generate a color based on class name hash
            hue = (hash(class_name) % 360) / 360.0
            self.class_colors[class_name] = hsv_to_rgb((hue, 0.8, 0.9))
        
        return self.class_colors[class_name]
    
    def plot_trajectory_map(self, save_path: Optional[str] = None) -> None:
        """
        Plot a map of all object trajectories.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot. If None, uses default naming.
        """
        if not self.tracks:
            print("No tracks available for visualization")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Set up plot
        width = self.metadata.get('width', 1920)
        height = self.metadata.get('height', 1080)
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)  # Invert y-axis to match image coordinates
        
        # Plot each trajectory
        for track in self.tracks:
            if len(track['trajectory']) < 2:
                continue  # Skip very short tracks
                
            # Get class color
            color = self._get_class_color(track['class_name'])
            
            # Extract trajectory points
            points = np.array(track['trajectory'])
            
            # Plot trajectory
            ax.plot(points[:, 0], points[:, 1], '-', color=color, linewidth=2, alpha=0.7,
                   label=f"{track['class_name']} ID:{track['object_id']}")
            
            # Mark start and end points
            ax.plot(points[0, 0], points[0, 1], 'o', color=color, markersize=8)
            ax.plot(points[-1, 0], points[-1, 1], 's', color=color, markersize=8)
            
            # Add ID label at end point
            ax.text(points[-1, 0] + 10, points[-1, 1] + 10, f"ID:{track['object_id']}",
                   color=color, fontsize=9, ha='left', va='bottom')
        
        # Set labels and title
        ax.set_xlabel('X Position (pixels)')
        ax.set_ylabel('Y Position (pixels)')
        ax.set_title('Object Trajectories Map')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Handle legend (unique classes only)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=9)
        
        # Tight layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path is None:
            save_path = self.output_dir / 'trajectory_map.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Trajectory map saved to {save_path}")
        
        # Show plot
        plt.close()
    
    def plot_class_distribution(self, save_path: Optional[str] = None) -> None:
        """
        Plot distribution of detected object classes.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot. If None, uses default naming.
        """
        class_counts = self.results.get('class_counts', {})
        
        if not class_counts:
            print("No class count data available")
            return
        
        # Sort by count
        classes = list(class_counts.keys())
        counts = [class_counts[c] for c in classes]
        
        # Sort by count (descending)
        sorted_indices = np.argsort(counts)[::-1]
        classes = [classes[i] for i in sorted_indices]
        counts = [counts[i] for i in sorted_indices]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create bar chart
        bars = ax.bar(classes, counts, color=[self._get_class_color(c) for c in classes])
        
        # Add count labels on top of bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{count}', ha='center', va='bottom', fontsize=9)
        
        # Set labels and title
        ax.set_xlabel('Object Class')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Detected Objects by Class')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Tight layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path is None:
            save_path = self.output_dir / 'class_distribution.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution plot saved to {save_path}")
        
        # Show plot
        plt.close()
    
    def plot_timeline(self, save_path: Optional[str] = None) -> None:
        """
        Plot a timeline of detected objects.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot. If None, uses default naming.
        """
        if not self.tracks:
            print("No tracks available for visualization")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Sort tracks by first appearance
        sorted_tracks = sorted(self.tracks, key=lambda x: x['timestamps'][0])
        
        # Get unique classes
        classes = set(track['class_name'] for track in sorted_tracks)
        
        # Map classes to y-positions
        class_positions = {cls: i for i, cls in enumerate(sorted(classes))}
        
        # Plot each track as a horizontal line
        for i, track in enumerate(sorted_tracks):
            class_name = track['class_name']
            y_pos = class_positions[class_name] + (i % 3) * 0.3  # Offset to avoid overlap
            
            # Get timestamps
            start_time = track['timestamps'][0]
            end_time = track['timestamps'][-1]
            
            # Get color
            color = self._get_class_color(class_name)
            
            # Plot track timeline
            ax.plot([start_time, end_time], [y_pos, y_pos], '-', linewidth=3, color=color)
            
            # Mark start and end
            ax.plot(start_time, y_pos, 'o', markersize=6, color=color)
            ax.plot(end_time, y_pos, 's', markersize=6, color=color)
            
            # Add ID label
            ax.text(end_time + 0.1, y_pos, f"ID:{track['object_id']}", fontsize=8,
                   va='center', ha='left', color=color)
        
        # Set y-ticks at class positions
        ax.set_yticks([class_positions[cls] + 0.3 for cls in sorted(classes)])
        ax.set_yticklabels(sorted(classes))
        
        # Set labels and title
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Object Class')
        ax.set_title('Timeline of Tracked Objects')
        
        # Set x limits
        video_duration = self.metadata.get('duration', 0)
        ax.set_xlim(-0.5, video_duration + 1)
        
        # Add grid
        ax.grid(True, axis='x', linestyle='--', alpha=0.6)
        
        # Tight layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path is None:
            save_path = self.output_dir / 'object_timeline.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Timeline plot saved to {save_path}")
        
        # Show plot
        plt.close()
    
    def plot_heatmap(self, class_filter: Optional[List[str]] = None, save_path: Optional[str] = None) -> None:
        """
        Plot heatmap of object positions.
        
        Parameters:
        -----------
        class_filter : List[str], optional
            List of classes to include. If None, includes all classes.
        save_path : str, optional
            Path to save the plot. If None, uses default naming.
        """
        if not self.tracks:
            print("No tracks available for visualization")
            return
        
        # Get video dimensions
        width = self.metadata.get('width', 1920)
        height = self.metadata.get('height', 1080)
        
        # Create grid for heatmap
        grid_size = 50  # Grid cells
        heatmap = np.zeros((grid_size, grid_size))
        
        # Scale points to grid
        scale_x = grid_size / width
        scale_y = grid_size / height
        
        # Accumulate points on grid
        for track in self.tracks:
            # Skip if not in class filter
            if class_filter and track['class_name'] not in class_filter:
                continue
                
            # Get trajectory points
            points = np.array(track['trajectory'])
            
            # Scale points to grid
            points_scaled = np.zeros_like(points)
            points_scaled[:, 0] = points[:, 0] * scale_x
            points_scaled[:, 1] = points[:, 1] * scale_y
            
            # Clip points to valid grid range
            points_scaled = np.clip(points_scaled, 0, grid_size - 1)
            
            # Add to heatmap
            for x, y in points_scaled:
                heatmap[int(y), int(x)] += 1
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot heatmap
        sns.heatmap(heatmap, cmap='hot', ax=ax)
        
        # Set labels and title
        title = 'Object Position Heatmap'
        if class_filter:
            title += f" ({', '.join(class_filter)})"
        ax.set_title(title)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        
        # Invert y-axis to match image coordinates
        ax.invert_yaxis()
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Tight layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path is None:
            save_path = self.output_dir / 'position_heatmap.png'
            if class_filter:
                class_str = '_'.join(class_filter)
                save_path = self.output_dir / f'position_heatmap_{class_str}.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to {save_path}")
        
        # Show plot
        plt.close()
    
    def create_report(self) -> None:
        """Generate all visualizations for a comprehensive report."""
        print("Generating comprehensive visualization report...")
        
        # Generate all plots
        self.plot_trajectory_map()
        self.plot_class_distribution()
        self.plot_timeline()
        self.plot_heatmap()
        
        # Create class-specific heatmaps for top classes
        class_counts = self.results.get('class_counts', {})
        top_classes = sorted(class_counts.keys(), key=lambda x: class_counts[x], reverse=True)[:3]
        
        for cls in top_classes:
            self.plot_heatmap([cls])
        
        print("Visualization report complete.")