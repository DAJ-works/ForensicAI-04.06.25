import os
import sys
import argparse
import cv2
import time
from pathlib import Path

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.models.person_reid import PersonReidentifier
from backend.models.multi_camera_analyzer import MultiCameraAnalyzer

def main():
    parser = argparse.ArgumentParser(description="Test multi-camera analysis with person re-identification")
    parser.add_argument("--output-dir", help="Output directory", default=None)
    parser.add_argument("--frame-interval", type=int, default=1, help="Process every nth frame")
    parser.add_argument("--reid-threshold", type=float, default=0.6, help="ReID similarity threshold")
    parser.add_argument("--cameras", type=str, nargs="+", required=True, 
                       help="Camera videos in format: id:path[:offset][:description]")
    
    args = parser.parse_args()
    
    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"./data/multi_camera_analysis/{timestamp}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Initialize reidentifier
    print(f"Initializing person re-identification with threshold {args.reid_threshold}")
    reidentifier = PersonReidentifier(similarity_threshold=args.reid_threshold)
    
    # Initialize multi-camera analyzer
    analyzer = MultiCameraAnalyzer(
        output_dir=output_dir,
        reidentifier=reidentifier
    )
    
    # Parse camera arguments and add to analyzer
    for camera_arg in args.cameras:
        parts = camera_arg.split(":")
        
        # Handle different argument formats
        if len(parts) == 2:
            camera_id, video_path = parts
            offset, description = 0.0, f"Camera {camera_id}"
        elif len(parts) == 3:
            camera_id, video_path, offset = parts
            description = f"Camera {camera_id}"
            offset = float(offset)
        elif len(parts) >= 4:
            camera_id, video_path, offset, *desc_parts = parts
            description = ":".join(desc_parts)  # Rejoin any colons in description
            offset = float(offset)
        else:
            print(f"Error: Invalid camera format: {camera_arg}")
            print("Use format: id:path[:offset][:description]")
            continue
        
        # Check if video exists
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            continue
        
        print(f"Adding camera: {camera_id} - {description}")
        print(f"  Video: {video_path}")
        print(f"  Time offset: {offset}s")
        
        analyzer.add_camera(
            camera_id=camera_id,
            video_path=video_path,
            timestamp_offset=offset,
            description=description
        )
    
    # Check if we have at least one camera
    if not analyzer.camera_analyzers:
        print("Error: No valid cameras to analyze")
        return 1
    
    # Run analysis
    print(f"Analyzing {len(analyzer.camera_analyzers)} cameras with frame interval {args.frame_interval}")
    results = analyzer.analyze_all_cameras(
        frame_interval=args.frame_interval,
        save_video=True,
        include_classes=["person"]  # Focus only on people
    )
    
    # Print summary
    print("\nMulti-Camera Analysis Summary:")
    print(f"Total cameras: {len(results['cameras'])}")
    
    person_identities = results["cross_camera_identities"]
    print(f"Total unique persons: {len(person_identities)}")
    
    # Count persons seen in multiple cameras
    multi_camera_persons = sum(1 for p in person_identities.values() 
                              if len(p["camera_appearances"]) > 1)
    print(f"Persons seen in multiple cameras: {multi_camera_persons}")
    
    # Print camera transitions
    transitions = [e for e in results["timeline"] if e["event_type"] == "camera_transition"]
    print(f"Camera transitions: {len(transitions)}")
    
    if transitions:
        print("\nCamera Transitions:")
        for i, transition in enumerate(transitions[:5]):
            print(f"  {i+1}. Person #{transition['person_id']} moved from camera {transition['from_camera']} " +
                  f"to {transition['to_camera']} at {transition['timestamp']:.2f}s " +
                  f"(transition time: {transition['transition_time']:.1f}s)")
            
        if len(transitions) > 5:
            print(f"  ... and {len(transitions) - 5} more transitions")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Report saved to: {output_dir / 'cross_camera_report.json'}")
    print(f"Visualizations saved to: {output_dir / 'visualizations'}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())