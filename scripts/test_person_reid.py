import os
import sys
import argparse
import cv2
import numpy as np
import time
from pathlib import Path

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.models.object_detector import ObjectDetector
from backend.models.object_tracker import ObjectTracker
from backend.models.person_reid import PersonReidentifier
from backend.models.video_analyzer_reid import VideoAnalyzerWithReID

def main():
    parser = argparse.ArgumentParser(description="Test person re-identification")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("--output-dir", help="Output directory", default=None)
    parser.add_argument("--model-size", help="YOLO model size (n, s, m, l, x)", default="m")
    parser.add_argument("--conf", type=float, help="Confidence threshold", default=0.4)
    parser.add_argument("--frame-interval", type=int, default=1, help="Process every nth frame")
    parser.add_argument("--reid-threshold", type=float, default=0.6, help="ReID similarity threshold")
    parser.add_argument("--save-frames", action="store_true", help="Save annotated frames")
    parser.add_argument("--query-image", help="Path to query image for person search", default=None)
    
    args = parser.parse_args()
    
    # Check if video exists
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return 1
    
    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Create a directory with the video name and timestamp
        video_name = Path(args.video_path).stem
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"./data/reid_analysis/{video_name}_{timestamp}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Initialize detector
    print(f"Initializing YOLOv8{args.model_size} detector with confidence threshold {args.conf}")
    detector = ObjectDetector(
        model_size=args.model_size,
        confidence_threshold=args.conf
    )
    
    # Initialize tracker and reidentifier
    print(f"Initializing person re-identification with threshold {args.reid_threshold}")
    reidentifier = PersonReidentifier(similarity_threshold=args.reid_threshold)
    
    # Initialize video analyzer with ReID
    analyzer = VideoAnalyzerWithReID(
        detector=detector,
        output_dir=output_dir,
        reidentifier=reidentifier,
        enable_reid=True
    )
    
    # Run analysis
    print(f"Running video analysis with person re-identification (frame interval: {args.frame_interval})")
    results = analyzer.analyze_video(
        video_path=args.video_path,
        frame_interval=args.frame_interval,
        save_video=True,
        save_frames=args.save_frames,
        include_classes=["person"],  # Focus only on people
        visualize_reid=True
    )
    
    # Generate report
    print("Generating analysis report...")
    report = analyzer.generate_analysis_report(results)
    
    # Print summary
    print("\nAnalysis Summary:")
    print(f"Total unique persons identified: {len(report['person_analysis']['identities'])}")
    print(f"Total detections: {report['summary']['total_detections']}")
    
    if report['person_analysis']['identities']:
        print("\nPerson Identities:")
        for i, person in enumerate(report['person_analysis']['identities']):
            print(f"  {i+1}. Person #{person['id']}: {person['total_appearances']} appearances, " +
                  f"visible for {person.get('duration', 0):.2f}s")
            
        print("\nPerson Interactions:")
        for i, interaction in enumerate(report['person_analysis']['interactions'][:5]):
            print(f"  {i+1}. Persons #{interaction['person1_id']} and #{interaction['person2_id']} " +
                  f"at {interaction['timestamp']:.2f}s (distance: {interaction['distance']:.1f} pixels)")
            
        if len(report['person_analysis']['interactions']) > 5:
            print(f"  ... and {len(report['person_analysis']['interactions']) - 5} more interactions")
    
    # If query image is provided, search for matching persons
    if args.query_image and os.path.exists(args.query_image):
        print(f"\nSearching for person in query image: {args.query_image}")
        search_results = reidentifier.query_person(args.query_image, top_k=5)
        
        if search_results:
            print("Matching Persons:")
            for i, match in enumerate(search_results):
                person_id = match['id']
                similarity = match['similarity']
                
                # Find person in report
                person_data = None
                for p in report['person_analysis']['identities']:
                    if p['id'] == person_id:
                        person_data = p
                        break
                
                appearances = person_data['total_appearances'] if person_data else "unknown"
                
                print(f"  {i+1}. Person #{person_id}: {similarity:.2f} similarity score, {appearances} appearances")
        else:
            print("No matching persons found.")
    
    print(f"\nOutput video: {results.get('output_video', 'Not saved')}")
    print(f"Report saved to: {output_dir / 'analysis_report.json'}")
    print(f"Person gallery saved to: {output_dir / 'person_gallery'}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())