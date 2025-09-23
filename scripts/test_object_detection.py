import os
import sys
import argparse
import time
from pathlib import Path

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.models.object_detector import ObjectDetector
from backend.models.object_tracker import ObjectTracker
from backend.models.video_analyzer import VideoAnalyzer

def main():
    parser = argparse.ArgumentParser(description="Test object detection and tracking")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("--output-dir", help="Output directory", default=None)
    parser.add_argument("--model-size", help="YOLO model size (n, s, m, l, x)", default="m")
    parser.add_argument("--conf", type=float, help="Confidence threshold", default=0.4)
    parser.add_argument("--frame-interval", type=int, default=1, help="Process every nth frame")
    parser.add_argument("--no-tracking", action="store_true", help="Disable tracking")
    parser.add_argument("--save-frames", action="store_true", help="Save annotated frames")
    parser.add_argument("--no-trajectories", action="store_true", help="Don't draw trajectories")
    parser.add_argument("--classes", nargs="+", help="Filter specific classes")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return 1
    
    print(f"Processing video: {args.video_path}")
    
    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Create a directory with the video name and timestamp
        video_name = Path(args.video_path).stem
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"./data/analysis/{video_name}_{timestamp}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Initialize detector
    print(f"Initializing YOLOv8{args.model_size} detector with confidence threshold {args.conf}")
    detector = ObjectDetector(
        model_size=args.model_size,
        confidence_threshold=args.conf
    )
    
    # If only detection is needed (no tracking)
    if args.no_tracking:
        print("Running object detection without tracking")
        results = detector.process_video(
            video_path=args.video_path,
            output_dir=output_dir,
            save_frames=args.save_frames,
            frame_interval=args.frame_interval,
            class_filter=args.classes
        )
        
        # Print summary
        print("\nDetection Summary:")
        print(f"Total detections: {results['total_detections']}")
        print("Detections by class:")
        for cls, count in results['class_counts'].items():
            print(f"  - {cls}: {count}")
        
        print(f"\nOutput video: {results.get('output_video', 'Not saved')}")
        return 0
    
    # Initialize tracker and analyzer
    tracker = ObjectTracker(max_age=30, min_hits=3)
    analyzer = VideoAnalyzer(
        detector=detector,
        tracker=tracker,
        output_dir=output_dir
    )
    
    # Run analysis
    print(f"Running video analysis with tracking (frame interval: {args.frame_interval})")
    results = analyzer.analyze_video(
        video_path=args.video_path,
        frame_interval=args.frame_interval,
        save_video=True,
        save_frames=args.save_frames,
        include_classes=args.classes,
        draw_trajectories=not args.no_trajectories
    )
    
    # Generate report
    print("Generating analysis report...")
    report = analyzer.generate_analysis_report(results)
    
    # Print summary
    print("\nAnalysis Summary:")
    print(f"Total unique objects tracked: {report['summary']['unique_objects']}")
    print(f"Total detections: {report['summary']['total_detections']}")
    print("Objects by class:")
    for cls, count in report['object_stats']['by_class'].items():
        print(f"  - {cls}: {count}")
    
    print("\nTimeline Events:")
    for i, event in enumerate(report['timeline'][:5]):
        print(f"  {i+1}. {event['event_type']} at {event['timestamp']:.2f}s")
    
    if len(report['timeline']) > 5:
        print(f"  ... and {len(report['timeline']) - 5} more events")
    
    print(f"\nOutput video: {results.get('output_video', 'Not saved')}")
    print(f"Report saved to: {output_dir / 'analysis_report.json'}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())