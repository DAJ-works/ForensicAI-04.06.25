import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path to import our module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.preprocessing.video_processor import VideoProcessor

def main():
    parser = argparse.ArgumentParser(description="Test video preprocessing functions")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("--output-dir", help="Output directory", default=None)
    parser.add_argument("--case-id", help="Case ID", default=None)
    parser.add_argument("--extract-frames", action="store_true", help="Extract frames")
    parser.add_argument("--frame-interval", type=int, default=30, help="Frame extraction interval")
    parser.add_argument("--denoise", action="store_true", help="Denoise video")
    parser.add_argument("--stabilize", action="store_true", help="Stabilize video")
    parser.add_argument("--enhance", action="store_true", help="Enhance low-light video")
    parser.add_argument("--analyze", action="store_true", help="Analyze video frames")
    parser.add_argument("--all", action="store_true", help="Run all preprocessing steps")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return 1
    
    print(f"Processing video: {args.video_path}")
    
    # Initialize video processor
    processor = VideoProcessor(
        input_path=args.video_path,
        output_dir=args.output_dir,
        case_id=args.case_id
    )
    
    print(f"Video metadata:")
    print(f"  - Resolution: {processor.width}x{processor.height}")
    print(f"  - Duration: {processor.duration:.2f} seconds")
    print(f"  - Frame rate: {processor.fps:.2f} fps")
    print(f"  - Frame count: {processor.frame_count}")
    print(f"  - Codec: {processor.fourcc}")
    print(f"  - Output directory: {processor.output_dir}")
    
    try:
        if args.extract_frames or args.all:
            print(f"\nExtracting frames (interval={args.frame_interval})...")
            frame_paths = processor.extract_frames(interval=args.frame_interval)
            print(f"Extracted {len(frame_paths)} frames")

        if args.denoise or args.all:
            print(f"\nDenoising video...")
            denoised_path = processor.denoise_video()
            print(f"Denoised video saved to: {denoised_path}")

        if args.stabilize or args.all:
            print(f"\nStabilizing video...")
            stabilized_path = processor.stabilize_video()
            print(f"Stabilized video saved to: {stabilized_path}")

        if args.enhance or args.all:
            print(f"\nEnhancing low-light video...")
            enhanced_path = processor.enhance_low_light()
            print(f"Enhanced video saved to: {enhanced_path}")

        if args.analyze or args.all:
            print(f"\nAnalyzing video frames...")
            analysis = processor.analyze_frames(interval=args.frame_interval)
            print(f"Found {len(analysis['potential_issues'])} frames with potential issues")
            for i, issue in enumerate(analysis["potential_issues"][:5]):  # Show first 5 issues
                print(f"  - Frame {issue['frame']}: {', '.join(issue['issues'])}")
            if len(analysis["potential_issues"]) > 5:
                print(f"  - ... and {len(analysis['potential_issues']) - 5} more issues")

        summary = processor.create_summary()
        print(f"\nProcessing summary saved to: {processor.output_dir / 'processing_summary.json'}")
        
    finally:
        processor.close()
    
    print("\nPreprocessing completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())