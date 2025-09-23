import os
import sys
import argparse
import torch
import time
import json
from pathlib import Path
import logging

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.models.object_detector import ObjectDetector
from backend.models.enhanced_reid import EnhancedPersonReidentifier
from backend.models.video_analyzer_reid import VideoAnalyzerWithReID

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Video analysis with enhanced person re-identification")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("--output-dir", help="Output directory", default=None)
    parser.add_argument("--model-type", choices=['osnet', 'mgn', 'resnet50'], default='osnet', 
                        help="ReID model architecture")
    parser.add_argument("--model-path", help="Path to fine-tuned ReID model", default=None)
    parser.add_argument("--detector-model", help="YOLO model size (n, s, m, l, x)", default="m")
    parser.add_argument("--use-attributes", action="store_true", help="Enable attribute-based reidentification")
    parser.add_argument("--attribute-model", help="Path to attribute model", default=None)
    parser.add_argument("--frame-interval", type=int, default=1, help="Process every nth frame")
    parser.add_argument("--similarity", type=float, default=0.6, help="Similarity threshold for ReID")
    parser.add_argument("--save-crops", action="store_true", help="Save person crops")
    parser.add_argument("--query-image", help="Path to query image for person search", default=None)
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID (-1 for CPU)")
    
    args = parser.parse_args()
    
    # Configure device
    if args.gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
        use_gpu = True
        logger.info(f"Using GPU: {torch.cuda.get_device_name(args.gpu_id)}")
    else:
        device = torch.device("cpu")
        use_gpu = False
        logger.info("Using CPU")
    
    # Check if video exists
    if not os.path.exists(args.video_path):
        logger.error(f"Video file not found: {args.video_path}")
        return 1
    
    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Create a directory with the video name and timestamp
        video_name = Path(args.video_path).stem
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"./data/enhanced_reid/{video_name}_{timestamp}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Initialize detector
    logger.info(f"Initializing YOLOv8{args.detector_model} detector")
    detector = ObjectDetector(
        model_size=args.detector_model,
        confidence_threshold=0.4,
        classes=[0]  # Class 0 is 'person' in COCO dataset
    )
    
    # Initialize enhanced reidentifier
    logger.info(f"Initializing {args.model_type} re-identification model")
    reidentifier = EnhancedPersonReidentifier(
        model_type=args.model_type,
        model_path=args.model_path,
        similarity_threshold=args.similarity,
        use_attributes=args.use_attributes,
        attribute_model_path=args.attribute_model,
        use_gpu=use_gpu
    )
    
    # Initialize video analyzer with ReID
    analyzer = VideoAnalyzerWithReID(
        detector=detector,
        output_dir=output_dir,
        reidentifier=reidentifier,
        enable_reid=True
    )
    
    # Run analysis
    logger.info(f"Analyzing video: {args.video_path}")
    results = analyzer.analyze_video(
        video_path=args.video_path,
        frame_interval=args.frame_interval,
        save_video=True,
        save_frames=args.save_crops,
        include_classes=["person"],  # Focus only on people
        visualize_reid=True
    )
    
    # Generate report
    logger.info("Generating analysis report...")
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
        
        # Save person thumbnails
        logger.info("Saving person gallery...")
        person_gallery_dir = output_dir / "person_gallery"
        person_gallery_dir.mkdir(exist_ok=True)
        
        # Create gallery visualization
        reidentifier.database.visualize_database(str(output_dir / "person_gallery.jpg"))
    
    # Query by image if provided
    if args.query_image and os.path.exists(args.query_image):
        logger.info(f"Searching for person in query image: {args.query_image}")
        
        # Perform search
        query_results = reidentifier.query_by_appearance(
            args.query_image,
            top_k=5,
            use_attributes=args.use_attributes
        )
        
        if query_results:
            print("\nMatching Persons:")
            for i, match in enumerate(query_results):
                person_id = match['id']
                similarity = match['similarity']
                
                # Get additional details if available
                metadata = match.get('metadata', {})
                appearances = metadata.get('appearances', "unknown")
                first_seen = metadata.get('first_seen_time', "unknown")
                last_seen = metadata.get('last_seen_time', "unknown")
                
                print(f"  {i+1}. Person #{person_id}: {similarity:.3f} similarity")
                print(f"     Appearances: {appearances}")
                print(f"     First seen: {first_seen}")
                print(f"     Last seen: {last_seen}")
                
                # Save match visualization
                if 'attribute_similarity' in match:
                    print(f"     Feature similarity: {match['feature_similarity']:.3f}")
                    print(f"     Attribute similarity: {match['attribute_similarity']:.3f}")
            
            # Create search results visualization
            query_img_path = output_dir / "query_image.jpg"
            import shutil
            shutil.copy(args.query_image, query_img_path)
            
            # TODO: Add visualization of query results
        else:
            print("No matching persons found.")
    
    # Print final paths
    print(f"\nOutput video: {results.get('output_video', 'Not saved')}")
    print(f"Report saved to: {output_dir / 'analysis_report.json'}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())