import os
import sys
import argparse
import cv2
import numpy as np
import time
import logging
from pathlib import Path
import matplotlib.pyplot as plt

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.reconstruction.scene_reconstruction import SceneReconstructor, PersonLocalization3D
from backend.reconstruction.integrate_with_analysis import ForensicSceneAnalyzer
from backend.models.video_analyzer_reid import VideoAnalyzerWithReID
from backend.models.object_detector import ObjectDetector
from backend.models.object_tracker import ObjectTracker
from backend.models.person_reid import PersonReidentifier

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('scene_reconstruction.log')
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Test 3D scene reconstruction")
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument("--calibration-pattern", help="Path to directory with calibration images")
    parser.add_argument("--camera-params", help="Path to camera parameters file")
    parser.add_argument("--frame-interval", type=int, default=5, help="Process every nth frame")
    parser.add_argument("--depth-method", choices=['monocular', 'stereo'], default='monocular', 
                        help="Depth estimation method")
    parser.add_argument("--calibration-method", choices=['auto', 'chessboard'], default='auto',
                        help="Camera calibration method")
    parser.add_argument("--skip-detection", action="store_true", 
                        help="Skip object detection/tracking and focus on reconstruction")
    parser.add_argument("--model-size", help="YOLOv8 model size (n, s, m, l, x)", default="m")
    parser.add_argument("--confidence", type=float, default=0.4, help="Detection confidence threshold")
    parser.add_argument("--reid-model", help="Path to person re-identification model")
    
    args = parser.parse_args()
    
    # Check if video file exists
    if not os.path.exists(args.video_path):
        logger.error(f"Video file not found: {args.video_path}")
        return 1
    
    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Create default directory
        video_name = Path(args.video_path).stem
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"./data/scene_reconstruction/{video_name}_{timestamp}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Get calibration images if provided
    calibration_images = []
    if args.calibration_pattern and os.path.exists(args.calibration_pattern):
        pattern_path = Path(args.calibration_pattern)
        if pattern_path.is_dir():
            # Get all images in directory
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                calibration_images.extend(list(pattern_path.glob(ext)))
        elif pattern_path.is_file():
            # Single image
            calibration_images = [pattern_path]
            
        logger.info(f"Using {len(calibration_images)} calibration images")
    
    if args.skip_detection:
        # Only perform scene reconstruction
        logger.info("Performing scene reconstruction only (skipping detection/tracking)")
        
        # Initialize scene reconstructor
        reconstructor = SceneReconstructor(
            output_dir=str(output_dir),
            use_depth_estimation=True,
            depth_method=args.depth_method,
            calibration_method=args.calibration_method
        )
        
        # Load camera parameters if provided
        if args.camera_params and os.path.exists(args.camera_params):
            logger.info(f"Loading camera parameters from: {args.camera_params}")
            reconstructor.load_camera_parameters(args.camera_params)
        elif calibration_images:
            logger.info("Calibrating camera from calibration images")
            reconstructor.calibrate_camera(image_paths=[str(img) for img in calibration_images])
        
        # Reconstruct scene
        logger.info(f"Reconstructing scene from: {args.video_path}")
        start_time = time.time()
        
        success = reconstructor.reconstruct_scene(
            video_path=args.video_path,
            frame_interval=args.frame_interval,
            visualize=True
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Scene reconstruction completed in {elapsed_time:.2f} seconds")
        
        if success:
            logger.info("Scene reconstruction successful")
            
            # Save reconstruction
            reconstructor.save_reconstruction(str(output_dir))
            
            # Visualize camera trajectory
            reconstructor._visualize_camera_trajectory(str(output_dir / "camera_trajectory.png"))
            
            # Visualize scene
            reconstructor._visualize_scene()
            
            return 0
        else:
            logger.error("Scene reconstruction failed")
            return 1
    else:
        # Perform full integrated analysis
        logger.info("Performing integrated video analysis and scene reconstruction")
        
        # Initialize object detector
        detector = ObjectDetector(
            model_size=args.model_size,
            confidence_threshold=args.confidence
        )
        
        # Initialize object tracker
        tracker = ObjectTracker()
        
        # Initialize person re-identifier
        reidentifier = PersonReidentifier()
        
        # Create video analyzer
        video_analyzer = VideoAnalyzerWithReID(
            detector=detector,
            tracker=tracker,
            reidentifier=reidentifier,
            output_dir=str(output_dir / "video_analysis"),
            enable_reid=True
        )
        
        # Initialize forensic scene analyzer
        scene_analyzer = ForensicSceneAnalyzer(
            output_dir=str(output_dir),
            use_depth_estimation=True,
            depth_method=args.depth_method,
            calibration_method=args.calibration_method
        )
        
        # Run integrated analysis
        logger.info(f"Analyzing video: {args.video_path}")
        start_time = time.time()
        
        results = scene_analyzer.analyze_video(
            video_path=args.video_path,
            video_analyzer=video_analyzer,
            frame_interval=args.frame_interval,
            calibration_images=[str(img) for img in calibration_images] if calibration_images else None,
            camera_params_file=args.camera_params,
            visualize=True
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Integrated analysis completed in {elapsed_time:.2f} seconds")
        
        # Print summary
        print("\nIntegrated Analysis Summary:")
        print(f"Video: {args.video_path}")
        print(f"Total frames: {results.get('video_metadata', {}).get('frame_count', 0)}")
        print(f"Total detections: {results.get('2d_analysis', {}).get('total_detections', 0)}")
        print(f"Total unique objects: {results.get('2d_analysis', {}).get('total_unique_objects', 0)}")
        print(f"Total unique persons: {len(results.get('2d_analysis', {}).get('person_identities', []))}")
        
        # Print 3D reconstruction summary
        print("\n3D Reconstruction Summary:")
        if results.get('3d_analysis', {}).get('reconstruction_available', False):
            print("Camera calibration: Success")
            print(f"Number of camera poses: {results.get('3d_analysis', {}).get('camera_poses', 0)}")
            print(f"Point cloud points: {results.get('3d_analysis', {}).get('scene_point_cloud', {}).get('num_points', 0)}")
            print(f"Persons localized in 3D: {len(results.get('3d_analysis', {}).get('person_positions', {}))}")
        else:
            print("Camera calibration: Failed")
        
        # Print output paths
        print(f"\nOutput directory: {output_dir}")
        print(f"Video analysis results: {output_dir / 'video_analysis'}")
        print(f"Scene reconstruction: {output_dir / 'reconstruction'}")
        print(f"Person localization: {output_dir / 'person_localization'}")
        print(f"Integrated results: {output_dir / 'integrated_analysis_results.json'}")
        
        return 0

if __name__ == "__main__":
    sys.exit(main())