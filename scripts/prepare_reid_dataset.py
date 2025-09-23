import os
import sys
import argparse
import shutil
import random
import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import logging

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.preprocessing.video_processor import VideoProcessor
from backend.models.object_detector import ObjectDetector
from backend.models.person_reid import PersonReidentifier, PersonFeatureExtractor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_persons_from_video(
    video_path: str,
    output_dir: Path,
    detector: ObjectDetector,
    frame_interval: int = 10,
    min_confidence: float = 0.5,
    min_height: int = 100,
    max_persons: int = None
):
    """
    Extract person images from a video for fine-tuning.
    
    Parameters:
    -----------
    video_path : str
        Path to the video file
    output_dir : Path
        Directory to save extracted person images
    detector : ObjectDetector
        Object detector
    frame_interval : int
        Process every nth frame
    min_confidence : float
        Minimum confidence for person detection
    min_height : int
        Minimum height for person detection (pixels)
    max_persons : int, optional
        Maximum persons to extract (per identity)
    
    Returns:
    --------
    int
        Number of persons extracted
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return 0
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    logger.info(f"Processing video: {video_path}")
    logger.info(f"Total frames: {frame_count}, FPS: {fps:.2f}")
    
    # Process frames
    frame_idx = 0
    processed_frames = 0
    extracted_count = 0
    
    # Dictionary to track persons by ID
    persons_by_id = {}
    
    # Progress bar
    pbar = tqdm(total=frame_count, desc="Extracting persons")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Update progress
        pbar.update(1)
        
        # Process at specified interval
        if frame_idx % frame_interval == 0:
            # Run detection
            detections, _ = detector.detect(frame)
            
            # Filter person detections
            person_detections = [d for d in detections 
                                if d.get('class_name') == 'person' 
                                and d.get('confidence', 0) >= min_confidence]
            
            # Process each person
            for i, det in enumerate(person_detections):
                box = det['box']
                x1, y1, x2, y2 = map(int, box)
                
                # Skip small detections
                if y2 - y1 < min_height:
                    continue
                
                # Extract person crop
                person_img = frame[y1:y2, x1:x2]
                
                # Get object ID
                person_id = det.get('object_id')
                if person_id is None:
                    person_id = i  # Use detection index if no object ID
                
                # Skip if we've reached the maximum for this person
                if max_persons and person_id in persons_by_id and len(persons_by_id[person_id]) >= max_persons:
                    continue
                
                # Save the image
                if person_id not in persons_by_id:
                    persons_by_id[person_id] = []
                
                # Add some random suffix for uniqueness
                img_name = f"person_{person_id}_{processed_frames}_{random.randint(1000, 9999)}.jpg"
                img_path = output_dir / img_name
                
                cv2.imwrite(str(img_path), person_img)
                
                # Add to tracking
                persons_by_id[person_id].append(str(img_path))
                extracted_count += 1
            
            processed_frames += 1
        
        frame_idx += 1
    
    # Close video and progress bar
    cap.release()
    pbar.close()
    
    logger.info(f"Extracted {extracted_count} person images from {len(persons_by_id)} identities")
    
    return extracted_count

def organize_dataset_for_training(
    source_dir: Path,
    output_dir: Path,
    test_split: float = 0.2,
    min_images_per_id: int = 4
):
    """
    Organize extracted person images into a training dataset structure.
    Creates train and test splits with one folder per identity.
    
    Parameters:
    -----------
    source_dir : Path
        Directory with extracted person images
    output_dir : Path
        Directory to save organized dataset
    test_split : float
        Fraction of data to use for testing
    min_images_per_id : int
        Minimum images required per identity
    
    Returns:
    --------
    Tuple[int, int]
        Number of identities and images in the dataset
    """
    # Create output directories
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"
    
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all person images
    logger.info(f"Organizing images from {source_dir}")
    
    # Group by person ID
    persons = defaultdict(list)
    
    for img_path in source_dir.glob("person_*.jpg"):
        # Extract person ID from filename
        person_id = img_path.stem.split('_')[1]
        persons[person_id].append(img_path)
    
    # Filter out persons with too few images
    filtered_persons = {pid: imgs for pid, imgs in persons.items() if len(imgs) >= min_images_per_id}
    
    logger.info(f"Found {len(filtered_persons)} persons with at least {min_images_per_id} images")
    
    # Assign IDs incrementally
    total_imgs = 0
    
    for i, (pid, img_paths) in enumerate(filtered_persons.items(), 1):
        # Create directories
        train_person_dir = train_dir / f"{i:04d}"
        test_person_dir = test_dir / f"{i:04d}"
        
        train_person_dir.mkdir(exist_ok=True)
        test_person_dir.mkdir(exist_ok=True)
        
        # Split images into train and test
        random.shuffle(img_paths)
        split_idx = max(1, int(len(img_paths) * (1 - test_split)))
        
        train_imgs = img_paths[:split_idx]
        test_imgs = img_paths[split_idx:]
        
        # Copy images
        for j, img_path in enumerate(train_imgs):
            dst_path = train_person_dir / f"{j:04d}.jpg"
            shutil.copy(img_path, dst_path)
        
        for j, img_path in enumerate(test_imgs):
            dst_path = test_person_dir / f"{j:04d}.jpg"
            shutil.copy(img_path, dst_path)
        
        total_imgs += len(img_paths)
    
    logger.info(f"Organized {total_imgs} images into {len(filtered_persons)} identities")
    logger.info(f"Train set: {train_dir}, Test set: {test_dir}")
    
    return len(filtered_persons), total_imgs

def generate_attributes(
    dataset_dir: Path,
    output_path: Path,
    reidentifier: PersonReidentifier
):
    """
    Generate attribute annotations for training attribute-based re-identification.
    Uses a pre-trained model to generate pseudo-labels.
    
    Parameters:
    -----------
    dataset_dir : Path
        Directory with organized dataset
    output_path : Path
        Path to save attribute annotations
    reidentifier : PersonReidentifier
        Reidentifier with feature extractor
    
    Returns:
    --------
    Dict
        Attribute annotations
    """
    logger.info(f"Generating attribute annotations for {dataset_dir}")
    
    # Get feature extractor from reidentifier
    feature_extractor = reidentifier.feature_extractor
    
    # Dictionary to store attributes
    attributes = {}
    
    # Find all images
    image_paths = []
    for split in ['train', 'test']:
        split_dir = dataset_dir / split
        if split_dir.exists():
            for img_path in split_dir.glob("**/*.jpg"):
                image_paths.append(img_path)
    
    logger.info(f"Found {len(image_paths)} images for attribute generation")
    
    # Process images in batches
    batch_size = 32
    num_batches = (len(image_paths) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Generating attributes"):
        batch_paths = image_paths[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        
        # Load images
        batch_imgs = []
        for img_path in batch_paths:
            img = cv2.imread(str(img_path))
            if img is not None:
                batch_imgs.append(img)
            else:
                logger.warning(f"Could not read image: {img_path}")
        
        # Skip empty batches
        if not batch_imgs:
            continue
        
        # Extract features
        features = feature_extractor.batch_extract_features(batch_imgs)
        
        # Generate pseudo-attributes (for demonstration)
        for i, (img_path, feat) in enumerate(zip(batch_paths, features)):
            # Generate some pseudo-attributes based on feature values
            # In a real implementation, you'd use a trained attribute classifier here
            # For this example, we'll create synthetic attributes
            
            # Use first few feature dimensions to create attribute values
            img_name = img_path.name
            
            # Create pseudo-attributes
            attributes[img_name] = {
                'gender': int(feat[0] > 0),  # Binary: 0 or 1
                'age': min(3, max(0, int((feat[1] + 1) * 2))),  # 0-3
                'upper_color': min(9, max(0, int((feat[2] + 1) * 5))),  # 0-9
                'lower_color': min(9, max(0, int((feat[3] + 1) * 5))),  # 0-9
                'has_bag': int(feat[4] > 0.2),  # Binary
                'has_hat': int(feat[5] > 0.3)   # Binary
            }
    
    # Save attributes
    with open(output_path, 'w') as f:
        json.dump(attributes, f, indent=2)
    
    logger.info(f"Generated attributes for {len(attributes)} images, saved to {output_path}")
    
    return attributes

def main():
    parser = argparse.ArgumentParser(description="Prepare ReID dataset for fine-tuning")
    parser.add_argument("--videos", type=str, nargs="+", help="Input video files or directories")
    parser.add_argument("--output-dir", type=str, default="./data/reid_training", help="Output directory")
    parser.add_argument("--frame-interval", type=int, default=30, help="Process every nth frame")
    parser.add_argument("--min-confidence", type=float, default=0.6, help="Minimum detection confidence")
    parser.add_argument("--model-size", type=str, default="m", help="YOLOv8 model size (n, s, m, l, x)")
    parser.add_argument("--test-split", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--min-images", type=int, default=6, help="Minimum images per identity")
    parser.add_argument("--generate-attributes", action="store_true", help="Generate attribute annotations")
    
    args = parser.parse_args()
    
    # Ensure we have input videos
    if not args.videos:
        logger.error("No input videos provided")
        parser.print_help()
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize detector
    logger.info(f"Initializing YOLOv8{args.model_size} detector")
    detector = ObjectDetector(
        model_size=args.model_size,
        confidence_threshold=args.min_confidence,
        classes=[0]  # Class 0 is 'person' in COCO dataset
    )
    
    # Create extraction directory
    extraction_dir = output_dir / "extracted"
    extraction_dir.mkdir(exist_ok=True)
    
    # Process input videos
    video_paths = []
    for path in args.videos:
        if os.path.isdir(path):
            for ext in ['.mp4', '.avi', '.mov', '.mkv']:
                video_paths.extend(list(Path(path).glob(f"*{ext}")))
        else:
            video_paths.append(Path(path))
    
    # Extract person images
    total_extracted = 0
    for video_path in video_paths:
        if not video_path.exists():
            logger.warning(f"Video not found: {video_path}")
            continue
            
        extracted = extract_persons_from_video(
            str(video_path),
            extraction_dir,
            detector,
            frame_interval=args.frame_interval,
            min_confidence=args.min_confidence
        )
        total_extracted += extracted
    
    logger.info(f"Total extracted: {total_extracted} person images")
    
    # Organize dataset
    dataset_dir = output_dir / "dataset"
    num_ids, num_imgs = organize_dataset_for_training(
        extraction_dir,
        dataset_dir,
        test_split=args.test_split,
        min_images_per_id=args.min_images
    )
    
    # Generate attribute annotations if requested
    if args.generate_attributes:
        logger.info("Generating attribute annotations")
        
        # Initialize reidentifier
        reidentifier = PersonReidentifier()
        
        # Generate attributes
        attribute_path = output_dir / "attributes.json"
        generate_attributes(dataset_dir, attribute_path, reidentifier)
    
    # Print summary
    logger.info("\nDataset Summary:")
    logger.info(f"- Videos processed: {len(video_paths)}")
    logger.info(f"- Person images extracted: {total_extracted}")
    logger.info(f"- Identities in dataset: {num_ids}")
    logger.info(f"- Images in final dataset: {num_imgs}")
    logger.info(f"- Dataset directory: {dataset_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())