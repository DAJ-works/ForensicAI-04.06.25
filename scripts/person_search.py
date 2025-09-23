import os
import sys
import argparse
import cv2
import numpy as np
import json
import time
from pathlib import Path

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.models.person_reid import PersonReidentifier, PersonFeatureExtractor

def main():
    parser = argparse.ArgumentParser(description="Search for a person across videos")
    parser.add_argument("--query", required=True, help="Path to query image or person ID")
    parser.add_argument("--reid-model", help="Path to ReID database", default=None)
    parser.add_argument("--threshold", type=float, default=0.6, help="Similarity threshold")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top matches to return")
    parser.add_argument("--output-dir", help="Output directory", default=None)
    parser.add_argument("--video-path", help="Video to search in (optional)", default=None)
    
    args = parser.parse_args()
    
    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"./data/person_search_{timestamp}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Initialize re-identifier
    reidentifier = PersonReidentifier(similarity_threshold=args.threshold)
    
    # If a saved database is provided, load it
    if args.reid_model and os.path.exists(args.reid_model):
        print(f"Loading ReID database from: {args.reid_model}")
        reid_dir = Path(args.reid_model)
        reidentifier.load(str(reid_dir))
    else:
        print("Using a new ReID database (no saved model provided)")
    
    # Prepare query
    if args.query.isdigit():
        # Query by person ID
        person_id = int(args.query)
        print(f"Searching for person ID: {person_id}")
        
        # Check if person exists in database
        person = reidentifier.database.get_person(person_id)
        if not person:
            print(f"Error: Person ID {person_id} not found in database")
            return 1
        
        # Use person's features as query
        query_features = np.array(person["features"])
        query_image = None
        
        if "thumbnail" in person:
            query_image = person["thumbnail"]
            # Save query image
            query_img_path = output_dir / f"query_person_{person_id}.jpg"
            cv2.imwrite(str(query_img_path), query_image)
            print(f"Query image saved to: {query_img_path}")
    else:
        # Query by image
        query_path = args.query
        if not os.path.exists(query_path):
            print(f"Error: Query image not found: {query_path}")
            return 1
            
        print(f"Using query image: {query_path}")
        
        # Load image
        query_image = cv2.imread(query_path)
        if query_image is None:
            print(f"Error: Could not load query image: {query_path}")
            return 1
        
        # Extract features
        query_features = reidentifier.feature_extractor.extract_features(query_image)
    
    # If a video is provided, run analysis to find the person
    if args.video_path and os.path.exists(args.video_path):
        from backend.models.video_analyzer_reid import VideoAnalyzerWithReID
        from backend.models.object_detector import ObjectDetector
        
        print(f"Analyzing video to find person: {args.video_path}")
        
        # Initialize detector and analyzer
        detector = ObjectDetector(model_size='m', confidence_threshold=0.4)
        
        analyzer = VideoAnalyzerWithReID(
            detector=detector,
            output_dir=output_dir,
            reidentifier=reidentifier,
            enable_reid=True
        )
        
        # Run analysis
        results = analyzer.analyze_video(
            video_path=args.video_path,
            frame_interval=1,
            save_video=True,
            include_classes=["person"]
        )
        
        # Search for the person in the results
        print("\nSearching for matching persons in video...")
        
        # Extract person identities from results
        identities = []
        for frame_data in results["frames"].values():
            for det in frame_data["detections"]:
                if det.get("class_name") == "person" and "reid_features" in det:
                    # Calculate similarity with query
                    similarity = reidentifier.feature_extractor.compute_similarity(
                        query_features, det["reid_features"]
                    )
                    
                    if similarity >= args.threshold:
                        identities.append({
                            "person_id": det.get("person_id"),
                            "frame": frame_data["frame_number"],
                            "timestamp": frame_data["timestamp"],
                            "box": det["box"],
                            "similarity": similarity
                        })
        
        # Group by person_id
        persons = {}
        for identity in identities:
            person_id = identity["person_id"]
            if person_id not in persons:
                persons[person_id] = []
            persons[person_id].append(identity)
        
        # Calculate average similarity for each person
        for person_id, matches in persons.items():
            avg_similarity = sum(m["similarity"] for m in matches) / len(matches)
            persons[person_id] = {
                "matches": matches,
                "avg_similarity": avg_similarity,
                "first_seen": min(m["timestamp"] for m in matches),
                "last_seen": max(m["timestamp"] for m in matches),
                "count": len(matches)
            }
        
        # Sort by similarity
        sorted_persons = sorted(
            persons.items(), 
            key=lambda x: x[1]["avg_similarity"], 
            reverse=True
        )
        
        # Print top matches
        if sorted_persons:
            print(f"\nFound {len(sorted_persons)} matching persons:")
            for i, (person_id, data) in enumerate(sorted_persons[:args.top_k]):
                print(f"  {i+1}. Person #{person_id}: {data['avg_similarity']:.3f} similarity, " +
                      f"seen {data['count']} times from {data['first_seen']:.2f}s to {data['last_seen']:.2f}s")
                
                # Extract thumbnails for this person
                if person_id is not None:
                    person_dir = output_dir / f"person_{person_id}"
                    person_dir.mkdir(exist_ok=True)
                    
                    # Save top 5 best matches
                    top_matches = sorted(data["matches"], key=lambda x: x["similarity"], reverse=True)[:5]
                    
                    for j, match in enumerate(top_matches):
                        # Find the frame
                        frame_number = match["frame"]
                        timestamp = match["timestamp"]
                        
                        # Load video at this timestamp
                        cap = cv2.VideoCapture(args.video_path)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                        ret, frame = cap.read()
                        cap.release()
                        
                        if ret:
                            # Extract bounding box
                            box = match["box"]
                            x1, y1, x2, y2 = map(int, box)
                            
                            # Ensure coordinates are within image boundaries
                            x1 = max(0, x1)
                            y1 = max(0, y1)
                            x2 = min(frame.shape[1], x2)
                            y2 = min(frame.shape[0], y2)
                            
                            # Extract person crop with some margin
                            margin = 20
                            crop_y1 = max(0, y1 - margin)
                            crop_y2 = min(frame.shape[0], y2 + margin)
                            crop_x1 = max(0, x1 - margin)
                            crop_x2 = min(frame.shape[1], x2 + margin)
                            
                            crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                            
                            # Save crop
                            crop_path = person_dir / f"match_{j+1}_frame_{frame_number}_sim_{match['similarity']:.3f}.jpg"
                            cv2.imwrite(str(crop_path), crop)
                            
                            # Draw bounding box on frame
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Add label
                            label = f"Person #{person_id}: {match['similarity']:.3f}"
                            cv2.putText(
                                frame, 
                                label, 
                                (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7, 
                                (0, 255, 0), 
                                2
                            )
                            
                            # Save annotated frame
                            frame_path = person_dir / f"frame_{frame_number}_time_{timestamp:.2f}s.jpg"
                            cv2.imwrite(str(frame_path), frame)
                    
                    print(f"    Saved {min(5, len(data['matches']))} thumbnails to {person_dir}")
            
            # Create summary visualization
            if query_image is not None:
                create_search_results_visualization(
                    query_image, 
                    sorted_persons, 
                    output_dir,
                    top_k=min(args.top_k, 5)
                )
        else:
            print("No matching persons found in the video.")
    else:
        # Search in the existing database
        print(f"Searching for similar persons in the database...")
        matches = reidentifier.database.find_matches(query_features, top_k=args.top_k)
        
        if matches:
            print(f"\nFound {len(matches)} matching persons:")
            for i, match in enumerate(matches):
                person_id = match["id"]
                similarity = match["similarity"]
                metadata = match.get("metadata", {})
                
                appearances = metadata.get("appearances", "unknown")
                first_seen = metadata.get("first_seen_time", "unknown")
                last_seen = metadata.get("last_seen_time", "unknown")
                
                print(f"  {i+1}. Person #{person_id}: {similarity:.3f} similarity score")
                print(f"     Appearances: {appearances}")
                print(f"     First seen: {first_seen}")
                print(f"     Last seen: {last_seen}")
                
                # Extract thumbnail if available
                person_data = reidentifier.database.get_person(person_id)
                if person_data and "thumbnail" in person_data:
                    thumb_path = output_dir / f"match_{i+1}_person_{person_id}.jpg"
                    cv2.imwrite(str(thumb_path), person_data["thumbnail"])
                    print(f"     Thumbnail saved to: {thumb_path}")
                
                print()
            
            # Create visualization of results
            if query_image is not None:
                create_database_results_visualization(
                    query_image,
                    matches,
                    reidentifier.database,
                    str(output_dir / "search_results.jpg")
                )
        else:
            print("No matching persons found in the database.")
    
    print(f"\nSearch results saved to: {output_dir}")
    return 0

def create_search_results_visualization(query_image, results, output_dir, top_k=5):
    """Create a visualization of search results from video."""
    # Limit to top results
    results = results[:top_k]
    
    # Create figure
    n_results = len(results)
    fig_width = 12
    fig_height = 4 + 3 * n_results
    
    # Create visualization using OpenCV
    # Calculate size
    thumb_size = (192, 384)  # Width, height
    padding = 20
    
    # Calculate canvas size
    width = thumb_size[0] * (1 + 5)  # Query + 5 matches per person
    width += padding * (2 + 5)  # Padding
    height = (thumb_size[1] + padding * 3) * (n_results + 1)  # Per result + header
    
    # Create white canvas
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Add title
    cv2.putText(
        canvas,
        "Person Search Results",
        (padding, padding + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        2
    )
    
    # Add query image
    query_resized = cv2.resize(query_image, thumb_size)
    y_offset = padding * 2 + 30
    x_offset = padding
    canvas[y_offset:y_offset + thumb_size[1], x_offset:x_offset + thumb_size[0]] = query_resized
    
    # Add "Query" label
    cv2.putText(
        canvas,
        "Query",
        (padding, y_offset - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2
    )
    
    # For each result, add thumbnails
    for i, (person_id, data) in enumerate(results):
        # Calculate y position
        y_pos = y_offset + (thumb_size[1] + padding * 3) * (i + 1)
        
        # Add person label
        cv2.putText(
            canvas,
            f"Person #{person_id}: {data['avg_similarity']:.3f} similarity",
            (padding, y_pos - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )
        
        # Find thumbnails for this person
        person_dir = output_dir / f"person_{person_id}"
        if person_dir.exists():
            # Get thumbnails
            thumbnails = list(person_dir.glob("match_*.jpg"))
            thumbnails = sorted(thumbnails, key=lambda x: float(str(x).split('_sim_')[1].split('.jpg')[0]), reverse=True)
            
            # Add up to 5 thumbnails
            for j, thumb_path in enumerate(thumbnails[:5]):
                x_pos = padding + (thumb_size[0] + padding) * (j + 1)
                
                # Load and resize thumbnail
                thumb = cv2.imread(str(thumb_path))
                if thumb is not None:
                    thumb_resized = cv2.resize(thumb, thumb_size)
                    canvas[y_pos:y_pos + thumb_size[1], x_pos:x_pos + thumb_size[0]] = thumb_resized
                    
                    # Add similarity from filename
                    similarity = float(str(thumb_path).split('_sim_')[1].split('.jpg')[0])
                    cv2.putText(
                        canvas,
                        f"{similarity:.3f}",
                        (x_pos + 5, y_pos + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1
                    )
    
    # Save visualization
    output_path = output_dir / "search_results.jpg"
    cv2.imwrite(str(output_path), canvas)
    print(f"Search results visualization saved to: {output_path}")

def create_database_results_visualization(query_image, matches, database, output_path):
    """Create a visualization of search results from database."""
    # Resize query image
    thumb_size = (192, 384)  # Width, height
    query_resized = cv2.resize(query_image, thumb_size)
    
    # Limit to top matches
    n_results = min(5, len(matches))
    
    # Calculate canvas size
    padding = 20
    width = thumb_size[0] * (1 + n_results)  # Query + matches
    width += padding * (2 + n_results)  # Padding
    height = thumb_size[1] + padding * 3
    
    # Create white canvas
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Add query image
    y_offset = padding
    x_offset = padding
    canvas[y_offset:y_offset + thumb_size[1], x_offset:x_offset + thumb_size[0]] = query_resized
    
    # Add "Query" label
    cv2.putText(
        canvas,
        "Query",
        (padding, y_offset - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2
    )
    
    # Add each match
    for i, match in enumerate(matches[:n_results]):
        person_id = match["id"]
        similarity = match["similarity"]
        
        # Get person thumbnail
        person_data = database.get_person(person_id)
        if person_data and "thumbnail" in person_data:
            thumbnail = person_data["thumbnail"]
            thumbnail_resized = cv2.resize(thumbnail, thumb_size)
            
            # Calculate position
            x_pos = padding + (thumb_size[0] + padding) * (i + 1)
            
            # Add thumbnail
            canvas[y_offset:y_offset + thumb_size[1], x_pos:x_pos + thumb_size[0]] = thumbnail_resized
            
            # Add label
            cv2.putText(
                canvas,
                f"#{person_id}: {similarity:.3f}",
                (x_pos, y_offset - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2
            )
    
    # Save visualization
    cv2.imwrite(output_path, canvas)
    print(f"Database search visualization saved to: {output_path}")

if __name__ == "__main__":
    sys.exit(main())