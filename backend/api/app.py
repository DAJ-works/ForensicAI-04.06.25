import os
import sys
import json
import time
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from queue import Queue
from threading import Lock

from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS

from huggingface_hub import InferenceClient
import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")
print(f"Current directory: {os.getcwd()}")

from backend.models.video_analyzer_reid import VideoAnalyzerWithReID
from backend.models.object_detector import ObjectDetector
from backend.models.object_tracker import ObjectTracker
from backend.models.person_reid import PersonReidentifier
from backend.models.enhanced_filter import EnhancedFilter
from backend.models.two_stage_detector import TwoStageDetector
from backend.models.weapon_detector import WeaponDetector

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('analysis_api.log')
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='../../frontend/build')
CORS(app)  # Enable CORS for all routes

DATA_DIR = Path("./data")
UPLOAD_FOLDER = DATA_DIR / "uploads"
ANALYSIS_FOLDER = DATA_DIR / "analysis"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
ANALYSIS_FOLDER.mkdir(parents=True, exist_ok=True)

logger.info(f"Data directory: {DATA_DIR.absolute()}")
logger.info(f"Upload folder: {UPLOAD_FOLDER.absolute()}")
logger.info(f"Analysis folder: {ANALYSIS_FOLDER.absolute()}")

active_jobs = {}
completed_jobs = {}

analysis_clients = {}
analysis_clients_lock = Lock()

detector = ObjectDetector(model_size='m', confidence_threshold=0.15)  # Lower threshold for better detection
tracker = ObjectTracker(max_age=60, min_hits=1, iou_threshold=0.2)  # More forgiving tracker
reidentifier = PersonReidentifier()

HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "")  # Get from environment variable
MODEL_ID = "google/flan-t5-base"  # Using a more capable model within free tier

hf_client = None
if HF_API_TOKEN:
    try:
        hf_client = InferenceClient(token=HF_API_TOKEN)
        logger.info(f"Initialized Hugging Face client with model {MODEL_ID}")
    except Exception as e:
        logger.error(f"Failed to initialize Hugging Face client: {e}")
else:
    logger.warning("No HF_API_TOKEN provided. Chat feature will run in demo mode.")

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

def get_case_by_name(case_name):
    for case_dir in ANALYSIS_FOLDER.iterdir():
        if case_dir.is_dir():
            result_file = case_dir / "analysis_results.json"
            if result_file.exists():
                try:
                    with open(result_file, 'r') as f:
                        results = json.load(f)
                    if results.get('case_name', '').lower() == case_name.lower():
                        return case_dir.name, results
                except Exception as e:
                    logger.error(f"Error reading case file {result_file}: {e}")
    return None, None

@app.route('/api/cases', methods=['GET'])
def get_cases():
    cases = []
    for case_dir in ANALYSIS_FOLDER.iterdir():
        if case_dir.is_dir():
            # Look for analysis results
            result_file = case_dir / "analysis_results.json"
            if result_file.exists():
                try:
                    with open(result_file, 'r') as f:
                        results = json.load(f)
                    
                    has_video = (case_dir / "output_video.mp4").exists()
                    
                    case_info = {
                        'case_id': case_dir.name,
                        'case_name': results.get('case_name', case_dir.name),  # Use case_name if available, fall back to ID
                        'video_path': results.get('video_path', 'Unknown'),
                        'timestamp': results.get('timestamp', ''),
                        'total_detections': results.get('total_detections', 0),
                        'total_unique_objects': results.get('total_unique_objects', 0),
                        'num_persons': len(results.get('person_identities', [])),
                        'thumbnail': f"/api/cases/{case_dir.name}/thumbnail",
                        'has_video': has_video,
                        'status': 'completed'
                    }
                    
                    if 'debug_info' in results and 'false_positive_reduction' in results.get('debug_info', {}):
                        fp_data = results['debug_info']['false_positive_reduction']
                        case_info['filtered_detections'] = fp_data.get('filtered_detections', 0)
                        case_info['reduction_rate'] = fp_data.get('reduction_rate', 0)
                    
                    cases.append(case_info)
                except Exception as e:
                    logger.error(f"Error loading case {case_dir.name}: {e}")
    
    for job_id, job in active_jobs.items():
        cases.append({
            'case_id': job_id,
            'case_name': job.get('case_name', job_id),  # Use case_name if available
            'video_path': job.get('video_path', 'Unknown'),
            'timestamp': job.get('start_time', ''),
            'status': 'processing',
            'progress': job.get('progress', 0)
        })
    
    return jsonify(cases)

@app.route('/api/cases/name/<path:case_name>', methods=['GET'])
def get_case_by_name_endpoint(case_name):
    """Get a case by its name."""
    case_id, case_data = get_case_by_name(case_name)
    
    if not case_id:
        return jsonify({'error': 'Case not found'}), 404
    
    case_dir = ANALYSIS_FOLDER / case_id
    case_data['visualizations'] = {}
    
    viz_files = {
        'trajectories': 'trajectories.jpg',
        'heatmap': 'heatmap.jpg',
        'reid_gallery': 'person_gallery.jpg',
        'timeline': 'event_timeline.png',
        'person_map': 'person_position_map.png'
    }
    
    for viz_key, viz_file in viz_files.items():
        viz_path = case_dir / viz_file
        if viz_path.exists():
            case_data['visualizations'][viz_key] = f"/api/cases/{case_id}/visualizations/{viz_file}"
    
    output_video = case_dir / "output_video.mp4"
    if output_video.exists():
        case_data['output_video'] = f"/api/cases/{case_id}/output_video"
    else:
        # Check for other MP4 files as fallback
        for video_file in case_dir.glob("*.mp4"):
            case_data['output_video'] = f"/api/cases/{case_id}/visualizations/{video_file.name}"
            break
    
    return jsonify(case_data)

@app.route('/api/cases/<case_id>', methods=['GET'])
def get_case(case_id):
    """Get details for a specific case."""
    # Check if it's an active job
    if case_id in active_jobs:
        return jsonify({
            'case_id': case_id,
            'case_name': active_jobs[case_id].get('case_name', case_id),  # Include case_name
            'status': 'processing',
            'progress': active_jobs[case_id].get('progress', 0),
            'video_path': active_jobs[case_id].get('video_path', ''),
            'timestamp': active_jobs[case_id].get('start_time', '')
        })
    
    case_dir = ANALYSIS_FOLDER / case_id
    result_file = case_dir / "analysis_results.json"
    
    if not result_file.exists():
        return jsonify({'error': 'Case not found'}), 404
    
    try:
        with open(result_file, 'r') as f:
            results = json.load(f)
        
        if 'debug_info' in results and 'false_positive_reduction' in results.get('debug_info', {}):
            fp_reduction = results['debug_info']['false_positive_reduction']
            results['filtering_stats'] = {
                'total_detections': fp_reduction.get('total_detections', 0),
                'valid_detections': fp_reduction.get('valid_detections', 0),
                'filtered_detections': fp_reduction.get('filtered_detections', 0),
                'reduction_rate': fp_reduction.get('reduction_rate', 0)
            }
        
        results['visualizations'] = {}
        
        output_video = case_dir / "output_video.mp4"
        if output_video.exists():
            results['output_video'] = f"/api/cases/{case_id}/output_video"
            logger.info(f"Found output video for case {case_id}: {output_video}")
        else:
            logger.warning(f"Output video not found for case {case_id}")
            # Check for other MP4 files as fallback
            for video_file in case_dir.glob("*.mp4"):
                results['output_video'] = f"/api/cases/{case_id}/visualizations/{video_file.name}"
                logger.info(f"Using alternate video for case {case_id}: {video_file}")
                break
        
        results['debug_info'] = results.get('debug_info', {})
        results['debug_info'].update({
            'has_persons': len(results.get('person_identities', [])) > 0,
            'has_video': output_video.exists(),
            'case_dir_exists': case_dir.exists(),
            'case_dir_contents': [f.name for f in case_dir.iterdir()] if case_dir.exists() else []
        })
        
        return jsonify(results)
    
    except Exception as e:
        logger.error(f"Error loading case {case_id}: {e}")
        return jsonify({'error': f'Error loading case: {str(e)}'}), 500

@app.route('/api/cases/<case_id>/thumbnail', methods=['GET'])
def get_thumbnail(case_id):
    """Get thumbnail for a case."""
    case_dir = ANALYSIS_FOLDER / case_id
    thumbnail_path = case_dir / "thumbnail.jpg"
    
    if thumbnail_path.exists():
        return send_from_directory(case_dir, "thumbnail.jpg")
    else:
        for ext in ['jpg', 'png']:
            for img_file in case_dir.glob(f"*.{ext}"):
                return send_from_directory(case_dir, img_file.name)
        
        return jsonify({'error': 'Thumbnail not found'}), 404

@app.route('/api/cases/<case_id>/visualizations/<path:filename>', methods=['GET'])
def get_visualization(case_id, filename):
    case_dir = ANALYSIS_FOLDER / case_id
    if not (case_dir / filename).exists():
        logger.warning(f"Visualization file not found: {case_dir / filename}")
        return jsonify({'error': 'File not found'}), 404
    return send_from_directory(case_dir, filename)

@app.route('/api/cases/<case_id>/output_video', methods=['GET'])
def get_output_video(case_id):
    """Get the output video for a case with proper download headers."""
    user = request.args.get('user', 'unknown_user')
    download_mode = request.args.get('download', 'false').lower() == 'true'
    
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"[{current_time}] Video request for case {case_id} by user {user}, download mode: {download_mode}")
    
    case_dir = ANALYSIS_FOLDER / case_id
    video_path = case_dir / "output_video.mp4"
    
    # Get case name for better filename
    result_file = case_dir / "analysis_results.json"
    case_name = None
    if result_file.exists():
        try:
            with open(result_file, 'r') as f:
                case_data = json.load(f)
            case_name = case_data.get('case_name')
        except Exception as e:
            logger.error(f"Error reading case name: {e}")
    
    original_filename = case_name or "analyzed_video"
    
    import re
    original_filename = re.sub(r'[^\w\s-]', '', original_filename).strip().replace(' ', '_')
    
    download_filename = f"{original_filename}.mp4"
    
    if not video_path.exists():
        logger.warning(f"Output video not found at: {video_path}")
        # Try to find any mp4 file
        for video_file in case_dir.glob("*.mp4"):
            logger.info(f"Found alternate video file: {video_file}")
            
            if download_mode:
                response = send_from_directory(
                    case_dir, 
                    video_file.name,
                    as_attachment=True,
                    download_name=download_filename,
                    mimetype="video/mp4"
                )
                response.headers["Content-Disposition"] = f"attachment; filename={download_filename}"
                response.headers["X-Content-Type-Options"] = "nosniff"
                logger.info(f"[{current_time}] Serving alternate video for download: {video_file}")
                return response
            else:
                # For viewing, just serve normally
                return send_from_directory(case_dir, video_file.name)
        
        return jsonify({'error': 'Output video not found'}), 404
    
    # Decide whether to serve for viewing or downloading
    if download_mode:
        logger.info(f"[{current_time}] Serving video for download: {video_path}")
        response = send_from_directory(
            case_dir, 
            "output_video.mp4",
            as_attachment=True,
            download_name=download_filename,
            mimetype="video/mp4"
        )
        # Add additional headers for download
        response.headers["Content-Disposition"] = f"attachment; filename={download_filename}"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response
    else:
        # For viewing, just serve normally
        logger.info(f"[{current_time}] Serving video for viewing: {video_path}")
        return send_from_directory(case_dir, "output_video.mp4")

# Add a dedicated download endpoint for better compatibility
@app.route('/api/cases/<case_id>/download_video', methods=['GET'])
def download_output_video(case_id):
    """Dedicated endpoint for downloading the analyzed video."""
    user = request.args.get('user', 'aaravgoel0')  # Default to the provided username
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    logger.info(f"[{current_time}] Video download requested for case {case_id} by user {user}")
    
    case_dir = ANALYSIS_FOLDER / case_id
    video_path = case_dir / "output_video.mp4"
    
    # Get case name for better filename
    result_file = case_dir / "analysis_results.json"
    case_name = None
    if result_file.exists():
        try:
            with open(result_file, 'r') as f:
                case_data = json.load(f)
            case_name = case_data.get('case_name')
        except Exception as e:
            logger.error(f"Error reading case name: {e}")
    
    # Generate a meaningful filename for download
    original_filename = case_name or "analyzed_video"
    
    # Clean up filename to ensure it's valid
    import re
    original_filename = re.sub(r'[^\w\s-]', '', original_filename).strip().replace(' ', '_')
    
    download_filename = f"{original_filename}.mp4"
    
    if not video_path.exists():
        logger.warning(f"Output video not found at: {video_path}")
        # Try to find any mp4 file
        for video_file in case_dir.glob("*.mp4"):
            logger.info(f"Found alternate video file: {video_file}")
            
            response = send_from_directory(
                case_dir, 
                video_file.name,
                as_attachment=True,
                download_name=download_filename,
                mimetype="video/mp4"
            )
            response.headers["Content-Disposition"] = f"attachment; filename={download_filename}"
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
            logger.info(f"[{current_time}] Serving alternate video for download: {video_file}")
            return response
        
        return jsonify({'error': 'Output video not found'}), 404
    
    # Serve for download with appropriate headers
    logger.info(f"[{current_time}] Serving video for download: {video_path}")
    response = send_from_directory(
        case_dir, 
        "output_video.mp4",
        as_attachment=True,
        download_name=download_filename,
        mimetype="video/mp4"
    )
    # Add additional headers for download
    response.headers["Content-Disposition"] = f"attachment; filename={download_filename}"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.route('/api/cases/<case_id>/persons', methods=['GET'])
def get_persons(case_id):
    """Get all persons detected in a case."""
    case_dir = ANALYSIS_FOLDER / case_id
    result_file = case_dir / "analysis_results.json"
    
    if not result_file.exists():
        return jsonify({'error': 'Case not found'}), 404
    
    try:
        with open(result_file, 'r') as f:
            results = json.load(f)
        
        # Extract person identities
        persons = results.get('person_identities', [])
        
        logger.info(f"Found {len(persons)} persons for case {case_id}")
        
        # Add case_name to each person for better display
        case_name = results.get('case_name', case_id)
        for person in persons:
            person['case_name'] = case_name
        
        # Add thumbnail paths
        reid_dir = case_dir / "reid_data" / "person_images"
        if reid_dir.exists():
            logger.info(f"Person images directory exists: {reid_dir}")
            for person in persons:
                person_id = person.get('id')
                # Look for thumbnail
                for ext in ['jpg', 'png']:
                    img_path = reid_dir / f"person_{person_id}.{ext}"
                    if img_path.exists():
                        person['thumbnail'] = f"/api/cases/{case_id}/persons/{person_id}/thumbnail"
                        break
                
                if 'thumbnail' not in person:
                    logger.warning(f"No thumbnail found for person {person_id} in {reid_dir}")
        else:
            logger.warning(f"Person images directory does not exist: {reid_dir}")
        
        return jsonify(persons)
    
    except Exception as e:
        logger.error(f"Error loading persons for case {case_id}: {e}")
        return jsonify({'error': f'Error loading persons: {str(e)}'}), 500

@app.route('/api/cases/<case_id>/persons/<person_id>', methods=['GET'])
def get_person(case_id, person_id):
    """Get details for a specific person."""
    case_dir = ANALYSIS_FOLDER / case_id
    result_file = case_dir / "analysis_results.json"
    
    if not result_file.exists():
        return jsonify({'error': 'Case not found'}), 404
    
    try:
        with open(result_file, 'r') as f:
            results = json.load(f)
        
        # Get case name
        case_name = results.get('case_name', case_id)
        
        # Find the person
        for person in results.get('person_identities', []):
            if str(person.get('id')) == str(person_id):
                # Add case name to the person record
                person['case_name'] = case_name
                
                # Add thumbnail path
                reid_dir = case_dir / "reid_data" / "person_images"
                if reid_dir.exists():
                    for ext in ['jpg', 'png']:
                        img_path = reid_dir / f"person_{person_id}.{ext}"
                        if img_path.exists():
                            person['thumbnail'] = f"/api/cases/{case_id}/persons/{person_id}/thumbnail"
                            break
                
                # Get appearances
                appearances = []
                for frame in results.get('frames', {}).values():
                    for det in frame.get('detections', []):
                        if det.get('person_id') == int(person_id):
                            appearances.append({
                                'frame': frame.get('frame_number'),
                                'timestamp': frame.get('timestamp'),
                                'box': det.get('box'),
                                'confidence': det.get('confidence')
                            })
                
                person['appearances'] = appearances
                
                # Get trajectory if available
                if '3d_analysis' in results and 'person_positions' in results['3d_analysis']:
                    positions = results['3d_analysis']['person_positions'].get(str(person_id), {})
                    if 'trajectory' in positions:
                        person['trajectory_3d'] = positions['trajectory']
                
                return jsonify(person)
        
        return jsonify({'error': 'Person not found'}), 404
    
    except Exception as e:
        logger.error(f"Error loading person {person_id} for case {case_id}: {e}")
        return jsonify({'error': f'Error loading person: {str(e)}'}), 500

@app.route('/api/cases/<case_id>/persons/<person_id>/thumbnail', methods=['GET'])
def get_person_thumbnail(case_id, person_id):
    """Get thumbnail for a specific person."""
    case_dir = ANALYSIS_FOLDER / case_id
    reid_dir = case_dir / "reid_data" / "person_images"
    
    if not reid_dir.exists():
        logger.warning(f"Person images directory does not exist: {reid_dir}")
        return jsonify({'error': 'Person images not found'}), 404
    
    # Look for thumbnail
    for ext in ['jpg', 'png']:
        img_path = reid_dir / f"person_{person_id}.{ext}"
        if img_path.exists():
            return send_from_directory(reid_dir, f"person_{person_id}.{ext}")
    
    logger.warning(f"No thumbnail found for person {person_id} in {reid_dir}")
    return jsonify({'error': 'Thumbnail not found'}), 404

@app.route('/api/cases/<case_id>/timeline', methods=['GET'])
def get_timeline(case_id):
    """Get timeline events for a case."""
    case_dir = ANALYSIS_FOLDER / case_id
    result_file = case_dir / "analysis_results.json"
    
    if not result_file.exists():
        return jsonify({'error': 'Case not found'}), 404
    
    try:
        with open(result_file, 'r') as f:
            results = json.load(f)
        
        # Extract timeline events
        timeline = []
        
        # Get case name
        case_name = results.get('case_name', case_id)
        
        # Add person appearances/disappearances
        for person in results.get('person_identities', []):
            person_id = person.get('id')
            metadata = person.get('metadata', {})
            
            # First appearance
            if 'first_seen_frame' in metadata and 'first_seen_time' in metadata:
                timeline.append({
                    'event_type': 'person_appearance',
                    'person_id': person_id,
                    'frame': metadata['first_seen_frame'],
                    'timestamp': metadata['first_seen_time'],
                    'description': f"Person #{person_id} first appears",
                    'case_name': case_name  # Add case name
                })
            
            # Last appearance
            if 'last_seen_frame' in metadata and 'last_seen_time' in metadata:
                timeline.append({
                    'event_type': 'person_disappearance',
                    'person_id': person_id,
                    'frame': metadata['last_seen_frame'],
                    'timestamp': metadata['last_seen_time'],
                    'description': f"Person #{person_id} last seen",
                    'case_name': case_name  # Add case name
                })
        
        # Add other events if available
        if 'timeline' in results:
            for event in results['timeline']:
                event['case_name'] = case_name  # Add case name to each event
                timeline.append(event)
        
        # Sort by timestamp
        timeline.sort(key=lambda x: x.get('timestamp', 0))
        
        return jsonify(timeline)
    
    except Exception as e:
        logger.error(f"Error loading timeline for case {case_id}: {e}")
        return jsonify({'error': f'Error loading timeline: {str(e)}'}), 500

@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Upload a video file for analysis."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Get case name from form data
    case_name = request.form.get('case_name', '')
    
    # Create a unique case ID
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    case_id = f"case_{timestamp}"
    
    # Create case directory
    case_dir = ANALYSIS_FOLDER / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the uploaded file
    video_path = UPLOAD_FOLDER / file.filename
    file.save(video_path)
    logger.info(f"File uploaded: {video_path} for case {case_id} with name '{case_name}'")
    
    return jsonify({
        'case_id': case_id,
        'case_name': case_name,  # Include the case name in the response
        'video_path': str(video_path),
        'status': 'uploaded'
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_video():
    """Start video analysis."""
    data = request.json
    
    if not data or 'video_path' not in data:
        return jsonify({'error': 'No video path provided'}), 400
    
    video_path = data['video_path']
    if not os.path.exists(video_path):
        return jsonify({'error': 'Video file not found'}), 404
    
    # Get case name and case ID
    case_name = data.get('case_name', '')
    case_id = data.get('case_id')
    
    # Create a unique job ID if not provided
    if not case_id:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        case_id = f"case_{timestamp}"
    
    # If no case name was provided, generate one from the video filename
    if not case_name:
        video_filename = os.path.basename(video_path)
        case_name = os.path.splitext(video_filename)[0].replace('_', ' ').title()
    
    # Create output directory
    output_dir = ANALYSIS_FOLDER / case_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a job for background processing
    job = {
        'case_id': case_id,
        'case_name': case_name,  # Store the case name
        'video_path': video_path,
        'output_dir': str(output_dir),
        'start_time': datetime.now().isoformat(),
        'status': 'queued',
        'progress': 0
    }
    
    # Add case details if provided
    if 'case_details' in data:
        job['case_details'] = data['case_details']
    
    # Add to active jobs
    active_jobs[case_id] = job
    
    logger.info(f"Starting analysis job for case {case_id} '{case_name}', video: {video_path}")
    
    # Start analysis in background
    import threading
    thread = threading.Thread(target=run_analysis, args=(case_id, case_name, video_path, output_dir, data.get('case_details')))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'case_id': case_id,
        'case_name': case_name,  # Include the case name in the response
        'status': 'processing',
        'message': 'Analysis started'
    })

@app.route('/api/search', methods=['POST'])
def search():
    """Search for persons or events."""
    data = request.json
    
    if not data:
        return jsonify({'error': 'No search criteria provided'}), 400
    
    search_type = data.get('type', 'person')
    query = data.get('query', '')
    case_id = data.get('case_id')
    
    results = []
    
    # Search in a specific case
    if case_id:
        case_dir = ANALYSIS_FOLDER / case_id
        result_file = case_dir / "analysis_results.json"
        
        if not result_file.exists():
            return jsonify({'error': 'Case not found'}), 404
        
        try:
            with open(result_file, 'r') as f:
                case_results = json.load(f)
            
            # Get case name
            case_name = case_results.get('case_name', case_id)
            
            if search_type == 'person':
                # Search for persons
                for person in case_results.get('person_identities', []):
                    # Search by ID
                    person_id = person.get('id')
                    if query and str(person_id) == query:
                        results.append({
                            'type': 'person',
                            'case_id': case_id,
                            'case_name': case_name,  # Include the case name
                            'id': person_id,
                            'thumbnail': f"/api/cases/{case_id}/persons/{person_id}/thumbnail",
                            'metadata': person.get('metadata', {})
                        })
            
            elif search_type == 'event':
                # Search for events
                for event in case_results.get('timeline', []):
                    # Search by description or type
                    if query.lower() in event.get('description', '').lower() or \
                       query.lower() in event.get('event_type', '').lower():
                        results.append({
                            'type': 'event',
                            'case_id': case_id,
                            'case_name': case_name,  # Include the case name
                            'event_type': event.get('event_type'),
                            'timestamp': event.get('timestamp'),
                            'frame': event.get('frame'),
                            'description': event.get('description')
                        })
            
        except Exception as e:
            logger.error(f"Error searching in case {case_id}: {e}")
            return jsonify({'error': f'Error searching: {str(e)}'}), 500
    
    # Search across all cases
    else:
        for case_dir in ANALYSIS_FOLDER.iterdir():
            if case_dir.is_dir():
                result_file = case_dir / "analysis_results.json"
                if result_file.exists():
                    try:
                        with open(result_file, 'r') as f:
                            case_results = json.load(f)
                        
                        # Get case name
                        case_name = case_results.get('case_name', case_dir.name)
                        
                        if search_type == 'person':
                            # Search for persons
                            for person in case_results.get('person_identities', []):
                                # Search by ID
                                person_id = person.get('id')
                                if query and str(person_id) == query:
                                    results.append({
                                        'type': 'person',
                                        'case_id': case_dir.name,
                                        'case_name': case_name,  # Include the case name
                                        'id': person_id,
                                        'thumbnail': f"/api/cases/{case_dir.name}/persons/{person_id}/thumbnail",
                                        'metadata': person.get('metadata', {})
                                    })
                        
                        elif search_type == 'event':
                            # Search for events
                            for event in case_results.get('timeline', []):
                                # Search by description or type
                                if query.lower() in event.get('description', '').lower() or \
                                   query.lower() in event.get('event_type', '').lower():
                                    results.append({
                                        'type': 'event',
                                        'case_id': case_dir.name,
                                        'case_name': case_name,  # Include the case name
                                        'event_type': event.get('event_type'),
                                        'timestamp': event.get('timestamp'),
                                        'frame': event.get('frame'),
                                        'description': event.get('description')
                                    })
                    
                    except Exception as e:
                        logger.error(f"Error searching in case {case_dir.name}: {e}")
    
    return jsonify(results)

# Enhanced chat endpoint with N-shot prompting and RAG
@app.route('/api/chat', methods=['POST'])
def chat_with_case():
    """Chat with the AI about a specific case using N-shot prompting and RAG."""
    data = request.json
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    case_id = data.get('caseId')
    if not case_id:
        return jsonify({'error': 'No case ID provided'}), 400
    
    message = data.get('message')
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    # Check if case exists
    case_dir = ANALYSIS_FOLDER / case_id
    if not case_dir.exists():
        return jsonify({'error': 'Case not found'}), 404
    
    # Load case data
    try:
        # Load main case data
        with open(case_dir / "analysis_results.json", 'r') as f:
            case_data = json.load(f)
            
        # Add case_id to the data if it's not there
        if 'case_id' not in case_data:
            case_data['case_id'] = case_id
        
        # Make sure case_name is available
        if 'case_name' not in case_data:
            case_data['case_name'] = case_data.get('case_name', case_id)
        
        # Import our enhanced chat service and generate response
        try:
            # First try to import in a way that avoids circular imports
            from chat_service import generate_chat_response
            response_text = generate_chat_response(case_id, case_data, message)
        except ImportError:
            # If the import fails, define a fallback function
            logger.warning("Failed to import chat_service, using fallback response")
            case_name = case_data.get('case_name', f"case #{case_id}")
            response_text = f"I can answer questions about the case '{case_name}'. This case contains {len(case_data.get('person_identities', []))} persons. Please ask a specific question about the video or persons detected."
        
        return jsonify({"response": response_text})
    
    except Exception as e:
        logger.error(f"Error in chat_with_case: {e}")
        logger.error(traceback.format_exc())
        # Provide a graceful fallback response
        return jsonify({
            "response": f"I encountered a problem accessing the case data. The case appears to exist, but I couldn't process the information properly. Please try a specific question about the video analysis."
        })

# New endpoint for Server-Sent Events to notify when analysis completes
@app.route('/api/events/analysis/<case_id>', methods=['GET'])
def analysis_events(case_id):
    """Stream notifications about analysis status using Server-Sent Events (SSE)."""
    def generate():
        """Generate SSE events."""
        # Register this client for the case
        client_id = request.remote_addr
        with analysis_clients_lock:
            if case_id not in analysis_clients:
                analysis_clients[case_id] = set()
            analysis_clients[case_id].add(client_id)
        
        try:
            # Send initial status
            if case_id in active_jobs:
                progress = active_jobs[case_id].get('progress', 0)
                status = active_jobs[case_id].get('status', 'processing')
                yield f"data: {json.dumps({'status': status, 'progress': progress})}\n\n"
            elif case_id in completed_jobs or (ANALYSIS_FOLDER / case_id).exists():
                # If already completed or exists as a directory
                yield f"data: {json.dumps({'status': 'completed', 'progress': 100})}\n\n"
            else:
                yield f"data: {json.dumps({'status': 'unknown', 'progress': 0})}\n\n"
            
            # Keep the connection open until the client disconnects or the analysis completes
            timeout = time.time() + 1800  # 30 minutes max
            while time.time() < timeout:
                # Check if analysis is still active
                if case_id in active_jobs:
                    progress = active_jobs[case_id].get('progress', 0)
                    status = active_jobs[case_id].get('status', 'processing')
                    yield f"data: {json.dumps({'status': status, 'progress': progress})}\n\n"
                    
                    # If status is failed, we're done
                    if status == 'failed':
                        error = active_jobs[case_id].get('error', 'Unknown error')
                        yield f"data: {json.dumps({'status': 'failed', 'error': error})}\n\n"
                        break
                    
                # Check if it's been completed
                elif case_id in completed_jobs or (ANALYSIS_FOLDER / case_id).exists():
                    yield f"data: {json.dumps({'status': 'completed', 'progress': 100})}\n\n"
                    break
                
                # Wait before sending the next update
                time.sleep(2)
        
        finally:
            # Unregister this client when done
            with analysis_clients_lock:
                if case_id in analysis_clients and client_id in analysis_clients[case_id]:
                    analysis_clients[case_id].remove(client_id)
                    if not analysis_clients[case_id]:
                        del analysis_clients[case_id]
    
    # Set up the SSE response
    response = Response(stream_with_context(generate()), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"  # Disable buffering in Nginx
    response.headers["Connection"] = "keep-alive"
    return response

def notify_analysis_completion(case_id, status='completed', error=None):
    """Notify all clients waiting for this analysis to complete."""
    with analysis_clients_lock:
        if case_id not in analysis_clients:
            return
        
        logger.info(f"Notifying {len(analysis_clients[case_id])} clients about case {case_id} completion")
        # The actual notification happens in the SSE endpoint

def run_analysis(case_id, case_name, video_path, output_dir, case_details=None):
    """Run the video analysis in the background."""
    print("\n===== STARTING VIDEO ANALYSIS =====")
    print(f"Case ID: {case_id}")
    print(f"Case Name: {case_name}")
    print(f"Video path: {video_path}")
    print(f"Output directory: {output_dir}")
    print("===================================\n")
    
    # First try to find weapon model path
    weapon_model_path = None
    try:
        # Get absolute path to weapon model using multiple methods
        script_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(script_dir, "models", "weapon_detect.pt"),
            os.path.join(script_dir, "../models", "weapon_detect.pt"),
            os.path.join(script_dir, "backend/models", "weapon_detect.pt"),
            os.path.join(os.getcwd(), "backend/models", "weapon_detect.pt"),
            os.path.join(os.getcwd(), "models", "weapon_detect.pt")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                weapon_model_path = path
                logger.info(f"✅ Found weapon model at: {path}")
                print(f"✅ Found weapon model at: {path}")
                break
            else:
                logger.info(f"❌ Model not found at: {path}")
                print(f"❌ Model not found at: {path}")
        
        if weapon_model_path is None:
            logger.error("⚠️ Could not find weapon model in any expected location!")
            print("⚠️ Could not find weapon model in any expected location!")
            # Try direct loading approach
            try:
                # Try to load directly using WeaponDetector to see what happens
                from backend.models.weapon_detector import WeaponDetector
                test_detector = WeaponDetector()
                logger.info(f"Direct WeaponDetector creation: {'success' if test_detector is not None else 'failed'}")
                logger.info(f"Model loaded: {'yes' if test_detector.model is not None else 'no'}")
                print(f"Direct WeaponDetector creation: {'success' if test_detector is not None else 'failed'}")
                print(f"Model loaded: {'yes' if test_detector.model is not None else 'no'}")
            except Exception as e:
                logger.error(f"Direct WeaponDetector creation failed: {e}")
                print(f"Direct WeaponDetector creation failed: {e}")
    except Exception as e:
        logger.error(f"Error during weapon model path resolution: {e}")
        print(f"Error during weapon model path resolution: {e}")
        logger.error(traceback.format_exc())

    try:
        # Update job status
        active_jobs[case_id]['status'] = 'processing'
        logger.info(f"Starting analysis for case {case_id} '{case_name}', video: {video_path}")
        
        # Check if video file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Create output directory if it doesn't exist
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components with optimized parameters for person detection
        detector = ObjectDetector(model_size='m', confidence_threshold=0.15)  # Very low threshold
        tracker = ObjectTracker(max_age=60, min_hits=1, iou_threshold=0.2)    # More forgiving tracker
        reidentifier = PersonReidentifier()
        
        # Create custom class-specific thresholds to balance detection with false-positive reduction
        class_confidence_thresholds = {
            'person': 0.15,         # Keep very low for person detection
            'car': 0.45,            # Higher threshold for vehicles
            'truck': 0.45,
            'bus': 0.45,
            'bicycle': 0.35,
            'motorcycle': 0.35,
            'knife': 0.55,          # Higher for weapons
            'gun': 0.6,
            'default': 0.35         # Default threshold for other classes
        }
        
        # Initialize the enhanced filter with our settings but customize for this use case
        enhanced_filter = EnhancedFilter(
            class_confidence_thresholds=class_confidence_thresholds,
            motion_threshold=0.3,   # Lower motion threshold for static scenes
            temporal_consistency_frames=2  # Fewer frames for consistency to maximize person detection
        )
        
        # Get the correct path to the model
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if not weapon_model_path:
            weapon_model_path = os.path.join(current_dir, "models", "weapon_detect.pt")
        
        # Print path for debugging
        logger.info(f"Weapon model path: {weapon_model_path}")
        logger.info(f"Weapon model exists: {os.path.exists(weapon_model_path)}")
        print(f"Weapon model path: {weapon_model_path}")
        print(f"Weapon model exists: {os.path.exists(weapon_model_path)}")
        
        logger.info("Creating VideoAnalyzerWithReID...")
        try:
            video_analyzer = VideoAnalyzerWithReID(
                detector=detector,
                tracker=tracker,
                reidentifier=reidentifier,
                output_dir=str(output_dir),
                enable_reid=True,
                enable_enhanced_filtering=True,
                enable_two_stage_detection=True,
                enable_weapon_detection=True,
                enable_interaction_detection=True,
                weapon_model_path=weapon_model_path
            )
            logger.info("VideoAnalyzerWithReID created successfully")
            
            # Check if weapon detector was initialized
            print("\n=== WEAPON DETECTOR STATUS CHECK ===")
            if hasattr(video_analyzer, 'weapon_detector'):
                print(f"Weapon detector exists: Yes")
                print(f"Weapon detector is None: {video_analyzer.weapon_detector is None}")
                logger.info(f"Weapon detector attribute exists: {video_analyzer.weapon_detector is not None}")
                if video_analyzer.weapon_detector is not None:
                    print(f"Model attribute exists: {hasattr(video_analyzer.weapon_detector, 'model')}")
                    if hasattr(video_analyzer.weapon_detector, 'model'):
                        print(f"Model is None: {video_analyzer.weapon_detector.model is None}")
                    logger.info(f"Weapon detector model: {video_analyzer.weapon_detector.model is not None}")
            else:
                print("Weapon detector does not exist on video_analyzer")
                logger.error("Weapon detector attribute doesn't exist in video_analyzer!")
            print("===================================\n")
        except Exception as e:
            logger.error(f"Error creating VideoAnalyzerWithReID: {e}")
            print(f"Error creating VideoAnalyzerWithReID: {e}")
            logger.error(traceback.format_exc())
            raise
        
        logger.info(f"Running video analysis with enhanced false-positive reduction for case {case_id}")
        
        # Analyze video with our false-positive reduction enabled
        results = video_analyzer.analyze_video(
            video_path=video_path,
            frame_interval=2,  # Process every 2nd frame for better tracking
            save_video=True,
            enable_enhanced_filtering=True,
            enable_two_stage_detection=True,
            enable_weapon_detection=True,
            enable_interaction_detection=True,
        )
        
        # Log filtering effectiveness
        total_detections = results.get('total_detections', 0)
        valid_detections = results.get('valid_detections', 0)
        filtered_detections = results.get('filtered_detections', 0)
        
        if total_detections > 0:
            filter_ratio = filtered_detections / total_detections
            logger.info(f"False positive reduction: {filtered_detections} filtered out of {total_detections} " +
                       f"({filter_ratio:.1%} reduction rate)")
        
        # Check if persons were detected
        person_count = len(results.get('person_identities', []))
        logger.info(f"Analysis completed with {person_count} persons detected")
        
        # If no persons detected, try to extract them from tracks
        if person_count == 0 and 'tracks' in results:
            logger.info("No persons detected, attempting to create from tracks...")
            person_identities = []
            person_id = 1
            
            # Find person tracks
            for track_id, track in results.get('tracks', {}).items():
                if track.get('class_name', '').lower() == 'person':
                    # Create person from track
                    person = {
                        'id': person_id,
                        'metadata': {
                            'appearances': len(track.get('detections', [])),
                            'first_seen_frame': track.get('first_frame', 0),
                            'last_seen_frame': track.get('last_frame', 0),
                            'first_seen_time': track.get('first_time', 0),
                            'last_seen_time': track.get('last_time', 0),
                            'created_from_track': True
                        }
                    }
                    person_identities.append(person)
                    person_id += 1
            
            # Update results with extracted persons
            if person_identities:
                results['person_identities'] = person_identities
                logger.info(f"Created {len(person_identities)} persons from tracks")
        
        # Create a thumbnail for the case
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            # Jump to the middle of the video
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
            ret, frame = cap.read()
            if ret:
                thumbnail_path = output_dir / "thumbnail.jpg"
                cv2.imwrite(str(thumbnail_path), frame)
                logger.info(f"Created thumbnail: {thumbnail_path}")
            cap.release()
        except Exception as e:
            logger.error(f"Error creating thumbnail: {e}")
        
        # Verify output video exists
        output_video = output_dir / "output_video.mp4"
        if output_video.exists():
            logger.info(f"Output video verified at: {output_video}")
        else:
            logger.warning(f"Output video not found at: {output_video}")
            # Check for output video in other locations
            for video_path in output_dir.glob("*.mp4"):
                logger.info(f"Found video file: {video_path}")
                if video_path.name != "output_video.mp4":
                    # Copy or rename to standard name
                    import shutil
                    shutil.copy(video_path, output_dir / "output_video.mp4")
                    logger.info(f"Copied {video_path} to {output_dir / 'output_video.mp4'}")
                    break
        
        # Generate report
        report = video_analyzer.generate_analysis_report(results)
        
        # Make sure the report contains persons
        if 'person_identities' not in report or len(report['person_identities']) == 0:
            report['person_identities'] = results.get('person_identities', [])
        
        # Add case name to the report
        report['case_name'] = case_name
        
        # Add case details if provided
        if case_details:
            report['case_details'] = case_details
        
        # Add custom fields to the report for debugging
        report["timestamp"] = datetime.now().isoformat()
        report["video_path"] = str(video_path)
        report["output_directory"] = str(output_dir)
        report["debug_info"] = {
            "total_tracks": len(results.get('tracks', [])),
            "person_tracks": sum(1 for t in results.get('tracks', {}).values() if t.get('class_name', '').lower() == 'person'),
            "original_person_count": person_count,
            "final_person_count": len(report.get('person_identities', [])),
            "confidence_threshold": detector.confidence_threshold,
            "enhanced_filtering": True,
            "two_stage_detection": True,
            "false_positive_reduction": {
                "total_detections": total_detections,
                "valid_detections": valid_detections,
                "filtered_detections": filtered_detections,
                "reduction_rate": filtered_detections / total_detections if total_detections > 0 else 0
            },
            "processing": {
                "frame_interval": 2,
                "tracker_params": {
                    "max_age": tracker.max_age,
                    "min_hits": tracker.min_hits,
                    "iou_threshold": tracker.iou_threshold
                }
            }
        }
        
        # Save results
        with open(output_dir / "analysis_results.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save raw results for debugging
        with open(output_dir / "analysis_raw_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Update job status
        active_jobs[case_id]['status'] = 'completed'
        active_jobs[case_id]['progress'] = 100
        
        # Notify any waiting clients
        notify_analysis_completion(case_id, 'completed')
        
        # Move to completed jobs
        completed_jobs[case_id] = active_jobs[case_id].copy()
        del active_jobs[case_id]
        
        logger.info(f"Analysis completed for case {case_id} '{case_name}'")
    
    except Exception as e:
        logger.error(f"Error in analysis job {case_id} '{case_name}': {e}")
        logger.error(traceback.format_exc())
        active_jobs[case_id]['status'] = 'failed'
        active_jobs[case_id]['error'] = str(e)
        
        # Notify any waiting clients about the failure
        notify_analysis_completion(case_id, 'failed', str(e))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
