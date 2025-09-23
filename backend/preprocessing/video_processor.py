import cv2
import numpy as np
import os
import json
import time
from pathlib import Path
from datetime import datetime


class VideoProcessor:
    """
    Core video preprocessing class for forensic analysis.
    Handles video loading, frame extraction, and basic enhancements.
    """
    
    def __init__(self, input_path, output_dir=None, case_id=None):
        """
        Initialize the video processor.
        
        Parameters:
        -----------
        input_path : str
            Path to the input video file
        output_dir : str, optional
            Directory to save processed outputs. If None, creates a timestamped directory.
        case_id : str, optional
            Unique identifier for this case/processing session
        """
        self.input_path = input_path

        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(f"./data/processed/video_{timestamp}")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate case ID if not provided
        self.case_id = case_id or f"case_{int(time.time())}"
        
        # Load video
        self.cap = cv2.VideoCapture(input_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {input_path}")
        
        # Extract video properties
        self._extract_metadata()
        
        # Create a log for processing steps
        self.processing_log = []
        self._log_step("Initialized video processor", {"input_path": input_path})
        
    def _extract_metadata(self):
        """Extract and store video metadata."""
        # Basic properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0
        
        # Video codec
        fourcc_int = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        self.fourcc = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
        
        # Store in metadata dict
        self.metadata = {
            "filename": os.path.basename(self.input_path),
            "filepath": self.input_path,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "duration": self.duration,
            "codec": self.fourcc,
            "case_id": self.case_id,
            "processing_date": datetime.now().isoformat()
        }
        
        # Save metadata to file
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=4)
            
        return self.metadata
    
    def _log_step(self, step_name, details=None):
        """Log a processing step with timestamp."""
        log_entry = {
            "step": step_name,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        self.processing_log.append(log_entry)
        
        # Save log to file
        log_path = self.output_dir / "processing_log.json"
        with open(log_path, "w") as f:
            json.dump(self.processing_log, f, indent=4)
    
    def extract_frames(self, output_subdir="frames", interval=1, max_frames=None):
        """
        Extract frames from the video.
        
        Parameters:
        -----------
        output_subdir : str
            Subdirectory name for saving frames
        interval : int
            Extract every nth frame
        max_frames : int, optional
            Maximum number of frames to extract. If None, extract all frames (based on interval).
            
        Returns:
        --------
        list
            Paths to extracted frames
        """
        frames_dir = self.output_dir / output_subdir
        frames_dir.mkdir(exist_ok=True)
        
        self._log_step("Extracting frames", {
            "interval": interval,
            "max_frames": max_frames,
            "output_dir": str(frames_dir)
        })
        
        # Reset video capture to start
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        frame_paths = []
        frame_idx = 0
        saved_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            if frame_idx % interval == 0:
                frame_path = frames_dir / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                frame_paths.append(str(frame_path))
                saved_count += 1
                
                if max_frames is not None and saved_count >= max_frames:
                    break
            
            frame_idx += 1
            
            # Show progress for large videos
            if frame_idx % 500 == 0:
                print(f"Processed {frame_idx}/{self.frame_count} frames")
        
        self._log_step("Frame extraction complete", {
            "total_extracted": len(frame_paths),
            "first_frame": frame_paths[0] if frame_paths else None,
            "last_frame": frame_paths[-1] if frame_paths else None
        })
        
        return frame_paths
    
    def denoise_video(self, output_filename=None, strength=10):
        """
        Apply denoising to the video.
        
        Parameters:
        -----------
        output_filename : str, optional
            Name for output file. If None, uses automatic naming.
        strength : int
            Denoising strength (higher = more denoising but potential detail loss)
            
        Returns:
        --------
        str
            Path to the denoised video
        """
        if output_filename is None:
            base_filename = os.path.splitext(os.path.basename(self.input_path))[0]
            output_filename = f"{base_filename}_denoised.mp4"
            
        output_path = self.output_dir / output_filename
        
        self._log_step("Denoising video", {
            "strength": strength,
            "output_path": str(output_path)
        })
        
        # Reset video capture
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec for compatibility
        out = cv2.VideoWriter(str(output_path), fourcc, self.fps, (self.width, self.height))
        
        frame_count = 0
        total_frames = int(self.frame_count)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Apply fastNlMeansDenoisingColored for color videos
            denoised = cv2.fastNlMeansDenoisingColored(
                frame, 
                None, 
                strength,  # Luminance component
                strength,  # Color components
                7,  # Template window size
                21  # Search window size
            )
            
            # Write denoised frame
            out.write(denoised)
            
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Denoising progress: {frame_count}/{total_frames} frames ({(frame_count/total_frames)*100:.1f}%)")
        
        # Release resources
        out.release()
        
        self._log_step("Denoising complete", {
            "processed_frames": frame_count,
            "output_path": str(output_path)
        })
        
        return str(output_path)
    
    def stabilize_video(self, output_filename=None, smooth_radius=30):
        """
        Apply video stabilization to reduce camera shake.
        
        Parameters:
        -----------
        output_filename : str, optional
            Name for output file. If None, uses automatic naming.
        smooth_radius : int
            Radius for trajectory smoothing (higher = smoother but less responsive)
            
        Returns:
        --------
        str
            Path to the stabilized video
        """
        if output_filename is None:
            base_filename = os.path.splitext(os.path.basename(self.input_path))[0]
            output_filename = f"{base_filename}_stabilized.mp4"
            
        output_path = self.output_dir / output_filename
        
        self._log_step("Stabilizing video", {
            "smooth_radius": smooth_radius,
            "output_path": str(output_path)
        })
        
        # Reset video capture
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Get the first frame
        ret, prev_frame = self.cap.read()
        if not ret:
            raise ValueError("Failed to read the first frame")
            
        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        # Create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, self.fps, (self.width, self.height))
        
        # Pre-define transformation-store array
        transforms = np.zeros((self.frame_count-1, 3), np.float32)
        
        print("Step 1/3: Analyzing frame transformations...")
        for i in range(self.frame_count-1):
            # Read next frame
            ret, curr_frame = self.cap.read()
            if not ret:
                print(f"Reached end of video at frame {i+1}/{self.frame_count}")
                break
                
            # Convert to grayscale
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            
            # Detect feature points in previous frame
            prev_pts = cv2.goodFeaturesToTrack(
                prev_gray,
                maxCorners=200,
                qualityLevel=0.01,
                minDistance=30,
                blockSize=3
            )
            
            # Handle case when no features are found
            if prev_pts is None:
                transforms[i] = [0, 0, 0]
                prev_gray = curr_gray
                if i % 100 == 0:
                    print(f"Progress: {i+1}/{self.frame_count-1} frames")
                continue
                
            # Calculate optical flow (i.e. track feature points)
            curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
            
            # Filter only valid points
            idx = np.where(status == 1)[0]
            if len(idx) < 4:  # Need at least 4 points for homography
                transforms[i] = [0, 0, 0]
                prev_gray = curr_gray
                if i % 100 == 0:
                    print(f"Progress: {i+1}/{self.frame_count-1} frames")
                continue
                
            # Keep only valid points
            prev_pts = prev_pts[idx]
            curr_pts = curr_pts[idx]
            
            # Find transformation matrix
            m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
            
            # Handle case where transform estimation fails
            if m is None:
                transforms[i] = [0, 0, 0]
                prev_gray = curr_gray
                if i % 100 == 0:
                    print(f"Progress: {i+1}/{self.frame_count-1} frames")
                continue
                
            # Extract translation and rotation angle
            dx = m[0, 2]
            dy = m[1, 2]
            da = np.arctan2(m[1, 0], m[0, 0])
            
            # Store transformation
            transforms[i] = [dx, dy, da]
            
            # Move to next frame
            prev_gray = curr_gray
            
            # Show progress
            if i % 100 == 0:
                print(f"Progress: {i+1}/{self.frame_count-1} frames")
        
        # Compute trajectory using cumulative sum of transformations
        print("Step 2/3: Computing smoothed trajectory...")
        trajectory = np.cumsum(transforms, axis=0)
        
        # Smooth trajectory using moving average filter
        smoothed_trajectory = np.copy(trajectory)
        
        # Apply Gaussian smoothing to the trajectory
        kernel_size = 2 * smooth_radius + 1
        for i in range(3):
            smoothed_trajectory[:, i] = cv2.GaussianBlur(
                trajectory[:, i], 
                (kernel_size, 1), 
                smooth_radius,
                borderType=cv2.BORDER_REFLECT
            )
        
        # Calculate difference between smoothed and original trajectory
        difference = smoothed_trajectory - trajectory
        
        # Apply transformations
        print("Step 3/3: Applying stabilizing transforms...")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Apply transformations to the first frame
        ret, frame = self.cap.read()
        if ret:
            out.write(frame)  # First frame remains unchanged
        
        for i in range(self.frame_count-1):
            # Read next frame
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Extract transformation
            dx = difference[i, 0]
            dy = difference[i, 1]
            da = difference[i, 2]
            
            # Reconstruct transformation matrix
            m = np.zeros((2, 3), np.float32)
            m[0, 0] = np.cos(da)
            m[0, 1] = -np.sin(da)
            m[1, 0] = np.sin(da)
            m[1, 1] = np.cos(da)
            m[0, 2] = dx
            m[1, 2] = dy
            
            # Apply affine transformation
            stabilized_frame = cv2.warpAffine(
                frame, 
                m, 
                (self.width, self.height),
                borderMode=cv2.BORDER_REPLICATE
            )
            
            # Write stabilized frame
            out.write(stabilized_frame)
            
            # Show progress
            if i % 100 == 0:
                print(f"Applying transforms: {i+1}/{self.frame_count-1} frames")
        
        # Release resources
        out.release()
        
        self._log_step("Stabilization complete", {
            "processed_frames": self.frame_count,
            "output_path": str(output_path)
        })
        
        return str(output_path)
    
    def enhance_low_light(self, output_filename=None, clip_limit=2.0, grid_size=(8, 8)):
        """
        Enhance low-light video using CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        Parameters:
        -----------
        output_filename : str, optional
            Name for output file. If None, uses automatic naming.
        clip_limit : float
            Threshold for contrast limiting
        grid_size : tuple
            Size of grid for histogram equalization
            
        Returns:
        --------
        str
            Path to the enhanced video
        """
        if output_filename is None:
            base_filename = os.path.splitext(os.path.basename(self.input_path))[0]
            output_filename = f"{base_filename}_enhanced.mp4"
            
        output_path = self.output_dir / output_filename
        
        self._log_step("Enhancing low-light video", {
            "clip_limit": clip_limit,
            "grid_size": grid_size,
            "output_path": str(output_path)
        })
        
        # Reset video capture
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, self.fps, (self.width, self.height))
        
        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        
        frame_count = 0
        total_frames = int(self.frame_count)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Convert to LAB color space (L: lightness, A: green-red, B: blue-yellow)
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            cl = clahe.apply(l)
            
            # Merge back with original A and B channels
            enhanced_lab = cv2.merge((cl, a, b))
            
            # Convert back to BGR color space
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            # Write enhanced frame
            out.write(enhanced)
            
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Enhancement progress: {frame_count}/{total_frames} frames ({(frame_count/total_frames)*100:.1f}%)")
        
        # Release resources
        out.release()
        
        self._log_step("Low-light enhancement complete", {
            "processed_frames": frame_count,
            "output_path": str(output_path)
        })
        
        return str(output_path)
    
    def analyze_frames(self, interval=30):
        """
        Perform basic frame analysis to detect potential issues.
        
        Parameters:
        -----------
        interval : int
            Analyze every nth frame
            
        Returns:
        --------
        dict
            Analysis results
        """
        # Reset video capture
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        self._log_step("Starting frame analysis", {"interval": interval})
        
        # Initialize analysis variables
        analysis = {
            "blur_scores": [],
            "brightness_values": [],
            "contrast_values": [],
            "potential_issues": []
        }
        
        frame_idx = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            if frame_idx % interval == 0:
                # Convert to grayscale for analysis
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Measure blurriness (Laplacian variance - lower means more blurry)
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                blur_score = laplacian.var()
                analysis["blur_scores"].append((frame_idx, blur_score))
                
                # Measure brightness (mean pixel value)
                brightness = gray.mean()
                analysis["brightness_values"].append((frame_idx, brightness))
                
                # Measure contrast (standard deviation of pixel values)
                contrast = gray.std()
                analysis["contrast_values"].append((frame_idx, contrast))
                
                # Detect potential issues
                issues = []
                
                if blur_score < 100:
                    issues.append("Blurry")
                
                if brightness < 50:
                    issues.append("Dark")
                elif brightness > 200:
                    issues.append("Overexposed")
                    
                if contrast < 30:
                    issues.append("Low contrast")
                
                if issues:
                    analysis["potential_issues"].append({
                        "frame": frame_idx,
                        "issues": issues,
                        "blur_score": blur_score,
                        "brightness": brightness,
                        "contrast": contrast
                    })
            
            frame_idx += 1
        
        # Save analysis results
        analysis_path = self.output_dir / "frame_analysis.json"
        with open(analysis_path, "w") as f:
            # Convert numpy values to Python native types for JSON serialization
            serializable_analysis = {
                "blur_scores": [(int(idx), float(score)) for idx, score in analysis["blur_scores"]],
                "brightness_values": [(int(idx), float(val)) for idx, val in analysis["brightness_values"]],
                "contrast_values": [(int(idx), float(val)) for idx, val in analysis["contrast_values"]],
                "potential_issues": [
                    {
                        "frame": int(issue["frame"]),
                        "issues": issue["issues"],
                        "blur_score": float(issue["blur_score"]),
                        "brightness": float(issue["brightness"]),
                        "contrast": float(issue["contrast"])
                    }
                    for issue in analysis["potential_issues"]
                ]
            }
            json.dump(serializable_analysis, f, indent=4)
        
        self._log_step("Frame analysis complete", {
            "analysis_path": str(analysis_path),
            "issues_found": len(analysis["potential_issues"])
        })
        
        return analysis
    
    def create_summary(self):
        """
        Create a summary of all processing steps and results.
        
        Returns:
        --------
        dict
            Processing summary
        """
        summary = {
            "case_id": self.case_id,
            "input_video": self.input_path,
            "output_directory": str(self.output_dir),
            "metadata": self.metadata,
            "processing_steps": self.processing_log,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save summary to file
        summary_path = self.output_dir / "processing_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)
            
        return summary
    
    def close(self):
        """Release resources."""
        if self.cap is not None:
            self.cap.release()