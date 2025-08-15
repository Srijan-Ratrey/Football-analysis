"""
Video processing module for football analysis pipeline.
"""

import cv2
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from ..config import OUTPUT_DIR, RESIZE_WIDTH, VIDEO_CODEC, USE_PITCH_MASK
from ..detectors.pose_detector import PoseDetector
from ..detectors.ball_tracker import BallTracker


class VideoProcessor:
    """
    Main video processing pipeline for football analysis.
    Handles video I/O, frame processing, and detection coordination.
    """
    
    def __init__(self, video_path: Path):
        """
        Initialize the video processor.
        
        Args:
            video_path: Path to the input video file
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(str(video_path))
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize data storage
        self.pose_data = []
        self.ball_data = []
        
        print(f"📹 Video: {video_path.name} | {self.width}x{self.height} | "
              f"{self.fps:.1f}fps | {self.frame_count/self.fps:.1f}s")
    
    def setup_video_writer(self, output_path: Path, 
                          resize_width: int = RESIZE_WIDTH) -> cv2.VideoWriter:
        """
        Setup video writer for annotated output.
        
        Args:
            output_path: Path for output video
            resize_width: Width for resized output
            
        Returns:
            Configured VideoWriter object
        """
        # Calculate output dimensions
        scale = resize_width / self.width
        output_height = int(self.height * scale)
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            self.fps,
            (resize_width, output_height)
        )
        
        return writer
    
    def process_frame(self, frame, pose_detector: PoseDetector, 
                     ball_tracker: BallTracker, frame_idx: int,
                     player_detector=None) -> Dict[str, Any]:
        """
        Process a single frame for pose and ball detection.
        
        Args:
            frame: Input video frame
            pose_detector: PoseDetector instance
            ball_tracker: BallTracker instance
            frame_idx: Current frame index
            
        Returns:
            Dictionary containing detection results
        """
        timestamp = frame_idx / self.fps
        
        # Compute pitch mask (green field) to improve robustness
        pitch_mask = self._compute_pitch_mask(frame) if USE_PITCH_MASK else None

        # Detect pose
        landmarks, pose_results = pose_detector.detect_pose(frame)
        key_points = pose_detector.get_key_points(landmarks)
        
        # Detect ball
        ball_detections = ball_tracker.detect_ball(frame)
        # Filter ball detections to pitch region if mask is reliable
        if pitch_mask is not None:
            green_ratio = pitch_mask.mean()
            # Only apply mask if the field occupies a reasonable portion
            if green_ratio > 0.15:
                filtered = []
                h, w = pitch_mask.shape[:2]
                for det in ball_detections:
                    cx, cy = det['center']
                    if 0 <= cy < h and 0 <= cx < w and pitch_mask[cy, cx] > 0:
                        filtered.append(det)
                # Apply only if we retain a reasonable fraction of detections
                if filtered and (len(filtered) >= max(1, len(ball_detections) // 2)):
                    ball_detections = filtered
        
        # Detect players
        player_detections = []
        if player_detector:
            player_detections = player_detector.detect_players(frame)
            # Assign teams based on jersey colors
            player_detections = player_detector.assign_teams(frame, player_detections)
        
        # Store frame data
        frame_data = {
            'frame_idx': frame_idx,
            'timestamp': timestamp,
            'landmarks': landmarks,
            'pose_results': pose_results,
            'key_points': key_points,
            'ball_detections': ball_detections,
            'player_detections': player_detections,
            'frame_width': frame.shape[1],
            'frame_height': frame.shape[0],
            'pitch_mask': pitch_mask
        }
        
        return frame_data
    
    def annotate_frame(self, frame, frame_data: Dict[str, Any], 
                      pose_detector: PoseDetector, 
                      ball_tracker: BallTracker,
                      player_detector=None) -> None:
        """
        Add annotations to frame based on detection results.
        
        Args:
            frame: Frame to annotate (modified in place)
            frame_data: Detection results for the frame
            pose_detector: PoseDetector instance for drawing
            ball_tracker: BallTracker instance for drawing
        """
        # Draw pose landmarks
        if frame_data['pose_results'].pose_landmarks:
            pose_detector.draw_pose(frame, frame_data['pose_results'])
        
        # Draw ball detections
        if frame_data['ball_detections']:
            ball_tracker.draw_ball_detections(frame, frame_data['ball_detections'])
        
        # Draw player detections with enhanced team visualization
        if frame_data.get('player_detections') and player_detector:
            player_detector.draw_players(frame, frame_data['player_detections'])
            # Add team legend
            # Team legend removed for cleaner display

        # Optional: visualize pitch mask outline (subtle)
        mask = frame_data.get('pitch_mask')
        if mask is not None and mask.mean() > 0.15:
            overlay = frame.copy()
            contours, _ = cv2.findContours((mask*255).astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
    
    def add_info_overlay(self, frame, frame_data: Dict[str, Any], 
                        additional_info: Dict[str, Any] = None) -> None:
        """
        Add information overlay to frame.
        
        Args:
            frame: Frame to add overlay to
            frame_data: Frame detection data
            additional_info: Additional information to display
        """
        timestamp = frame_data['timestamp']
        frame_idx = frame_data['frame_idx']
        
        # Basic info
        info_text = [
            f"Frame: {frame_idx} | Time: {timestamp:.1f}s",
            f"Pose: {'✓' if frame_data['landmarks'] else '✗'} | "
            f"Ball: {'✓' if frame_data['ball_detections'] else '✗'}",
        ]
        
        # Add additional info if provided
        if additional_info:
            for key, value in additional_info.items():
                info_text.append(f"{key}: {value}")
        
        # Draw text with shadow effect
        for i, text in enumerate(info_text):
            y_pos = 25 + i * 20
            # Shadow
            cv2.putText(frame, text, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            # Main text
            cv2.putText(frame, text, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    def resize_frame(self, frame, target_width: int = RESIZE_WIDTH):
        """
        Resize frame maintaining aspect ratio.
        
        Args:
            frame: Input frame
            target_width: Target width for resizing
            
        Returns:
            Resized frame
        """
        h, w = frame.shape[:2]
        scale = target_width / w
        new_height = int(h * scale)
        
        return cv2.resize(frame, (target_width, new_height))

    def _compute_pitch_mask(self, frame):
        """Compute a binary mask of likely pitch (green areas)."""
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # Broad green range; can be refined
            lower = (35, 40, 40)
            upper = (85, 255, 255)
            mask = cv2.inRange(hsv, lower, upper)
            # Morphological cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            # Normalize to 0/1 float
            return (mask > 0).astype('float32')
        except Exception:
            return None
    
    def seek_to_frame(self, frame_number: int) -> bool:
        """
        Seek to specific frame in video.
        
        Args:
            frame_number: Target frame number
            
        Returns:
            True if successful, False otherwise
        """
        return self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    def get_frame(self, frame_number: Optional[int] = None):
        """
        Get a specific frame or next frame from video.
        
        Args:
            frame_number: Specific frame to get (None for next frame)
            
        Returns:
            Tuple of (success, frame)
        """
        if frame_number is not None:
            if not self.seek_to_frame(frame_number):
                return False, None
        
        return self.cap.read()
    
    def create_sample_visualization(self, frame_number: int = 50,
                                  pose_detector: PoseDetector = None,
                                  ball_tracker: BallTracker = None) -> Path:
        """
        Create a sample visualization from a specific frame.
        
        Args:
            frame_number: Frame number to visualize
            pose_detector: PoseDetector instance
            ball_tracker: BallTracker instance
            
        Returns:
            Path to saved visualization
        """
        if pose_detector is None:
            pose_detector = PoseDetector()
        if ball_tracker is None:
            ball_tracker = BallTracker()
        
        # Get frame
        success, frame = self.get_frame(frame_number)
        if not success:
            raise ValueError(f"Could not read frame {frame_number}")
        
        # Process frame
        frame_data = self.process_frame(frame, pose_detector, ball_tracker, frame_number)
        
        # Annotate frame
        self.annotate_frame(frame, frame_data, pose_detector, ball_tracker)
        self.add_info_overlay(frame, frame_data)
        
        # Save visualization
        viz_path = OUTPUT_DIR / f"detection_sample_{self.video_path.stem}.png"
        cv2.imwrite(str(viz_path), frame)
        
        print(f"💾 Visualization saved: {viz_path}")
        return viz_path
    
    def get_video_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive video statistics.
        
        Returns:
            Dictionary with video statistics
        """
        return {
            'filename': self.video_path.name,
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'frame_count': self.frame_count,
            'duration_seconds': self.frame_count / self.fps,
            'file_size_mb': self.video_path.stat().st_size / (1024 * 1024)
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
    
    def release(self):
        """Release video capture resources."""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
    
    def __del__(self):
        """Destructor to ensure resources are released."""
        self.release()