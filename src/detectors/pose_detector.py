"""
Pose detection module using MediaPipe for human pose estimation.
"""

import cv2
import mediapipe as mp
from typing import Dict, Tuple, Optional, Any
from ..config import POSE_CONFIDENCE


class PoseDetector:
    """
    Advanced pose detection using MediaPipe.
    Detects and analyzes human body pose in video frames.
    """
    
    def __init__(self, confidence: float = POSE_CONFIDENCE):
        """
        Initialize the pose detector.
        
        Args:
            confidence: Minimum detection confidence threshold
        """
        self.confidence = confidence
        
        # Initialize MediaPipe pose solution
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=confidence,
            min_tracking_confidence=confidence
        )
        
    def detect_pose(self, frame) -> Tuple[Optional[Any], Any]:
        """
        Detect pose landmarks in frame.
        
        Args:
            frame: Input video frame (BGR format)
            
        Returns:
            Tuple of (landmarks, results) where landmarks can be None if no pose detected
        """
        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if results.pose_landmarks:
            return results.pose_landmarks.landmark, results
        return None, results
    
    def get_key_points(self, landmarks) -> Dict[str, Tuple[float, float]]:
        """
        Extract key body points for football analysis.
        
        Args:
            landmarks: MediaPipe pose landmarks
            
        Returns:
            Dictionary mapping body part names to (x, y) coordinates
        """
        if landmarks is None:
            return {}
        
        # Key points for football analysis (normalized coordinates 0-1)
        key_points = {
            'nose': (landmarks[0].x, landmarks[0].y),
            'left_shoulder': (landmarks[11].x, landmarks[11].y),
            'right_shoulder': (landmarks[12].x, landmarks[12].y),
            'left_elbow': (landmarks[13].x, landmarks[13].y),
            'right_elbow': (landmarks[14].x, landmarks[14].y),
            'left_wrist': (landmarks[15].x, landmarks[15].y),
            'right_wrist': (landmarks[16].x, landmarks[16].y),
            'left_hip': (landmarks[23].x, landmarks[23].y),
            'right_hip': (landmarks[24].x, landmarks[24].y),
            'left_knee': (landmarks[25].x, landmarks[25].y),
            'right_knee': (landmarks[26].x, landmarks[26].y),
            'left_ankle': (landmarks[27].x, landmarks[27].y),
            'right_ankle': (landmarks[28].x, landmarks[28].y),
            'left_foot': (landmarks[31].x, landmarks[31].y),
            'right_foot': (landmarks[32].x, landmarks[32].y)
        }
        return key_points
    
    def draw_pose(self, frame, pose_results):
        """
        Draw pose landmarks and connections on frame.
        
        Args:
            frame: Input frame to draw on
            pose_results: MediaPipe pose detection results
        """
        if pose_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, 
                pose_results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
    
    def get_pose_quality_score(self, landmarks) -> float:
        """
        Calculate pose detection quality score.
        
        Args:
            landmarks: MediaPipe pose landmarks
            
        Returns:
            Quality score between 0-1 (1 = highest quality)
        """
        if landmarks is None:
            return 0.0
        
        # Check visibility of key landmarks
        key_landmarks = [0, 11, 12, 23, 24, 25, 26, 27, 28, 31, 32]  # Important joints
        visible_count = 0
        total_visibility = 0.0
        
        for idx in key_landmarks:
            if idx < len(landmarks):
                visibility = landmarks[idx].visibility
                total_visibility += visibility
                if visibility > 0.5:
                    visible_count += 1
        
        # Combine visibility ratio and average visibility
        visibility_ratio = visible_count / len(key_landmarks)
        avg_visibility = total_visibility / len(key_landmarks)
        
        return (visibility_ratio + avg_visibility) / 2.0
    
    def __del__(self):
        """Clean up MediaPipe resources."""
        if hasattr(self, 'pose'):
            self.pose.close()