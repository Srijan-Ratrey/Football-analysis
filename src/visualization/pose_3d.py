"""
3D Pose Reconstruction and Visualization Module
FIFA-quality 3D analysis for football player evaluation.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from typing import List, Dict, Any, Tuple, Optional
import math

# MediaPipe pose landmark indices
POSE_CONNECTIONS = [
    # Head and neck
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    # Arms
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    # Torso
    (11, 23), (12, 24), (23, 24),
    # Legs
    (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32),
    (27, 31), (28, 32)
]

class Pose3DReconstructor:
    """FIFA-quality 3D pose reconstruction from MediaPipe landmarks."""
    
    def __init__(self):
        """Initialize 3D pose reconstructor."""
        self.field_length = 105.0  # FIFA standard field length (meters)
        self.field_width = 68.0    # FIFA standard field width (meters)
        self.player_height = 1.75  # Average player height (meters)
        
        # Camera calibration parameters (estimated)
        self.focal_length = 1000
        self.camera_height = 10.0  # Estimated camera height (meters)
        
        # 3D pose history for smoothing
        self.pose_history = {}
        self.smoothing_factor = 0.7
        
    def reconstruct_3d_pose(self, landmarks_2d: np.ndarray, frame_shape: Tuple[int, int]) -> np.ndarray:
        """
        Reconstruct 3D pose from 2D MediaPipe landmarks.
        
        Args:
            landmarks_2d: 2D pose landmarks (33 x 2)
            frame_shape: (height, width) of input frame
            
        Returns:
            3D pose landmarks (33 x 3)
        """
        if landmarks_2d is None or len(landmarks_2d) == 0:
            return None
            
        height, width = frame_shape
        landmarks_3d = np.zeros((len(landmarks_2d), 3))
        
        # Use hip midpoint as reference for depth estimation
        if len(landmarks_2d) >= 24:  # Ensure we have hip landmarks
            left_hip = landmarks_2d[23]
            right_hip = landmarks_2d[24]
            hip_center = (left_hip + right_hip) / 2
            
            # Estimate depth using perspective projection
            # Assume hip is at ground level (z=0) and scale accordingly
            reference_hip_width = 0.3  # Meters
            pixel_hip_width = np.linalg.norm(right_hip - left_hip)
            
            if pixel_hip_width > 0:
                depth_scale = (reference_hip_width * self.focal_length) / pixel_hip_width
            else:
                depth_scale = 10.0  # Default depth
                
            # Convert 2D to 3D
            for i, landmark_2d in enumerate(landmarks_2d):
                # Convert pixel coordinates to normalized coordinates
                x_norm = (landmark_2d[0] - width/2) / width
                y_norm = (landmark_2d[1] - height/2) / height
                
                # Estimate 3D coordinates
                x_3d = x_norm * depth_scale
                y_3d = -y_norm * depth_scale  # Flip Y for correct orientation
                z_3d = self._estimate_landmark_height(i, landmarks_2d, hip_center)
                
                landmarks_3d[i] = [x_3d, y_3d, z_3d]
                
        return landmarks_3d
    
    def _estimate_landmark_height(self, landmark_idx: int, landmarks_2d: np.ndarray, 
                                 hip_center: np.ndarray) -> float:
        """Estimate height of landmark relative to ground."""
        landmark_heights = {
            # Head landmarks
            0: 1.65, 1: 1.70, 2: 1.68, 3: 1.66, 4: 1.66, 5: 1.64, 6: 1.62, 7: 1.64, 8: 1.62,
            9: 1.68, 10: 1.68,
            # Shoulder and arm landmarks
            11: 1.45, 12: 1.45, 13: 1.35, 14: 1.35, 15: 1.25, 16: 1.25,
            17: 1.20, 18: 1.20, 19: 1.15, 20: 1.15, 21: 1.10, 22: 1.10,
            # Hip landmarks
            23: 0.90, 24: 0.90,
            # Leg landmarks
            25: 0.50, 26: 0.50, 27: 0.05, 28: 0.05, 29: 0.0, 30: 0.0, 31: 0.0, 32: 0.0
        }
        
        return landmark_heights.get(landmark_idx, 1.0)
    
    def analyze_body_mechanics(self, pose_3d: np.ndarray, ball_position: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        FIFA-quality biomechanical analysis.
        
        Args:
            pose_3d: 3D pose landmarks
            ball_position: 3D ball position if available
            
        Returns:
            Biomechanical analysis results
        """
        if pose_3d is None:
            return {}
            
        analysis = {}
        
        # Body lean analysis
        analysis['body_lean'] = self._calculate_body_lean(pose_3d)
        
        # Hip orientation and torque
        analysis['hip_torque'] = self._calculate_hip_torque(pose_3d)
        
        # Shooting mechanics (if ball is detected)
        if ball_position is not None:
            analysis['shooting_mechanics'] = self._analyze_shooting_mechanics(pose_3d, ball_position)
        
        # Balance and stability
        analysis['balance'] = self._analyze_balance(pose_3d)
        
        # Joint angles
        analysis['joint_angles'] = self._calculate_joint_angles(pose_3d)
        
        return analysis
    
    def _calculate_body_lean(self, pose_3d: np.ndarray) -> Dict[str, float]:
        """Calculate body lean and posture."""
        if len(pose_3d) < 25:
            return {'lean_angle': 0.0, 'lean_direction': 'neutral'}
            
        # Use shoulder and hip landmarks
        left_shoulder = pose_3d[11]
        right_shoulder = pose_3d[12]
        left_hip = pose_3d[23]
        right_hip = pose_3d[24]
        
        # Calculate torso vector
        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2
        torso_vector = shoulder_center - hip_center
        
        # Calculate lean angle from vertical
        vertical = np.array([0, 0, 1])
        torso_normalized = torso_vector / (np.linalg.norm(torso_vector) + 1e-6)
        
        lean_angle = np.arccos(np.clip(np.dot(torso_normalized, vertical), -1.0, 1.0))
        lean_angle_degrees = np.degrees(lean_angle)
        
        # Determine lean direction
        if abs(torso_vector[0]) > abs(torso_vector[1]):
            lean_direction = 'left' if torso_vector[0] < 0 else 'right'
        else:
            lean_direction = 'forward' if torso_vector[1] > 0 else 'backward'
            
        return {
            'lean_angle': lean_angle_degrees,
            'lean_direction': lean_direction,
            'torso_vector': torso_vector.tolist()
        }
    
    def _calculate_hip_torque(self, pose_3d: np.ndarray) -> Dict[str, float]:
        """Calculate hip rotation and torque."""
        if len(pose_3d) < 25:
            return {'hip_rotation': 0.0}
            
        left_hip = pose_3d[23]
        right_hip = pose_3d[24]
        
        # Hip orientation vector
        hip_vector = right_hip - left_hip
        
        # Calculate rotation from frontal plane
        frontal_plane_normal = np.array([0, 1, 0])
        hip_normalized = hip_vector / (np.linalg.norm(hip_vector) + 1e-6)
        
        # Project hip vector onto horizontal plane
        hip_horizontal = hip_vector.copy()
        hip_horizontal[2] = 0  # Remove vertical component
        hip_horizontal = hip_horizontal / (np.linalg.norm(hip_horizontal) + 1e-6)
        
        # Calculate rotation angle
        reference = np.array([1, 0, 0])  # Forward direction
        rotation_angle = np.arccos(np.clip(np.dot(hip_horizontal, reference), -1.0, 1.0))
        rotation_degrees = np.degrees(rotation_angle)
        
        return {
            'hip_rotation': rotation_degrees,
            'hip_vector': hip_vector.tolist()
        }
    
    def _analyze_shooting_mechanics(self, pose_3d: np.ndarray, ball_position: np.ndarray) -> Dict[str, Any]:
        """Analyze shooting mechanics and contact timing."""
        if len(pose_3d) < 33:
            return {}
            
        # Get foot positions
        left_foot = pose_3d[31] if len(pose_3d) > 31 else None
        right_foot = pose_3d[32] if len(pose_3d) > 32 else None
        
        analysis = {}
        
        # Determine shooting foot (closest to ball)
        if left_foot is not None and right_foot is not None:
            left_dist = np.linalg.norm(left_foot - ball_position)
            right_dist = np.linalg.norm(right_foot - ball_position)
            
            shooting_foot = 'left' if left_dist < right_dist else 'right'
            contact_distance = min(left_dist, right_dist)
            
            analysis['shooting_foot'] = shooting_foot
            analysis['contact_distance'] = contact_distance
            analysis['contact_imminent'] = contact_distance < 0.3  # 30cm threshold
            
            # Analyze shooting posture
            if shooting_foot == 'right':
                shooting_foot_pos = right_foot
                supporting_foot_pos = left_foot
            else:
                shooting_foot_pos = left_foot
                supporting_foot_pos = right_foot
                
            # Calculate approach angle
            foot_to_ball = ball_position - shooting_foot_pos
            approach_angle = np.arctan2(foot_to_ball[1], foot_to_ball[0])
            
            analysis['approach_angle'] = np.degrees(approach_angle)
            analysis['foot_position'] = shooting_foot_pos.tolist()
            
        return analysis
    
    def _analyze_balance(self, pose_3d: np.ndarray) -> Dict[str, float]:
        """Analyze player balance and stability."""
        if len(pose_3d) < 33:
            return {'balance_score': 0.5}
            
        # Get key landmarks
        left_foot = pose_3d[31] if len(pose_3d) > 31 else None
        right_foot = pose_3d[32] if len(pose_3d) > 32 else None
        left_hip = pose_3d[23]
        right_hip = pose_3d[24]
        
        if left_foot is None or right_foot is None:
            return {'balance_score': 0.5}
            
        # Calculate center of gravity (simplified)
        hip_center = (left_hip + right_hip) / 2
        foot_center = (left_foot + right_foot) / 2
        
        # Balance score based on alignment
        horizontal_offset = np.linalg.norm(hip_center[:2] - foot_center[:2])
        foot_separation = np.linalg.norm(left_foot - right_foot)
        
        # Good balance: hips over feet, appropriate foot separation
        ideal_foot_separation = 0.4  # 40cm
        balance_score = max(0.0, 1.0 - (horizontal_offset / 0.5) - abs(foot_separation - ideal_foot_separation) / 0.3)
        
        return {
            'balance_score': balance_score,
            'horizontal_offset': horizontal_offset,
            'foot_separation': foot_separation
        }
    
    def _calculate_joint_angles(self, pose_3d: np.ndarray) -> Dict[str, float]:
        """Calculate key joint angles for biomechanical analysis."""
        if len(pose_3d) < 33:
            return {}
            
        angles = {}
        
        # Knee angles
        if len(pose_3d) > 27:
            # Left knee angle
            left_hip = pose_3d[23]
            left_knee = pose_3d[25]
            left_ankle = pose_3d[27]
            angles['left_knee'] = self._calculate_angle(left_hip, left_knee, left_ankle)
            
            # Right knee angle
            right_hip = pose_3d[24]
            right_knee = pose_3d[26]
            right_ankle = pose_3d[28]
            angles['right_knee'] = self._calculate_angle(right_hip, right_knee, right_ankle)
        
        # Elbow angles
        if len(pose_3d) > 16:
            # Left elbow
            left_shoulder = pose_3d[11]
            left_elbow = pose_3d[13]
            left_wrist = pose_3d[15]
            angles['left_elbow'] = self._calculate_angle(left_shoulder, left_elbow, left_wrist)
            
            # Right elbow
            right_shoulder = pose_3d[12]
            right_elbow = pose_3d[14]
            right_wrist = pose_3d[16]
            angles['right_elbow'] = self._calculate_angle(right_shoulder, right_elbow, right_wrist)
        
        return angles
    
    def _calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle between three 3D points."""
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        return np.degrees(angle)

class Field3DMapper:
    """3D Football field mapping and coordinate transformation."""
    
    def __init__(self):
        """Initialize 3D field mapper."""
        self.field_length = 105.0  # FIFA standard
        self.field_width = 68.0    # FIFA standard
        
        # Field markings (FIFA standard)
        self.penalty_area_length = 16.5
        self.penalty_area_width = 40.3
        self.goal_area_length = 5.5
        self.goal_area_width = 18.3
        self.center_circle_radius = 9.15
        
    def create_3d_field(self) -> Dict[str, np.ndarray]:
        """Create 3D field model with FIFA standard markings."""
        field_data = {}
        
        # Field boundary
        field_data['boundary'] = self._create_field_boundary()
        
        # Center line and circle
        field_data['center_line'] = self._create_center_line()
        field_data['center_circle'] = self._create_center_circle()
        
        # Penalty areas
        field_data['penalty_areas'] = self._create_penalty_areas()
        
        # Goal areas
        field_data['goal_areas'] = self._create_goal_areas()
        
        # Goals
        field_data['goals'] = self._create_goals()
        
        return field_data
    
    def _create_field_boundary(self) -> np.ndarray:
        """Create field boundary points."""
        boundary = np.array([
            [-self.field_length/2, -self.field_width/2, 0],
            [self.field_length/2, -self.field_width/2, 0],
            [self.field_length/2, self.field_width/2, 0],
            [-self.field_length/2, self.field_width/2, 0],
            [-self.field_length/2, -self.field_width/2, 0]  # Close the rectangle
        ])
        return boundary
    
    def _create_center_line(self) -> np.ndarray:
        """Create center line."""
        return np.array([
            [0, -self.field_width/2, 0],
            [0, self.field_width/2, 0]
        ])
    
    def _create_center_circle(self) -> np.ndarray:
        """Create center circle."""
        angles = np.linspace(0, 2*np.pi, 64)
        x_coords = np.zeros_like(angles)
        y_coords = self.center_circle_radius * np.cos(angles)
        z_coords = np.zeros_like(angles)  # Circle on ground (z=0)
        
        circle = np.column_stack([x_coords, y_coords, z_coords])
        return circle
    
    def _create_penalty_areas(self) -> List[np.ndarray]:
        """Create penalty areas."""
        penalty_areas = []
        
        # Left penalty area
        left_penalty = np.array([
            [-self.field_length/2, -self.penalty_area_width/2, 0],
            [-self.field_length/2 + self.penalty_area_length, -self.penalty_area_width/2, 0],
            [-self.field_length/2 + self.penalty_area_length, self.penalty_area_width/2, 0],
            [-self.field_length/2, self.penalty_area_width/2, 0],
            [-self.field_length/2, -self.penalty_area_width/2, 0]
        ])
        penalty_areas.append(left_penalty)
        
        # Right penalty area
        right_penalty = np.array([
            [self.field_length/2, -self.penalty_area_width/2, 0],
            [self.field_length/2 - self.penalty_area_length, -self.penalty_area_width/2, 0],
            [self.field_length/2 - self.penalty_area_length, self.penalty_area_width/2, 0],
            [self.field_length/2, self.penalty_area_width/2, 0],
            [self.field_length/2, -self.penalty_area_width/2, 0]
        ])
        penalty_areas.append(right_penalty)
        
        return penalty_areas
    
    def _create_goal_areas(self) -> List[np.ndarray]:
        """Create goal areas."""
        goal_areas = []
        
        # Left goal area
        left_goal = np.array([
            [-self.field_length/2, -self.goal_area_width/2, 0],
            [-self.field_length/2 + self.goal_area_length, -self.goal_area_width/2, 0],
            [-self.field_length/2 + self.goal_area_length, self.goal_area_width/2, 0],
            [-self.field_length/2, self.goal_area_width/2, 0],
            [-self.field_length/2, -self.goal_area_width/2, 0]
        ])
        goal_areas.append(left_goal)
        
        # Right goal area
        right_goal = np.array([
            [self.field_length/2, -self.goal_area_width/2, 0],
            [self.field_length/2 - self.goal_area_length, -self.goal_area_width/2, 0],
            [self.field_length/2 - self.goal_area_length, self.goal_area_width/2, 0],
            [self.field_length/2, self.goal_area_width/2, 0],
            [self.field_length/2, -self.goal_area_width/2, 0]
        ])
        goal_areas.append(right_goal)
        
        return goal_areas
    
    def _create_goals(self) -> List[np.ndarray]:
        """Create goal posts."""
        goal_width = 7.32  # FIFA standard
        goal_height = 2.44  # FIFA standard
        
        goals = []
        
        # Left goal
        left_goal = np.array([
            [-self.field_length/2, -goal_width/2, 0],
            [-self.field_length/2, -goal_width/2, goal_height],
            [-self.field_length/2, goal_width/2, goal_height],
            [-self.field_length/2, goal_width/2, 0]
        ])
        goals.append(left_goal)
        
        # Right goal
        right_goal = np.array([
            [self.field_length/2, -goal_width/2, 0],
            [self.field_length/2, -goal_width/2, goal_height],
            [self.field_length/2, goal_width/2, goal_height],
            [self.field_length/2, goal_width/2, 0]
        ])
        goals.append(right_goal)
        
        return goals
    
    def map_player_to_field(self, player_2d_pos: np.ndarray, frame_shape: Tuple[int, int]) -> np.ndarray:
        """Map 2D player position to 3D field coordinates."""
        height, width = frame_shape
        
        # Normalize pixel coordinates
        x_norm = (player_2d_pos[0] - width/2) / width
        y_norm = (player_2d_pos[1] - height/2) / height
        
        # Map to field coordinates (simplified perspective)
        field_x = x_norm * self.field_length * 0.8  # Scale factor
        field_y = y_norm * self.field_width * 0.8
        field_z = 0  # On ground level
        
        return np.array([field_x, field_y, field_z])