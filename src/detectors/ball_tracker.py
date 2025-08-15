"""
Advanced ball tracking module using YOLOv11 with multi-object tracking and Kalman filtering.
Implements state-of-the-art tracking algorithms for maximum performance.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from collections import deque
import time
import math
from ultralytics import YOLO
from ..config import BALL_CONFIDENCE, BALL_CLASS_ID, YOLO_MODEL_PATH


@dataclass
class BallTrack:
    """Individual ball track with complete state information."""
    track_id: int
    positions: deque  # Last N positions
    confidences: deque  # Last N confidence scores
    velocities: deque  # Last N velocity vectors
    predicted_position: Tuple[int, int]
    last_seen_frame: int
    total_detections: int
    avg_confidence: float
    is_active: bool
    kalman_filter: Any
    age: int  # Number of frames since creation
    hits: int  # Number of successful detections
    misses: int  # Number of consecutive misses


class AdvancedKalmanFilter:
    """Advanced Kalman filter for ball tracking with velocity and acceleration."""
    
    def __init__(self, initial_pos: Tuple[int, int]):
        """Initialize Kalman filter for 2D ball tracking."""
        # State: [x, y, dx, dy, ddx, ddy] - position, velocity, acceleration
        self.state = np.array([initial_pos[0], initial_pos[1], 0, 0, 0, 0], dtype=np.float32)
        
        # State transition matrix (constant acceleration model)
        self.F = np.array([
            [1, 0, 1, 0, 0.5, 0],
            [0, 1, 0, 1, 0, 0.5],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix (we observe position only)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ], dtype=np.float32)
        
        # Process noise covariance
        self.Q = np.eye(6, dtype=np.float32) * 0.1
        
        # Measurement noise covariance
        self.R = np.eye(2, dtype=np.float32) * 1.0
        
        # Error covariance matrix
        self.P = np.eye(6, dtype=np.float32) * 10.0
    
    def predict(self) -> Tuple[int, int]:
        """Predict next position."""
        # Predict state
        self.state = self.F @ self.state
        
        # Predict error covariance
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return (int(self.state[0]), int(self.state[1]))
    
    def update(self, measurement: Tuple[int, int]) -> None:
        """Update filter with new measurement."""
        z = np.array([measurement[0], measurement[1]], dtype=np.float32)
        
        # Innovation
        y = z - self.H @ self.state
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.state = self.state + K @ y
        
        # Update error covariance
        I = np.eye(6)
        self.P = (I - K @ self.H) @ self.P
    
    def get_velocity(self) -> Tuple[float, float]:
        """Get current velocity estimate."""
        return (self.state[2], self.state[3])
    
    def get_acceleration(self) -> Tuple[float, float]:
        """Get current acceleration estimate."""
        return (self.state[4], self.state[5])


class BallTracker:
    """
    Advanced multi-object ball tracking using YOLOv11 with Kalman filtering.
    Implements state-of-the-art tracking algorithms for maximum performance.
    """
    
    def __init__(self, confidence: float = BALL_CONFIDENCE, max_tracks: int = 5):
        """
        Initialize the advanced ball tracker.
        
        Args:
            confidence: Minimum detection confidence threshold
            max_tracks: Maximum number of simultaneous ball tracks
        """
        self.confidence = confidence
        self.ball_class_id = BALL_CLASS_ID
        self.max_tracks = max_tracks
        
        # Tracking parameters
        self.max_distance_threshold = 150  # Maximum distance for association
        self.max_missed_frames = 10  # Max frames before deleting track
        self.min_hits = 3  # Minimum hits before confirming track
        self.history_length = 20  # Length of position history
        
        # Track management
        self.tracks: List[BallTrack] = []
        self.next_track_id = 0
        self.frame_count = 0
        
        # Performance optimization
        self.detection_cache = {}
        self.last_detection_time = 0
        
        # Initialize YOLOv11 model
        print("🔄 Loading Advanced YOLOv11 Ball Tracker...")
        try:
            self.model = YOLO(YOLO_MODEL_PATH)
            print("✅ Advanced YOLOv11 tracker loaded successfully!")
        except Exception as e:
            print(f"⚠️ YOLOv11 loading error: {e}")
            print("🔄 Downloading YOLOv11 model...")
            self.model = YOLO(YOLO_MODEL_PATH)
            print("✅ Advanced YOLOv11 tracker downloaded and loaded!")
    
    def detect_ball(self, frame) -> List[Dict[str, Any]]:
        """
        Advanced ball detection with multi-scale processing and confidence enhancement.
        
        Args:
            frame: Input video frame (BGR format)
            
        Returns:
            List of enhanced detection dictionaries with tracking information
        """
        # Adaptive confidence based on tracking state
        adaptive_conf = self._get_adaptive_confidence()
        
        # Multi-scale detection for better accuracy
        detections = self._multi_scale_detection(frame, adaptive_conf)
        
        # Filter and enhance detections
        detections = self._filter_detections(detections, frame.shape)
        
        # Update tracking
        tracked_detections = self._update_tracking(detections)
        
        self.frame_count += 1
        return tracked_detections
    
    def _get_adaptive_confidence(self) -> float:
        """Get adaptive confidence threshold based on tracking state."""
        if not self.tracks:
            return self.confidence
        
        # Lower confidence if we have active tracks
        active_tracks = [t for t in self.tracks if t.is_active]
        if active_tracks:
            # Reduce confidence for better detection when tracking
            return max(self.confidence * 0.7, 0.3)
        
        return self.confidence
    
    def _multi_scale_detection(self, frame, confidence: float) -> List[Dict[str, Any]]:
        """Perform multi-scale detection for better accuracy."""
        detections = []
        
        # Original scale detection
        results = self.model(frame, conf=confidence, classes=[self.ball_class_id], verbose=False)
        detections.extend(self._process_yolo_results(results))
        
        # Small scale detection (zoom in)
        h, w = frame.shape[:2]
        if min(h, w) > 800:
            center_crop = frame[h//4:3*h//4, w//4:3*w//4]
            results = self.model(center_crop, conf=confidence * 0.8, classes=[self.ball_class_id], verbose=False)
            crop_detections = self._process_yolo_results(results, offset=(w//4, h//4))
            detections.extend(crop_detections)
        
        return detections
    
    def _process_yolo_results(self, results, offset: Tuple[int, int] = (0, 0)) -> List[Dict[str, Any]]:
        """Process YOLO results into detection dictionaries."""
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    # Apply offset for cropped detections
                    x1, y1, x2, y2 = x1 + offset[0], y1 + offset[1], x2 + offset[0], y2 + offset[1]
                    
                    # Calculate center and radius
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    width = x2 - x1
                    height = y2 - y1
                    radius = int(max(width, height) / 2)
                    
                    # Calculate detection quality score
                    aspect_ratio = min(width, height) / max(width, height)
                    area = width * height
                    quality_score = confidence * aspect_ratio * min(area / 1000, 1.0)
                    
                    detection = {
                        'center': (center_x, center_y),
                        'radius': radius,
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': confidence,
                        'width': width,
                        'height': height,
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'quality_score': quality_score
                    }
                    detections.append(detection)
        
        return detections
    
    def _filter_detections(self, detections: List[Dict], frame_shape: Tuple) -> List[Dict]:
        """Filter detections using advanced criteria."""
        if not detections:
            return []
        
        h, w = frame_shape[:2]
        filtered = []
        
        for detection in detections:
            # Size filtering
            radius = detection['radius']
            if not (5 <= radius <= min(w, h) // 4):
                continue
            
            # Aspect ratio filtering (balls should be roughly circular)
            if detection['aspect_ratio'] < 0.6:
                continue
            
            # Boundary filtering
            center = detection['center']
            if not (radius <= center[0] <= w - radius and radius <= center[1] <= h - radius):
                continue
            
            # Quality threshold
            if detection['quality_score'] < 0.1:
                continue
            
            filtered.append(detection)
        
        # Non-maximum suppression
        return self._non_maximum_suppression(filtered)
    
    def _non_maximum_suppression(self, detections: List[Dict], iou_threshold: float = 0.3) -> List[Dict]:
        """Apply non-maximum suppression to remove overlapping detections."""
        if len(detections) <= 1:
            return detections
        
        # Sort by quality score
        detections = sorted(detections, key=lambda x: x['quality_score'], reverse=True)
        
        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)
            
            # Remove overlapping detections
            detections = [d for d in detections if self._calculate_iou(best, d) < iou_threshold]
        
        return keep
    
    def _calculate_iou(self, det1: Dict, det2: Dict) -> float:
        """Calculate Intersection over Union between two detections."""
        x1_1, y1_1, x2_1, y2_1 = det1['bbox']
        x1_2, y1_2, x2_2, y2_2 = det2['bbox']
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _update_tracking(self, detections: List[Dict]) -> List[Dict]:
        """Update tracking state with new detections."""
        # Predict positions for existing tracks
        for track in self.tracks:
            track.predicted_position = track.kalman_filter.predict()
            track.age += 1
        
        # Associate detections with tracks
        matches, unmatched_detections, unmatched_tracks = self._associate_detections_to_tracks(detections)
        
        # Update matched tracks
        for match in matches:
            detection_idx, track_idx = match
            detection = detections[detection_idx]
            track = self.tracks[track_idx]
            
            self._update_track(track, detection)
        
        # Handle unmatched tracks
        for track_idx in unmatched_tracks:
            track = self.tracks[track_idx]
            track.misses += 1
            if track.misses > self.max_missed_frames:
                track.is_active = False
        
        # Create new tracks for unmatched detections
        for detection_idx in unmatched_detections:
            if len(self.tracks) < self.max_tracks:
                self._create_new_track(detections[detection_idx])
        
        # Clean up inactive tracks
        self.tracks = [t for t in self.tracks if t.is_active or t.age < self.max_missed_frames * 2]
        
        # Return tracked detections
        return self._get_tracked_detections()
    
    def _associate_detections_to_tracks(self, detections: List[Dict]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Associate detections to existing tracks using Hungarian algorithm."""
        if not self.tracks or not detections:
            return [], list(range(len(detections))), list(range(len(self.tracks)))
        
        # Calculate distance matrix
        cost_matrix = []
        for track in self.tracks:
            track_costs = []
            for detection in detections:
                distance = self._calculate_distance(track.predicted_position, detection['center'])
                if distance < self.max_distance_threshold:
                    # Factor in confidence and track quality
                    cost = distance / (detection['confidence'] * track.avg_confidence + 0.1)
                else:
                    cost = float('inf')
                track_costs.append(cost)
            cost_matrix.append(track_costs)
        
        # Simple greedy assignment (can be replaced with Hungarian algorithm)
        matches = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))
        
        cost_matrix = np.array(cost_matrix)
        
        while len(unmatched_detections) > 0 and len(unmatched_tracks) > 0:
            # Find minimum cost
            min_cost = float('inf')
            min_track_idx = -1
            min_detection_idx = -1
            
            for track_idx in unmatched_tracks:
                for detection_idx in unmatched_detections:
                    if cost_matrix[track_idx][detection_idx] < min_cost:
                        min_cost = cost_matrix[track_idx][detection_idx]
                        min_track_idx = track_idx
                        min_detection_idx = detection_idx
            
            if min_cost == float('inf'):
                break
            
            matches.append((min_detection_idx, min_track_idx))
            unmatched_detections.remove(min_detection_idx)
            unmatched_tracks.remove(min_track_idx)
        
        return matches, unmatched_detections, unmatched_tracks
    
    def _calculate_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two positions."""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _update_track(self, track: BallTrack, detection: Dict) -> None:
        """Update track with new detection."""
        position = detection['center']
        confidence = detection['confidence']
        
        # Update Kalman filter
        track.kalman_filter.update(position)
        
        # Update track data
        track.positions.append(position)
        track.confidences.append(confidence)
        track.last_seen_frame = self.frame_count
        track.total_detections += 1
        track.hits += 1
        track.misses = 0
        track.is_active = True
        
        # Calculate average confidence
        track.avg_confidence = sum(track.confidences) / len(track.confidences)
        
        # Calculate velocities
        if len(track.positions) >= 2:
            last_pos = track.positions[-2]
            velocity = (position[0] - last_pos[0], position[1] - last_pos[1])
            track.velocities.append(velocity)
        
        # Maintain history length
        if len(track.positions) > self.history_length:
            track.positions.popleft()
            track.confidences.popleft()
        if len(track.velocities) > self.history_length:
            track.velocities.popleft()
    
    def _create_new_track(self, detection: Dict) -> None:
        """Create new track from detection."""
        position = detection['center']
        confidence = detection['confidence']
        
        # Create Kalman filter
        kalman_filter = AdvancedKalmanFilter(position)
        
        # Create track
        track = BallTrack(
            track_id=self.next_track_id,
            positions=deque([position], maxlen=self.history_length),
            confidences=deque([confidence], maxlen=self.history_length),
            velocities=deque(maxlen=self.history_length),
            predicted_position=position,
            last_seen_frame=self.frame_count,
            total_detections=1,
            avg_confidence=confidence,
            is_active=True,
            kalman_filter=kalman_filter,
            age=0,
            hits=1,
            misses=0
        )
        
        self.tracks.append(track)
        self.next_track_id += 1
    
    def _get_tracked_detections(self) -> List[Dict]:
        """Get current detections from active tracks."""
        detections = []
        
        for track in self.tracks:
            if track.is_active and track.hits >= self.min_hits:
                # Get current position (Kalman filtered)
                if track.positions:
                    position = track.positions[-1]
                else:
                    position = track.predicted_position
                
                # Estimate radius from track history
                radius = 15  # Default radius
                if len(track.positions) >= 2:
                    # Use recent positions to estimate size
                    radius = max(10, min(50, int(track.avg_confidence * 30)))
                
                detection = {
                    'center': position,
                    'radius': radius,
                    'bbox': (position[0] - radius, position[1] - radius, 
                            position[0] + radius, position[1] + radius),
                    'confidence': track.avg_confidence,
                    'track_id': track.track_id,
                    'velocity': track.kalman_filter.get_velocity(),
                    'acceleration': track.kalman_filter.get_acceleration(),
                    'track_age': track.age,
                    'track_hits': track.hits,
                    'is_tracked': True
                }
                detections.append(detection)
        
        return detections
    
    def draw_ball_detections(self, frame, detections: List[Dict], 
                            draw_tracks: bool = True, draw_predictions: bool = True):
        """
        Advanced drawing with tracking information and predictions.
        
        Args:
            frame: Input frame to draw on
            detections: List of ball detection dictionaries
            draw_tracks: Whether to draw tracking trails
            draw_predictions: Whether to draw predicted positions
        """
        for detection in detections:
            center = detection['center']
            radius = detection['radius']
            confidence = detection['confidence']
            is_tracked = detection.get('is_tracked', False)
            
            # Choose color based on tracking status
            if is_tracked:
                color = (0, 255, 0)  # Green for tracked balls
                track_id = detection.get('track_id', -1)
            else:
                color = (0, 255, 255)  # Yellow for new detections
                track_id = -1
            
            # Draw ball circle and center dot
            cv2.circle(frame, center, radius, color, 3)
            cv2.circle(frame, center, 5, color, -1)
            
            # Draw tracking information
            info_lines = [f'Ball {track_id}' if is_tracked else 'New Ball']
            info_lines.append(f'Conf: {confidence:.2f}')
            
            if is_tracked:
                velocity = detection.get('velocity', (0, 0))
                speed = math.sqrt(velocity[0]**2 + velocity[1]**2)
                info_lines.append(f'Speed: {speed:.1f}')
                
                # Draw velocity vector
                if speed > 2:
                    end_point = (
                        int(center[0] + velocity[0] * 3),
                        int(center[1] + velocity[1] * 3)
                    )
                    cv2.arrowedLine(frame, center, end_point, color, 2)
            
            # Draw text information
            for i, line in enumerate(info_lines):
                y_offset = -radius - 15 - (i * 15)
                cv2.putText(
                    frame, line,
                    (center[0] - 40, center[1] + y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
                )
            
            # Draw tracking trail
            if draw_tracks and is_tracked:
                track = self._get_track_by_id(detection.get('track_id', -1))
                if track and len(track.positions) > 1:
                    points = list(track.positions)
                    for i in range(1, len(points)):
                        alpha = i / len(points)
                        trail_color = (int(color[0] * alpha), int(color[1] * alpha), int(color[2] * alpha))
                        cv2.line(frame, points[i-1], points[i], trail_color, 2)
            
            # Draw prediction
            if draw_predictions and is_tracked:
                track = self._get_track_by_id(detection.get('track_id', -1))
                if track:
                    pred_pos = track.predicted_position
                    cv2.circle(frame, pred_pos, 8, (255, 0, 255), 2)  # Magenta for prediction
    
    def _get_track_by_id(self, track_id: int) -> Optional[BallTrack]:
        """Get track by ID."""
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        return None
    
    def get_best_detection(self, detections: List[Dict]) -> Dict[str, Any]:
        """
        Get the best detection (highest confidence tracked ball or best untracked).
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Best detection dictionary, or empty dict if no detections
        """
        if not detections:
            return {}
        
        # Prefer tracked detections
        tracked = [d for d in detections if d.get('is_tracked', False)]
        if tracked:
            return max(tracked, key=lambda x: x['confidence'])
        
        return max(detections, key=lambda x: x['confidence'])
    
    def get_trajectory_for_track(self, track_id: int) -> List[Tuple[int, int]]:
        """
        Get trajectory for a specific track.
        
        Args:
            track_id: ID of the track
            
        Returns:
            List of (x, y) coordinates for the track
        """
        track = self._get_track_by_id(track_id)
        if track:
            return list(track.positions)
        return []
    
    def get_all_trajectories(self) -> Dict[int, List[Tuple[int, int]]]:
        """Get trajectories for all active tracks."""
        trajectories = {}
        for track in self.tracks:
            if track.is_active and len(track.positions) > 1:
                trajectories[track.track_id] = list(track.positions)
        return trajectories
    
    def predict_future_positions(self, track_id: int, frames_ahead: int = 5) -> List[Tuple[int, int]]:
        """
        Predict future positions for a track.
        
        Args:
            track_id: ID of the track
            frames_ahead: Number of frames to predict
            
        Returns:
            List of predicted (x, y) coordinates
        """
        track = self._get_track_by_id(track_id)
        if not track:
            return []
        
        predictions = []
        # Create temporary Kalman filter state
        temp_state = track.kalman_filter.state.copy()
        temp_P = track.kalman_filter.P.copy()
        
        for _ in range(frames_ahead):
            # Predict next state
            temp_state = track.kalman_filter.F @ temp_state
            temp_P = track.kalman_filter.F @ temp_P @ track.kalman_filter.F.T + track.kalman_filter.Q
            
            prediction = (int(temp_state[0]), int(temp_state[1]))
            predictions.append(prediction)
        
        return predictions
    
    def get_track_statistics(self) -> Dict[str, Any]:
        """Get comprehensive tracking statistics."""
        active_tracks = [t for t in self.tracks if t.is_active]
        
        stats = {
            'total_tracks': len(self.tracks),
            'active_tracks': len(active_tracks),
            'frame_count': self.frame_count,
            'detection_rate': len(active_tracks) / max(1, self.frame_count),
            'tracks': []
        }
        
        for track in active_tracks:
            if len(track.positions) > 1:
                # Calculate track statistics
                positions = list(track.positions)
                velocities = list(track.velocities) if track.velocities else []
                
                track_stats = {
                    'track_id': track.track_id,
                    'age': track.age,
                    'hits': track.hits,
                    'avg_confidence': track.avg_confidence,
                    'position_count': len(positions),
                    'current_position': positions[-1] if positions else None,
                    'avg_speed': np.mean([math.sqrt(v[0]**2 + v[1]**2) for v in velocities]) if velocities else 0,
                    'trajectory_length': self._calculate_trajectory_length(positions)
                }
                stats['tracks'].append(track_stats)
        
        return stats
    
    def _calculate_trajectory_length(self, positions: List[Tuple[int, int]]) -> float:
        """Calculate total length of trajectory."""
        if len(positions) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            total_length += math.sqrt(dx*dx + dy*dy)
        
        return total_length
    
    def reset_tracking(self):
        """Reset all tracking state."""
        self.tracks.clear()
        self.next_track_id = 0
        self.frame_count = 0
        self.detection_cache.clear()
        print("🔄 Ball tracking reset - all tracks cleared")
    
    def smooth_trajectory(self, trajectory: List[Tuple[int, int]], 
                         window_size: int = 5, method: str = 'gaussian') -> List[Tuple[int, int]]:
        """
        Advanced trajectory smoothing with multiple methods.
        
        Args:
            trajectory: List of (x, y) coordinates
            window_size: Size of smoothing window
            method: Smoothing method ('moving_average', 'gaussian', 'kalman')
            
        Returns:
            Smoothed trajectory
        """
        if len(trajectory) < window_size:
            return trajectory
        
        if method == 'gaussian':
            return self._gaussian_smooth(trajectory, window_size)
        elif method == 'kalman':
            return self._kalman_smooth(trajectory)
        else:  # moving_average
            return self._moving_average_smooth(trajectory, window_size)
    
    def _moving_average_smooth(self, trajectory: List[Tuple[int, int]], window_size: int) -> List[Tuple[int, int]]:
        """Apply moving average smoothing."""
        smoothed = []
        for i in range(len(trajectory)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(trajectory), i + window_size // 2 + 1)
            
            x_sum = sum(pos[0] for pos in trajectory[start_idx:end_idx])
            y_sum = sum(pos[1] for pos in trajectory[start_idx:end_idx])
            count = end_idx - start_idx
            
            smoothed.append((int(x_sum / count), int(y_sum / count)))
        
        return smoothed
    
    def _gaussian_smooth(self, trajectory: List[Tuple[int, int]], window_size: int) -> List[Tuple[int, int]]:
        """Apply Gaussian smoothing."""
        if len(trajectory) < 3:
            return trajectory
        
        # Create Gaussian kernel
        sigma = window_size / 3.0
        kernel_size = window_size
        kernel = np.array([math.exp(-(x - kernel_size//2)**2 / (2*sigma**2)) 
                          for x in range(kernel_size)])
        kernel = kernel / np.sum(kernel)
        
        # Apply smoothing
        x_coords = [p[0] for p in trajectory]
        y_coords = [p[1] for p in trajectory]
        
        # Pad arrays
        pad_size = kernel_size // 2
        x_padded = [x_coords[0]] * pad_size + x_coords + [x_coords[-1]] * pad_size
        y_padded = [y_coords[0]] * pad_size + y_coords + [y_coords[-1]] * pad_size
        
        smoothed_x = np.convolve(x_padded, kernel, mode='valid')
        smoothed_y = np.convolve(y_padded, kernel, mode='valid')
        
        return [(int(x), int(y)) for x, y in zip(smoothed_x, smoothed_y)]
    
    def _kalman_smooth(self, trajectory: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Apply Kalman filter smoothing."""
        if len(trajectory) < 2:
            return trajectory
        
        # Create temporary Kalman filter for smoothing
        kf = AdvancedKalmanFilter(trajectory[0])
        smoothed = [trajectory[0]]
        
        for i in range(1, len(trajectory)):
            kf.predict()
            kf.update(trajectory[i])
            smoothed_pos = (int(kf.state[0]), int(kf.state[1]))
            smoothed.append(smoothed_pos)
        
        return smoothed