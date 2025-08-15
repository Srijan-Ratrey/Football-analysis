"""
Ball control analysis module for performance metrics and coaching insights.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from ..config import TOUCH_THRESHOLD, TOUCH_DEBOUNCE_FRAMES, MIN_TOUCH_DURATION, FEEDBACK_MESSAGES


class BallControlAnalyzer:
    """
    Analyzes ball control patterns and player performance.
    Generates coaching insights and performance metrics.
    """
    
    def __init__(self, touch_threshold: int = TOUCH_THRESHOLD):
        """
        Initialize the ball control analyzer.
        
        Args:
            touch_threshold: Distance threshold for ball-foot contact in pixels
        """
        self.touch_threshold = touch_threshold
        self.touches_history = []
        self.last_touch_frame = {}  # Track last touch frame for each foot to prevent duplicates
        self.continuous_touches = {}  # Track continuous touch sequences
        self.performance_metrics = {
            'total_touches': 0,
            'foot_contacts': {'left': 0, 'right': 0},
            'touch_quality_scores': [],
            'touch_intervals': []
        }
    
    def detect_ball_touches(self, pose_data: List[Dict], 
                           ball_data: List[Dict]) -> List[Dict[str, Any]]:
        """
        Detect when player touches the ball based on pose and ball data with proper debouncing.
        
        Args:
            pose_data: List of pose detection data for each frame
            ball_data: List of ball detection data for each frame
            
        Returns:
            List of touch events with timestamps and details
        """
        touches = []
        
        for pose_frame in pose_data:
            frame_idx = pose_frame['frame_idx']
            key_points = pose_frame['key_points']
            
            # Find ball position at this frame
            ball_at_frame = [b for b in ball_data if b['frame_idx'] == frame_idx]
            
            if ball_at_frame and key_points:
                ball_pos = ball_at_frame[0]['center']
                
                # Check distance to both feet
                for foot_name in ['left_foot', 'right_foot']:
                    if foot_name in key_points:
                        foot_pos = key_points[foot_name]
                        foot_pixel = (
                            int(foot_pos[0] * pose_frame['frame_width']),
                            int(foot_pos[1] * pose_frame['frame_height'])
                        )
                        
                        distance = np.sqrt(
                            (ball_pos[0] - foot_pixel[0])**2 + 
                            (ball_pos[1] - foot_pixel[1])**2
                        )
                        
                        # Check if this is a valid touch (within threshold)
                        is_within_threshold = distance < self.touch_threshold
                        
                        # Track continuous touches for this foot
                        if foot_name not in self.continuous_touches:
                            self.continuous_touches[foot_name] = {
                                'active': False,
                                'start_frame': None,
                                'frames_count': 0,
                                'min_distance': float('inf')
                            }
                        
                        touch_data = self.continuous_touches[foot_name]
                        
                        if is_within_threshold:
                            if not touch_data['active']:
                                # Start new touch sequence
                                touch_data['active'] = True
                                touch_data['start_frame'] = frame_idx
                                touch_data['frames_count'] = 1
                                touch_data['min_distance'] = distance
                            else:
                                # Continue existing touch sequence
                                touch_data['frames_count'] += 1
                                touch_data['min_distance'] = min(touch_data['min_distance'], distance)
                        else:
                            if touch_data['active']:
                                # End touch sequence - validate it
                                if self._is_valid_touch(touch_data, frame_idx):
                                    # Register as a valid touch
                                    valid_touch = self._create_touch_event(
                                        pose_frame, foot_name, ball_pos, foot_pixel,
                                        touch_data['min_distance'], touch_data['start_frame']
                                    )
                                    touches.append(valid_touch)
                                    
                                    # Update metrics only for valid touches
                                    self.performance_metrics['total_touches'] += 1
                                    foot_side = foot_name.split('_')[0]
                                    self.performance_metrics['foot_contacts'][foot_side] += 1
                                    self.performance_metrics['touch_quality_scores'].append(
                                        valid_touch['quality_score']
                                    )
                                
                                # Reset touch data
                                touch_data['active'] = False
                                touch_data['start_frame'] = None
                                touch_data['frames_count'] = 0
                                touch_data['min_distance'] = float('inf')
        
        # Calculate touch intervals for valid touches
        if len(touches) > 1:
            for i in range(1, len(touches)):
                interval = touches[i]['timestamp'] - touches[i-1]['timestamp']
                self.performance_metrics['touch_intervals'].append(interval)
        
        self.touches_history.extend(touches)
        return touches
    
    def _is_valid_touch(self, touch_data: Dict, current_frame: int) -> bool:
        """Check if a touch sequence is valid based on duration and debouncing."""
        # Must have minimum duration
        if touch_data['frames_count'] < MIN_TOUCH_DURATION:
            return False
        
        # Check debouncing - ensure enough frames since last touch of this foot
        foot_key = f"last_touch_{id(touch_data)}"
        if foot_key in self.last_touch_frame:
            frames_since_last = current_frame - self.last_touch_frame[foot_key]
            if frames_since_last < TOUCH_DEBOUNCE_FRAMES:
                return False
        
        # Update last touch frame
        self.last_touch_frame[foot_key] = current_frame
        return True
    
    def _create_touch_event(self, pose_frame: Dict, foot_name: str, ball_pos: tuple,
                           foot_pixel: tuple, min_distance: float, start_frame: int) -> Dict[str, Any]:
        """Create a touch event object."""
        return {
            'timestamp': pose_frame['timestamp'],
            'foot': foot_name,
            'ball_position': ball_pos,
            'foot_position': foot_pixel,
            'distance': min_distance,
            'frame_idx': start_frame,  # Use start frame of touch sequence
            'quality_score': self._calculate_touch_quality(min_distance)
        }
    
    def _calculate_touch_quality(self, distance: float) -> float:
        """
        Calculate quality score for a ball touch based on distance.
        
        Args:
            distance: Distance between ball and foot in pixels
            
        Returns:
            Quality score between 0-1 (1 = perfect touch)
        """
        # Linear quality score: closer = better
        max_quality_distance = 20  # Very close touch
        
        if distance <= max_quality_distance:
            return 1.0
        elif distance >= self.touch_threshold:
            return 0.0
        else:
            # Linear interpolation between perfect and threshold
            return 1.0 - (distance - max_quality_distance) / (self.touch_threshold - max_quality_distance)
    
    def analyze_dribbling_pattern(self, ball_trajectory: List[tuple], 
                                 touches: List[Dict]) -> Optional[Dict[str, Any]]:
        """
        Analyze dribbling patterns and ball control consistency.
        
        Args:
            ball_trajectory: List of ball positions over time
            touches: List of ball touch events
            
        Returns:
            Analysis results dictionary or None if insufficient data
        """
        if len(touches) < 2:
            return None
        
        total_touches = len(touches)
        time_span = touches[-1]['timestamp'] - touches[0]['timestamp']
        touch_frequency = total_touches / max(time_span, 1)
        
        # Foot preference analysis
        left_touches = sum(1 for t in touches if 'left' in t['foot'])
        right_touches = sum(1 for t in touches if 'right' in t['foot'])
        
        foot_preference = {
            'left': left_touches,
            'right': right_touches,
            'preference': self._determine_foot_preference(left_touches, right_touches)
        }
        
        # Touch consistency analysis
        touch_intervals = [t['timestamp'] for t in touches]
        consistency_metrics = self._analyze_touch_consistency(touch_intervals)
        
        # Touch quality analysis
        quality_scores = [t['quality_score'] for t in touches]
        avg_quality = np.mean(quality_scores) if quality_scores else 0
        
        # Ball movement analysis
        movement_metrics = self._analyze_ball_movement(ball_trajectory, touches)
        
        return {
            'total_touches': total_touches,
            'touch_frequency': touch_frequency,
            'foot_preference': foot_preference,
            'consistency_metrics': consistency_metrics,
            'average_touch_quality': avg_quality,
            'movement_metrics': movement_metrics,
            'time_span': time_span
        }
    
    def _determine_foot_preference(self, left_count: int, right_count: int) -> str:
        """Determine foot preference based on touch counts."""
        if abs(left_count - right_count) <= 1:
            return 'balanced'
        elif left_count > right_count:
            return 'left'
        else:
            return 'right'
    
    def _analyze_touch_consistency(self, touch_times: List[float]) -> Dict[str, float]:
        """
        Analyze consistency of ball touches over time.
        
        Args:
            touch_times: List of touch timestamps
            
        Returns:
            Dictionary with consistency metrics
        """
        if len(touch_times) < 3:
            return {'consistency_score': 0.0, 'avg_interval': 0.0, 'interval_std': 0.0}
        
        # Calculate intervals between touches
        intervals = []
        for i in range(1, len(touch_times)):
            intervals.append(touch_times[i] - touch_times[i-1])
        
        avg_interval = np.mean(intervals)
        interval_std = np.std(intervals)
        
        # Consistency score: lower std deviation = higher consistency
        consistency_score = 1.0 / (1.0 + interval_std) if interval_std > 0 else 1.0
        
        return {
            'consistency_score': consistency_score,
            'avg_interval': avg_interval,
            'interval_std': interval_std
        }
    
    def _analyze_ball_movement(self, trajectory: List[tuple], 
                              touches: List[Dict]) -> Dict[str, float]:
        """
        Analyze ball movement patterns relative to touches.
        
        Args:
            trajectory: Ball trajectory coordinates
            touches: Ball touch events
            
        Returns:
            Movement analysis metrics
        """
        if len(trajectory) < 2:
            return {'total_distance': 0.0, 'avg_speed': 0.0, 'direction_changes': 0}
        
        # Calculate total distance traveled
        total_distance = 0.0
        direction_changes = 0
        prev_direction = None
        
        for i in range(1, len(trajectory)):
            # Distance between consecutive points
            dist = np.sqrt(
                (trajectory[i][0] - trajectory[i-1][0])**2 + 
                (trajectory[i][1] - trajectory[i-1][1])**2
            )
            total_distance += dist
            
            # Direction analysis
            if i > 1:
                curr_direction = np.arctan2(
                    trajectory[i][1] - trajectory[i-1][1],
                    trajectory[i][0] - trajectory[i-1][0]
                )
                
                if prev_direction is not None:
                    angle_diff = abs(curr_direction - prev_direction)
                    if angle_diff > np.pi / 4:  # 45 degree threshold
                        direction_changes += 1
                
                prev_direction = curr_direction
        
        # Calculate average speed (distance per frame)
        avg_speed = total_distance / len(trajectory) if trajectory else 0.0
        
        return {
            'total_distance': total_distance,
            'avg_speed': avg_speed,
            'direction_changes': direction_changes
        }
    
    def generate_feedback(self, analysis: Dict[str, Any]) -> List[str]:
        """
        Generate coaching feedback based on analysis results.
        
        Args:
            analysis: Analysis results from analyze_dribbling_pattern
            
        Returns:
            List of feedback messages
        """
        if analysis is None:
            return ["Insufficient data for analysis. Need more ball touches."]
        
        feedback = []
        
        # Touch frequency feedback
        touch_freq = analysis['touch_frequency']
        if touch_freq < 1.0:
            feedback.append("Increase ball contact frequency for better control")
        elif touch_freq > 3.0:
            feedback.append("Good ball contact frequency - maintain this rhythm")
        else:
            feedback.append("Moderate ball contact frequency")
        
        # Consistency feedback
        consistency = analysis['consistency_metrics']['consistency_score']
        if consistency < 0.5:
            feedback.append("Work on consistent touch timing for better control")
        else:
            feedback.append("Excellent touch consistency!")
        
        # Foot preference feedback
        foot_pref = analysis['foot_preference']
        if foot_pref['preference'] == 'balanced':
            feedback.append(FEEDBACK_MESSAGES['balanced_feet'])
        else:
            feedback.append(
                FEEDBACK_MESSAGES['foot_dominance'].format(
                    foot=foot_pref['preference'].capitalize()
                )
            )
        
        # Touch quality feedback
        avg_quality = analysis['average_touch_quality']
        if avg_quality > 0.8:
            feedback.append("Excellent ball touch quality!")
        elif avg_quality > 0.6:
            feedback.append("Good ball touch quality - keep improving")
        else:
            feedback.append("Focus on getting closer ball contacts")
        
        # Movement feedback
        movement = analysis['movement_metrics']
        if movement['direction_changes'] > 5:
            feedback.append("Good ball control with multiple direction changes")
        elif movement['avg_speed'] > 20:
            feedback.append("High ball movement speed - work on close control")
        
        return feedback
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive performance summary.
        
        Returns:
            Dictionary with performance metrics and statistics
        """
        total_touches = self.performance_metrics['total_touches']
        
        summary = {
            'total_touches': total_touches,
            'foot_preference': self.performance_metrics['foot_contacts'],
            'average_touch_quality': np.mean(self.performance_metrics['touch_quality_scores']) 
                                   if self.performance_metrics['touch_quality_scores'] else 0,
            'touch_frequency': len(self.touches_history) / max(1, len(set(t['frame_idx'] for t in self.touches_history))),
            'consistency_score': self._calculate_overall_consistency()
        }
        
        return summary
    
    def _calculate_overall_consistency(self) -> float:
        """Calculate overall touch consistency score."""
        if len(self.performance_metrics['touch_intervals']) < 2:
            return 0.0
        
        intervals = self.performance_metrics['touch_intervals']
        std_dev = np.std(intervals)
        return 1.0 / (1.0 + std_dev) if std_dev > 0 else 1.0
    
    def reset_metrics(self):
        """Reset all performance metrics and history."""
        self.touches_history.clear()
        self.last_touch_frame.clear()
        self.continuous_touches.clear()
        self.performance_metrics = {
            'total_touches': 0,
            'foot_contacts': {'left': 0, 'right': 0},
            'touch_quality_scores': [],
            'touch_intervals': []
        }