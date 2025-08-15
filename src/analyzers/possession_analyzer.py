"""
Ball possession analysis module inspired by professional football analytics.
Implements team possession calculation and ball-to-player assignment.
"""

import cv2
import numpy as np
import math
from typing import List, Dict, Any, Optional, Tuple
from collections import deque
from ..config import TEAM_COLORS, MAX_PLAYER_SPEED_KMH


class PossessionAnalyzer:
    """
    Advanced ball possession analysis system.
    Tracks which team/player has possession of the ball.
    """
    
    def __init__(self, possession_threshold: float = 80.0, grace_period: int = 10):
        """
        Initialize possession analyzer.
        
        Args:
            possession_threshold: Distance threshold for ball possession (pixels)
            grace_period: Number of frames to maintain possession after losing ball
        """
        self.possession_threshold = possession_threshold
        self.grace_period = grace_period
        
        # Possession tracking
        self.current_possession = None  # Current team with possession
        self.possession_player = None   # Current player with possession
        self.possession_history = deque(maxlen=1000)
        self.team_possession_time = {'team_1': 0, 'team_2': 0, 'unknown': 0}
        self.grace_counter = 0
        self.last_valid_possession = None
        
        # Ball validation
        self.ball_positions = deque(maxlen=5)
        self.max_ball_speed = 500  # Maximum realistic ball speed (pixels/frame)
        
    def analyze_possession(self, ball_detections: List[Dict], 
                          player_detections: List[Dict], 
                          frame_idx: int) -> Dict[str, Any]:
        """
        Analyze ball possession for current frame.
        
        Args:
            ball_detections: List of ball detection dictionaries
            player_detections: List of player detection dictionaries
            frame_idx: Current frame index
            
        Returns:
            Possession analysis results
        """
        possession_result = {
            'frame_idx': frame_idx,
            'current_possession': None,
            'possession_player': None,
            'ball_position': None,
            'nearest_player': None,
            'distance_to_ball': None,
            'valid_ball': False,
            'team_possession_percentages': self.get_possession_percentages()
        }
        
        # Check if we have valid ball detection
        if not ball_detections:
            self._handle_no_ball(possession_result, frame_idx)
            return possession_result
        
        # Get best ball detection
        ball = max(ball_detections, key=lambda x: x.get('confidence', 0))
        ball_pos = ball['center']
        possession_result['ball_position'] = ball_pos
        
        # Validate ball position
        if not self._is_valid_ball_position(ball_pos):
            self._handle_invalid_ball(possession_result, frame_idx)
            return possession_result
        
        possession_result['valid_ball'] = True
        self.ball_positions.append(ball_pos)
        
        # Find nearest player to ball
        nearest_player, min_distance = self._find_nearest_player(ball_pos, player_detections)
        
        if nearest_player:
            possession_result['nearest_player'] = nearest_player
            possession_result['distance_to_ball'] = min_distance
            
            # Check if player has possession
            if min_distance <= self.possession_threshold:
                self._assign_possession(nearest_player, possession_result, frame_idx)
            else:
                self._handle_no_possession(possession_result, frame_idx)
        else:
            self._handle_no_possession(possession_result, frame_idx)
        
        return possession_result
    
    def _is_valid_ball_position(self, ball_pos: Tuple[int, int]) -> bool:
        """Validate ball position based on movement speed."""
        if len(self.ball_positions) < 2:
            return True
        
        last_pos = self.ball_positions[-1]
        distance = math.sqrt(
            (ball_pos[0] - last_pos[0])**2 + 
            (ball_pos[1] - last_pos[1])**2
        )
        
        # Check if ball moved too fast (likely detection error)
        if distance > self.max_ball_speed:
            return False
        
        return True
    
    def _find_nearest_player(self, ball_pos: Tuple[int, int], 
                           players: List[Dict]) -> Tuple[Optional[Dict], float]:
        """Find the player nearest to the ball."""
        if not players:
            return None, float('inf')
        
        min_distance = float('inf')
        nearest_player = None
        
        for player in players:
            player_pos = player['center']
            distance = math.sqrt(
                (ball_pos[0] - player_pos[0])**2 + 
                (ball_pos[1] - player_pos[1])**2
            )
            
            if distance < min_distance:
                min_distance = distance
                nearest_player = player
        
        return nearest_player, min_distance
    
    def _assign_possession(self, player: Dict, result: Dict, frame_idx: int):
        """Assign possession to a player and their team."""
        player_id = player.get('player_id', -1)
        team = player.get('team', 'unknown')
        
        # Update current possession
        if self.current_possession != team or self.possession_player != player_id:
            self.current_possession = team
            self.possession_player = player_id
            self.last_valid_possession = team
            self.grace_counter = 0
            
            # Log possession change
            self.possession_history.append({
                'frame_idx': frame_idx,
                'team': team,
                'player_id': player_id,
                'event': 'possession_gained'
            })
        
        # Update possession time
        if team in self.team_possession_time:
            self.team_possession_time[team] += 1
        else:
            self.team_possession_time['unknown'] += 1
        
        result['current_possession'] = team
        result['possession_player'] = player_id
    
    def _handle_no_possession(self, result: Dict, frame_idx: int):
        """Handle case where no player has possession."""
        self.grace_counter += 1
        
        if self.grace_counter <= self.grace_period and self.last_valid_possession:
            # Maintain possession during grace period
            result['current_possession'] = self.last_valid_possession
            result['possession_player'] = self.possession_player
            
            # Continue counting possession time
            team = self.last_valid_possession
            if team in self.team_possession_time:
                self.team_possession_time[team] += 1
            else:
                self.team_possession_time['unknown'] += 1
        else:
            # Lose possession
            if self.current_possession is not None:
                self.possession_history.append({
                    'frame_idx': frame_idx,
                    'team': self.current_possession,
                    'player_id': self.possession_player,
                    'event': 'possession_lost'
                })
            
            self.current_possession = None
            self.possession_player = None
            self.team_possession_time['unknown'] += 1
    
    def _handle_no_ball(self, result: Dict, frame_idx: int):
        """Handle case where no ball is detected."""
        self.grace_counter += 1
        
        if self.grace_counter <= self.grace_period and self.last_valid_possession:
            # Maintain possession during grace period
            result['current_possession'] = self.last_valid_possession
            result['possession_player'] = self.possession_player
            
            # Continue counting possession time
            team = self.last_valid_possession
            if team in self.team_possession_time:
                self.team_possession_time[team] += 1
            else:
                self.team_possession_time['unknown'] += 1
        else:
            # Lose possession
            self._handle_no_possession(result, frame_idx)
    
    def _handle_invalid_ball(self, result: Dict, frame_idx: int):
        """Handle case where ball position is invalid."""
        # Treat as no ball detected
        self._handle_no_ball(result, frame_idx)
    
    def get_possession_percentages(self) -> Dict[str, float]:
        """Calculate possession percentages for each team."""
        total_time = sum(self.team_possession_time.values())
        
        if total_time == 0:
            return {'team_1': 0.0, 'team_2': 0.0, 'unknown': 0.0}
        
        percentages = {}
        for team, time in self.team_possession_time.items():
            percentages[team] = (time / total_time) * 100.0
        
        return percentages
    
    def get_possession_events(self) -> List[Dict]:
        """Get list of possession change events."""
        return list(self.possession_history)
    
    def get_detailed_statistics(self) -> Dict[str, Any]:
        """Get comprehensive possession statistics."""
        percentages = self.get_possession_percentages()
        total_frames = sum(self.team_possession_time.values())
        
        stats = {
            'possession_percentages': percentages,
            'total_frames_analyzed': total_frames,
            'possession_changes': len(self.possession_history),
            'current_possession': self.current_possession,
            'current_player': self.possession_player,
            'team_possession_frames': dict(self.team_possession_time),
            'possession_events': list(self.possession_history)[-10:],  # Last 10 events
            'average_possession_duration': self._calculate_average_possession_duration()
        }
        
        return stats
    
    def _calculate_average_possession_duration(self) -> Dict[str, float]:
        """Calculate average possession duration for each team."""
        if len(self.possession_history) < 2:
            return {'team_1': 0.0, 'team_2': 0.0}
        
        team_durations = {'team_1': [], 'team_2': [], 'unknown': []}
        
        for i in range(1, len(self.possession_history)):
            if self.possession_history[i]['event'] == 'possession_lost':
                # Find corresponding gain event
                team = self.possession_history[i]['team']
                end_frame = self.possession_history[i]['frame_idx']
                
                # Look backwards for possession gain
                for j in range(i-1, -1, -1):
                    if (self.possession_history[j]['team'] == team and 
                        self.possession_history[j]['event'] == 'possession_gained'):
                        start_frame = self.possession_history[j]['frame_idx']
                        duration = end_frame - start_frame
                        if team in team_durations:
                            team_durations[team].append(duration)
                        break
        
        # Calculate averages
        averages = {}
        for team, durations in team_durations.items():
            averages[team] = np.mean(durations) if durations else 0.0
        
        return averages
    
    def draw_possession_info(self, frame, possession_result: Dict[str, Any], 
                           ball_detections: List[Dict]):
        """Draw possession information on the frame."""
        h, w = frame.shape[:2]
        
        # Draw possession indicator
        current_team = possession_result.get('current_possession')
        if current_team and current_team in TEAM_COLORS:
            color = TEAM_COLORS[current_team]
            
            # Draw possession indicator in top-left corner
            cv2.rectangle(frame, (10, 10), (200, 60), color, -1)
            cv2.rectangle(frame, (10, 10), (200, 60), (255, 255, 255), 2)
            
            text = f"Possession: {current_team.upper()}"
            cv2.putText(frame, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                       (255, 255, 255), 2)
            
            # Draw player ID if available
            player_id = possession_result.get('possession_player')
            if player_id is not None:
                player_text = f"Player: {player_id}"
                cv2.putText(frame, player_text, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 
                           (255, 255, 255), 1)
        
        # Draw possession percentages
        percentages = possession_result.get('team_possession_percentages', {})
        y_offset = 80
        
        for team, percentage in percentages.items():
            if team in TEAM_COLORS and percentage > 0:
                color = TEAM_COLORS[team]
                text = f"{team.upper()}: {percentage:.1f}%"
                cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, color, 2)
                y_offset += 25
        
        # Draw connection line between ball and possessing player
        if (possession_result.get('valid_ball') and 
            possession_result.get('nearest_player') and 
            possession_result.get('distance_to_ball', float('inf')) <= self.possession_threshold):
            
            ball_pos = possession_result['ball_position']
            player_center = possession_result['nearest_player']['center']
            
            # Draw line
            cv2.line(frame, ball_pos, player_center, (255, 255, 0), 2)
            
            # Draw distance text
            distance = possession_result['distance_to_ball']
            mid_point = (
                (ball_pos[0] + player_center[0]) // 2,
                (ball_pos[1] + player_center[1]) // 2
            )
            cv2.putText(frame, f"{distance:.0f}px", mid_point, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    def reset_analysis(self):
        """Reset all possession analysis data."""
        self.current_possession = None
        self.possession_player = None
        self.possession_history.clear()
        self.team_possession_time = {'team_1': 0, 'team_2': 0, 'unknown': 0}
        self.grace_counter = 0
        self.last_valid_possession = None
        self.ball_positions.clear()
        print("🔄 Possession analysis reset")