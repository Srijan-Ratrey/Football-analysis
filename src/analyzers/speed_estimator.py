"""
Speed estimation module for football analysis.
Calculates real-world player speeds using field scaling and homography.
"""

import cv2
import numpy as np
import math
from typing import List, Dict, Any, Tuple, Optional
from collections import deque
from ..config import FIELD_LENGTH_M, FIELD_WIDTH_M, MAX_PLAYER_SPEED_KMH


class SpeedEstimator:
    """
    Real-world speed estimation for football players.
    Converts pixel movements to real-world speeds using field scaling.
    """
    
    def __init__(self, fps: float = 30.0, smoothing_window: int = 5):
        """
        Initialize speed estimator.
        
        Args:
            fps: Frames per second of the video
            smoothing_window: Number of frames for speed smoothing
        """
        self.fps = fps
        self.smoothing_window = smoothing_window
        self.max_speed_ms = MAX_PLAYER_SPEED_KMH / 3.6  # Convert km/h to m/s
        
        # Field scaling (pixels to meters) - will be calculated from field detection
        self.scale_x = None  # pixels per meter in x direction
        self.scale_y = None  # pixels per meter in y direction
        self.field_detected = False
        
        # Default scaling if field detection fails
        self.default_scale = 10.0  # Rough estimate: 10 pixels per meter
        
    def set_field_scale(self, field_width_pixels: float, field_height_pixels: float):
        """
        Set field scaling based on detected field dimensions.
        
        Args:
            field_width_pixels: Field width in pixels
            field_height_pixels: Field height in pixels
        """
        self.scale_x = field_width_pixels / FIELD_WIDTH_M
        self.scale_y = field_height_pixels / FIELD_LENGTH_M
        self.field_detected = True
        
        print(f"🏟️ Field scale set: {self.scale_x:.2f} px/m (width), {self.scale_y:.2f} px/m (length)")
    
    def estimate_speeds(self, players: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Estimate speeds for all players.
        
        Args:
            players: List of player detection dictionaries
            
        Returns:
            Players list with added speed information
        """
        enhanced_players = []
        
        for player in players:
            enhanced_player = player.copy()
            
            # Calculate speed if we have trajectory data
            trajectory = player.get('trajectory', [])
            if len(trajectory) >= 2:
                speed_ms, speed_kmh = self._calculate_speed(trajectory)
                enhanced_player['speed_ms'] = speed_ms
                enhanced_player['speed_kmh'] = speed_kmh
                enhanced_player['speed_smoothed'] = self._smooth_speed(player, speed_kmh)
            else:
                enhanced_player['speed_ms'] = 0.0
                enhanced_player['speed_kmh'] = 0.0
                enhanced_player['speed_smoothed'] = 0.0
            
            enhanced_players.append(enhanced_player)
        
        return enhanced_players
    
    def _calculate_speed(self, trajectory: List[Tuple[int, int]]) -> Tuple[float, float]:
        """
        Calculate speed based on trajectory.
        
        Args:
            trajectory: List of (x, y) positions
            
        Returns:
            Tuple of (speed_m/s, speed_km/h)
        """
        if len(trajectory) < 2:
            return 0.0, 0.0
        
        # Get last two positions
        pos1 = trajectory[-2]
        pos2 = trajectory[-1]
        
        # Calculate pixel distance
        pixel_distance = math.sqrt(
            (pos2[0] - pos1[0])**2 + 
            (pos2[1] - pos1[1])**2
        )
        
        # Convert to real-world distance
        real_distance = self._pixels_to_meters(pixel_distance)
        
        # Calculate time difference (assuming consecutive frames)
        time_diff = 1.0 / self.fps
        
        # Calculate speed
        speed_ms = real_distance / time_diff
        
        # Apply maximum speed limit
        speed_ms = min(speed_ms, self.max_speed_ms)
        
        # Convert to km/h
        speed_kmh = speed_ms * 3.6
        
        return speed_ms, speed_kmh
    
    def _pixels_to_meters(self, pixel_distance: float) -> float:
        """
        Convert pixel distance to real-world meters.
        
        Args:
            pixel_distance: Distance in pixels
            
        Returns:
            Distance in meters
        """
        if self.field_detected and self.scale_x and self.scale_y:
            # Use average of x and y scales
            avg_scale = (self.scale_x + self.scale_y) / 2
            return pixel_distance / avg_scale
        else:
            # Use default scaling
            return pixel_distance / self.default_scale
    
    def _smooth_speed(self, player: Dict[str, Any], current_speed: float) -> float:
        """
        Apply smoothing to speed calculation.
        
        Args:
            player: Player dictionary
            current_speed: Current calculated speed
            
        Returns:
            Smoothed speed
        """
        # Initialize speed history if not present
        if 'speed_history' not in player:
            player['speed_history'] = deque(maxlen=self.smoothing_window)
        
        # Add current speed to history
        player['speed_history'].append(current_speed)
        
        # Calculate smoothed speed (moving average)
        if len(player['speed_history']) > 0:
            return sum(player['speed_history']) / len(player['speed_history'])
        else:
            return current_speed
    
    def get_speed_statistics(self, players: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate comprehensive speed statistics.
        
        Args:
            players: List of players with speed data
            
        Returns:
            Speed statistics dictionary
        """
        speeds = []
        team_speeds = {'team_1': [], 'team_2': [], 'unknown': []}
        
        for player in players:
            speed = player.get('speed_kmh', 0)
            if speed > 0:
                speeds.append(speed)
                team = player.get('team', 'unknown')
                if team in team_speeds:
                    team_speeds[team].append(speed)
                else:
                    team_speeds['unknown'].append(speed)
        
        stats = {
            'overall_stats': {
                'max_speed': max(speeds) if speeds else 0,
                'avg_speed': np.mean(speeds) if speeds else 0,
                'min_speed': min(speeds) if speeds else 0,
                'total_players': len([s for s in speeds if s > 0])
            },
            'team_stats': {},
            'speed_distribution': self._calculate_speed_distribution(speeds)
        }
        
        # Calculate team-specific statistics
        for team, team_speed_list in team_speeds.items():
            if team_speed_list:
                stats['team_stats'][team] = {
                    'max_speed': max(team_speed_list),
                    'avg_speed': np.mean(team_speed_list),
                    'min_speed': min(team_speed_list),
                    'player_count': len(team_speed_list)
                }
            else:
                stats['team_stats'][team] = {
                    'max_speed': 0, 'avg_speed': 0, 'min_speed': 0, 'player_count': 0
                }
        
        return stats
    
    def _calculate_speed_distribution(self, speeds: List[float]) -> Dict[str, int]:
        """Calculate speed distribution in ranges."""
        if not speeds:
            return {}
        
        distribution = {
            'stationary (0-2 km/h)': 0,
            'walking (2-7 km/h)': 0,
            'jogging (7-15 km/h)': 0,
            'running (15-25 km/h)': 0,
            'sprinting (25+ km/h)': 0
        }
        
        for speed in speeds:
            if speed <= 2:
                distribution['stationary (0-2 km/h)'] += 1
            elif speed <= 7:
                distribution['walking (2-7 km/h)'] += 1
            elif speed <= 15:
                distribution['jogging (7-15 km/h)'] += 1
            elif speed <= 25:
                distribution['running (15-25 km/h)'] += 1
            else:
                distribution['sprinting (25+ km/h)'] += 1
        
        return distribution
    
    def draw_speed_info(self, frame, players: List[Dict[str, Any]], 
                       show_individual: bool = True, show_stats: bool = True):
        """
        Draw speed information on the frame.
        
        Args:
            frame: Video frame to draw on
            players: List of players with speed data
            show_individual: Whether to show individual player speeds
            show_stats: Whether to show overall speed statistics
        """
        h, w = frame.shape[:2]
        
        # Draw individual player speeds
        if show_individual:
            for player in players:
                if player.get('speed_kmh', 0) > 1:  # Only show if moving
                    bbox = player['bbox']
                    speed_kmh = player.get('speed_smoothed', player.get('speed_kmh', 0))
                    
                    # Draw speed text above bounding box
                    speed_text = f"{speed_kmh:.1f} km/h"
                    text_pos = (bbox[0], bbox[1] - 25)
                    
                    # Color code by speed
                    if speed_kmh > 20:
                        color = (0, 0, 255)  # Red for high speed
                    elif speed_kmh > 10:
                        color = (0, 165, 255)  # Orange for medium speed
                    else:
                        color = (0, 255, 0)  # Green for low speed
                    
                    cv2.putText(frame, speed_text, text_pos, 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw speed statistics
        if show_stats:
            stats = self.get_speed_statistics(players)
            overall = stats['overall_stats']
            
            # Draw stats panel
            panel_x, panel_y = w - 250, 10
            panel_w, panel_h = 240, 120
            
            # Semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, (panel_x, panel_y), 
                         (panel_x + panel_w, panel_y + panel_h), 
                         (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Border
            cv2.rectangle(frame, (panel_x, panel_y), 
                         (panel_x + panel_w, panel_y + panel_h), 
                         (255, 255, 255), 2)
            
            # Title
            cv2.putText(frame, "SPEED STATS", (panel_x + 10, panel_y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Statistics
            y_offset = panel_y + 40
            stats_text = [
                f"Max: {overall['max_speed']:.1f} km/h",
                f"Avg: {overall['avg_speed']:.1f} km/h",
                f"Active: {overall['total_players']} players"
            ]
            
            for text in stats_text:
                cv2.putText(frame, text, (panel_x + 10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y_offset += 20
    
    def estimate_distance_covered(self, players: List[Dict[str, Any]], 
                                 time_period: float) -> Dict[str, Any]:
        """
        Estimate total distance covered by players.
        
        Args:
            players: List of players with trajectory data
            time_period: Time period in seconds
            
        Returns:
            Distance statistics
        """
        distance_stats = {
            'total_distance': {},
            'average_distance': {},
            'team_totals': {'team_1': 0, 'team_2': 0, 'unknown': 0}
        }
        
        for player in players:
            player_id = player.get('player_id', -1)
            team = player.get('team', 'unknown')
            trajectory = player.get('trajectory', [])
            
            if len(trajectory) < 2:
                continue
            
            # Calculate total distance for trajectory
            total_distance = 0
            for i in range(1, len(trajectory)):
                pixel_dist = math.sqrt(
                    (trajectory[i][0] - trajectory[i-1][0])**2 + 
                    (trajectory[i][1] - trajectory[i-1][1])**2
                )
                real_dist = self._pixels_to_meters(pixel_dist)
                total_distance += real_dist
            
            distance_stats['total_distance'][player_id] = total_distance
            
            # Add to team total
            if team in distance_stats['team_totals']:
                distance_stats['team_totals'][team] += total_distance
        
        # Calculate averages per team
        team_counts = {'team_1': 0, 'team_2': 0, 'unknown': 0}
        for player in players:
            team = player.get('team', 'unknown')
            if team in team_counts:
                team_counts[team] += 1
        
        for team in distance_stats['team_totals']:
            if team_counts[team] > 0:
                distance_stats['average_distance'][team] = (
                    distance_stats['team_totals'][team] / team_counts[team]
                )
            else:
                distance_stats['average_distance'][team] = 0
        
        return distance_stats