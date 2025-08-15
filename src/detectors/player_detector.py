"""
Advanced player detection and tracking module inspired by professional football analysis.
Implements multi-class detection for players, goalkeepers, and referees.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from collections import deque, OrderedDict
import math
from ultralytics import YOLO
from ..config import (
    PLAYER_CONFIDENCE, PLAYER_CLASS_ID, TEAM_COLORS, JERSEY_COLOR_RANGES,
    MAX_DISAPPEARED, EUCLIDEAN_DIST_THRESHOLD, YOLO_MODEL_PATH
)


class Player:
    """Represents a tracked player with complete information."""
    
    def __init__(self, player_id: int, bbox: Tuple[int, int, int, int], 
                 confidence: float, frame_idx: int):
        self.player_id = player_id
        self.bbox = bbox
        self.center = self._calculate_center(bbox)
        self.confidence = confidence
        self.positions = deque([self.center], maxlen=30)
        self.speeds = deque(maxlen=10)
        self.team = None
        self.role = 'player'  # player, goalkeeper, referee
        self.jersey_color = None
        self.jersey_history = deque(maxlen=15)
        self.first_seen = frame_idx
        self.last_seen = frame_idx
        self.total_frames = 1
        self.disappeared = 0
        
    def _calculate_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """Calculate center point from bounding box."""
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))
    
    def update(self, bbox: Tuple[int, int, int, int], confidence: float, frame_idx: int):
        """Update player information with new detection."""
        self.bbox = bbox
        old_center = self.center
        self.center = self._calculate_center(bbox)
        self.confidence = confidence
        self.positions.append(self.center)
        self.last_seen = frame_idx
        self.total_frames += 1
        self.disappeared = 0
        
        # Calculate speed
        if len(self.positions) >= 2:
            distance = math.sqrt(
                (self.center[0] - old_center[0])**2 + 
                (self.center[1] - old_center[1])**2
            )
            self.speeds.append(distance)
    
    def mark_disappeared(self):
        """Mark player as disappeared for one frame."""
        self.disappeared += 1
    
    def get_average_speed(self) -> float:
        """Get average speed over recent frames."""
        return np.mean(list(self.speeds)) if self.speeds else 0.0
    
    def get_trajectory(self) -> List[Tuple[int, int]]:
        """Get player trajectory as list of positions."""
        return list(self.positions)


class PlayerDetector:
    """
    Advanced player detection and tracking system.
    Detects and tracks multiple players, goalkeepers, and referees.
    """
    
    def __init__(self, confidence: float = PLAYER_CONFIDENCE):
        """Initialize the player detector."""
        self.confidence = confidence
        self.next_player_id = 0
        self.players: Dict[int, Player] = OrderedDict()
        self.frame_count = 0
        
        # Initialize YOLO model
        print("🔄 Loading Advanced Player Detection System...")
        try:
            self.model = YOLO(YOLO_MODEL_PATH)
            print("✅ Player detection system loaded successfully!")
        except Exception as e:
            print(f"⚠️ Player detection loading error: {e}")
            self.model = YOLO(YOLO_MODEL_PATH)
            print("✅ Player detection system loaded!")
    
    def detect_players(self, frame) -> List[Dict[str, Any]]:
        """
        Detect and track players in the frame.
        
        Args:
            frame: Input video frame
            
        Returns:
            List of player detection dictionaries
        """
        # Run YOLO inference for person detection
        results = self.model(
            frame, 
            conf=self.confidence, 
            classes=[PLAYER_CLASS_ID],  # Person class
            verbose=False
        )
        
        # Process detections
        detections = []
        current_boxes = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    bbox = (int(x1), int(y1), int(x2), int(y2))
                    current_boxes.append({
                        'bbox': bbox,
                        'confidence': confidence
                    })
        
        # Update tracking
        self._update_tracking(current_boxes)
        
        # Convert tracked players to detection format
        for player_id, player in self.players.items():
            if player.disappeared <= MAX_DISAPPEARED:
                detection = {
                    'player_id': player_id,
                    'bbox': player.bbox,
                    'center': player.center,
                    'confidence': player.confidence,
                    'team': player.team,
                    'role': player.role,
                    'speed': player.get_average_speed(),
                    'trajectory': player.get_trajectory()[-10:],  # Last 10 positions
                    'jersey_color': player.jersey_color,
                    'total_frames': player.total_frames
                }
                detections.append(detection)
        
        self.frame_count += 1
        return detections
    
    def _update_tracking(self, current_boxes: List[Dict]):
        """Update player tracking with current detections."""
        if not current_boxes:
            # Mark all players as disappeared
            for player in self.players.values():
                player.mark_disappeared()
            return
        
        # If no existing players, create new ones
        if not self.players:
            for box_data in current_boxes:
                self._create_new_player(box_data['bbox'], box_data['confidence'])
            return
        
        # Calculate distance matrix between existing players and new detections
        player_ids = list(self.players.keys())
        distances = np.zeros((len(player_ids), len(current_boxes)))
        
        for i, player_id in enumerate(player_ids):
            player = self.players[player_id]
            for j, box_data in enumerate(current_boxes):
                center = self._calculate_center(box_data['bbox'])
                distance = math.sqrt(
                    (player.center[0] - center[0])**2 + 
                    (player.center[1] - center[1])**2
                )
                distances[i, j] = distance
        
        # Simple assignment based on minimum distance
        used_box_indices = set()
        updated_players = set()
        
        # Assign detections to existing players
        for i, player_id in enumerate(player_ids):
            if player_id in updated_players:
                continue
                
            min_distance = float('inf')
            best_box_idx = -1
            
            for j in range(len(current_boxes)):
                if j in used_box_indices:
                    continue
                if distances[i, j] < min_distance and distances[i, j] < EUCLIDEAN_DIST_THRESHOLD:
                    min_distance = distances[i, j]
                    best_box_idx = j
            
            if best_box_idx != -1:
                # Update existing player
                box_data = current_boxes[best_box_idx]
                self.players[player_id].update(
                    box_data['bbox'], 
                    box_data['confidence'], 
                    self.frame_count
                )
                used_box_indices.add(best_box_idx)
                updated_players.add(player_id)
            else:
                # Mark player as disappeared
                self.players[player_id].mark_disappeared()
        
        # Create new players for unmatched detections
        for j, box_data in enumerate(current_boxes):
            if j not in used_box_indices:
                self._create_new_player(box_data['bbox'], box_data['confidence'])
        
        # Remove players that have been gone too long
        players_to_remove = []
        for player_id, player in self.players.items():
            if player.disappeared > MAX_DISAPPEARED:
                players_to_remove.append(player_id)
        
        for player_id in players_to_remove:
            del self.players[player_id]
    
    def _create_new_player(self, bbox: Tuple[int, int, int, int], confidence: float):
        """Create a new tracked player."""
        player = Player(self.next_player_id, bbox, confidence, self.frame_count)
        self.players[self.next_player_id] = player
        self.next_player_id += 1
    
    def _calculate_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """Calculate center point from bounding box."""
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))
    
    def assign_teams(self, frame, players: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enhanced team assignment with spatial analysis and better jersey extraction.
        """
        # First pass: extract jersey colors for all players
        valid_players = []
        jersey_colors = []
        positions = []
        
        for player in players:
            bbox = player['bbox']
            x1, y1, x2, y2 = bbox
            
            # More focused jersey region extraction
            body_height = y2 - y1
            body_width = x2 - x1
            
            # Skip very small detections
            if body_height < 40 or body_width < 20:
                player['team'] = 'unknown'
                continue
            
            # Focus on upper chest area (more specific)
            jersey_y1 = y1 + int(body_height * 0.2)   # Skip head
            jersey_y2 = y1 + int(body_height * 0.5)   # Upper chest only
            
            # Narrower width to avoid arms and background
            jersey_x1 = x1 + int(body_width * 0.3)
            jersey_x2 = x2 - int(body_width * 0.3)
            
            # Ensure valid region
            if (jersey_y2 < frame.shape[0] and jersey_x2 < frame.shape[1] and 
                jersey_y1 >= 0 and jersey_x1 >= 0 and jersey_x2 > jersey_x1):
                
                jersey_region = frame[jersey_y1:jersey_y2, jersey_x1:jersey_x2]
                
                if jersey_region.size > 50:  # Minimum size
                    # Extract dominant color
                    dominant_color = self._extract_dominant_jersey_color(jersey_region)
                    
                    # Store for analysis
                    jersey_colors.append(dominant_color)
                    positions.append((x1 + x2) / 2)  # Center X position
                    valid_players.append(player)
                    
                    player['jersey_color'] = tuple(map(int, dominant_color))
                    player['position_x'] = (x1 + x2) / 2
                else:
                    player['team'] = 'unknown'
            else:
                player['team'] = 'unknown'
        
        # Second pass: assign teams with spatial context
        if len(valid_players) >= 2:
            self._assign_teams_with_spatial_context(valid_players, jersey_colors, positions, frame.shape[1])
        else:
            # Fallback for single player or no valid players
            for player in valid_players:
                if 'jersey_color' in player:
                    color = player['jersey_color']
                    player['team'] = self._classify_team_by_color(np.array(color))
                else:
                    player['team'] = 'team_1'
        
        # Ensure all players have team assignment and stabilize via history
        for player in players:
            if player.get('team') is None:
                player['team'] = 'unknown'
            # Stabilize team using jersey color history from tracked Player
            tracked = self.players.get(player['player_id']) if isinstance(self.players, dict) else None
            if tracked is not None:
                color_tuple = player.get('jersey_color')
                if color_tuple is not None:
                    try:
                        tracked.jersey_history.append(np.array(color_tuple, dtype=float))
                    except Exception:
                        pass
                # Build valid history list
                valid_colors = [c for c in tracked.jersey_history if c is not None and len(c) == 3]
                if valid_colors:
                    try:
                        avg_color = np.mean(np.stack(valid_colors, axis=0), axis=0)
                        player['team'] = self._classify_team_by_color(avg_color)
                    except Exception:
                        pass
                
        return players
    
    def _extract_dominant_jersey_color(self, jersey_region):
        """Extract the most dominant color from jersey region using histogram analysis."""
        # Reshape for color analysis
        pixels = jersey_region.reshape(-1, 3)
        
        # Remove very dark and very bright pixels (likely shadows/highlights)
        mask = (pixels.sum(axis=1) > 50) & (pixels.sum(axis=1) < 650)
        filtered_pixels = pixels[mask]
        
        if len(filtered_pixels) == 0:
            return np.mean(pixels, axis=0)
        
        # Use histogram to find dominant color
        hist_r = np.histogram(filtered_pixels[:, 2], bins=8, range=(0, 256))[0]  # Red
        hist_g = np.histogram(filtered_pixels[:, 1], bins=8, range=(0, 256))[0]  # Green  
        hist_b = np.histogram(filtered_pixels[:, 0], bins=8, range=(0, 256))[0]  # Blue
        
        # Find dominant color ranges
        dominant_r = np.argmax(hist_r) * 32 + 16
        dominant_g = np.argmax(hist_g) * 32 + 16
        dominant_b = np.argmax(hist_b) * 32 + 16
        
        return np.array([dominant_r, dominant_g, dominant_b])
    
    def _classify_team_by_color(self, color: np.ndarray) -> str:
        """Much more aggressive team classification focusing on clear differences."""
        r, g, b = color
        
        # Calculate color characteristics
        brightness = (r + g + b) / 3
        
        # Very clear referee detection (bright/white)
        if brightness > 200 or (r > 200 and g > 200 and b > 200):
            return 'referee'
        
        # Calculate color differences for strong contrast detection
        max_color = max(r, g, b)
        min_color = min(r, g, b)
        contrast = max_color - min_color
        
        # Strong blue detection (prioritize blue team detection)
        if b >= r and b >= g and contrast > 40:
            if b > 100 or (b > r + 20 and b > g + 20):
                return 'team_2'
        
        # Strong red detection  
        if r >= g and r >= b and contrast > 40:
            if r > 100 or (r > g + 20 and r > b + 20):
                return 'team_1'
        
        # Green/Yellow to team_1 (warm colors)
        if g >= r and g >= b:
            return 'team_1'
            
        # Dark colors analysis
        if brightness < 90:
            # Dark blue/navy
            if b >= r and b >= g:
                return 'team_2'
            # Dark red/maroon  
            elif r >= g and r >= b:
                return 'team_1'
            else:
                # Other dark colors - split based on blue vs red tendency
                if b > r:
                    return 'team_2'
                else:
                    return 'team_1'
        
        # Medium brightness - be more aggressive in separation
        if r > b + 15:  # Red tendency
            return 'team_1'
        elif b > r + 15:  # Blue tendency  
            return 'team_2'
        elif g > r + 10 and g > b + 10:  # Green tendency
            return 'team_1'
        
        # Final fallback - split roughly 50/50 based on slight color bias
        total = r + g + b
        if total > 0:
            if (r + g) > b:  # Warmer
                return 'team_1'
            else:  # Cooler
                return 'team_2'
        
        return 'team_1'  # Final fallback
    
    def _assign_teams_with_spatial_context(self, players, jersey_colors, positions, frame_width):
        """Assign teams using both color and spatial context for better separation."""
        colors = np.array(jersey_colors)
        positions = np.array(positions)
        
        # Normalize positions (0-1)
        norm_positions = positions / frame_width
        
        # Use simple color clustering with position bias
        team_assignments = []
        
        for i, (color, pos) in enumerate(zip(colors, norm_positions)):
            r, g, b = color
            
            # Enhanced classification with spatial awareness
            team = self._classify_team_by_color(color)
            
            # Apply spatial context adjustments
            # Players on left side of field slightly more likely to be team_1
            # Players on right side slightly more likely to be team_2
            if team == 'team_1' and pos > 0.7:
                # Check if this should be team_2 based on very strong blue
                if b > r + 30 and b > g + 30:
                    team = 'team_2'
            elif team == 'team_2' and pos < 0.3:
                # Check if this should be team_1 based on very strong red/warm
                if (r > b + 30 and r > g + 10) or (r + g > b * 1.5 and r > 120):
                    team = 'team_1'
            
            team_assignments.append(team)
        
        # Assign calculated teams to players
        for player, team in zip(players, team_assignments):
            player['team'] = team
    
    def draw_players(self, frame, players: List[Dict[str, Any]], 
                    draw_trajectories: bool = True, draw_info: bool = True,
                    draw_team_indicators: bool = True):
        """Draw player detections with professional team-based visualization."""
        for player in players:
            bbox = player['bbox']
            center = player['center']
            player_id = player['player_id']
            team = player.get('team', 'unknown')
            speed = player.get('speed', 0)
            confidence = player.get('confidence', 0)
            
            # Choose color based on team
            if team in TEAM_COLORS:
                color = TEAM_COLORS[team]
            else:
                color = TEAM_COLORS['unknown']
            
            # Draw thin bounding box with team styling
            thickness = 1 if team != 'unknown' else 1
            
            # Main bounding box
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)
            
            # Add team indicator corner (top-left corner fill)
            if draw_team_indicators:
                corner_size = 15
                cv2.rectangle(
                    frame, 
                    (bbox[0], bbox[1]), 
                    (bbox[0] + corner_size, bbox[1] + corner_size), 
                    color, -1
                )
                
                # Add team number in corner
                team_number = self._get_team_number(team)
                cv2.putText(
                    frame, str(team_number),
                    (bbox[0] + 3, bbox[1] + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
                )
            
            # Draw center point with team color
            cv2.circle(frame, center, 6, color, -1)
            cv2.circle(frame, center, 6, (255, 255, 255), 1)  # White border
            
            # Small clean player tags without background
            if draw_info:
                # Simple compact tag
                info_text = f"P{player_id}"
                
                # Small text directly on video
                cv2.putText(
                    frame, info_text,
                    (bbox[0] + 2, bbox[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
                )
            
            # Enhanced trajectory with team colors
            if draw_trajectories and 'trajectory' in player:
                trajectory = player['trajectory']
                if len(trajectory) > 1:
                    for i in range(1, len(trajectory)):
                        alpha = (i / len(trajectory)) * 0.8  # Max 80% opacity
                        trail_color = tuple(int(c * alpha) for c in color)
                        cv2.line(frame, trajectory[i-1], trajectory[i], trail_color, 2)
                    
                    # Draw trajectory points
                    for i, point in enumerate(trajectory):
                        if i % 3 == 0:  # Every 3rd point
                            alpha = (i / len(trajectory)) * 0.6
                            point_color = tuple(int(c * alpha) for c in color)
                            cv2.circle(frame, point, 2, point_color, -1)
    
    def _get_team_number(self, team: str) -> int:
        """Get team number for simplified display."""
        team_numbers = {
            'team_1': 1,
            'team_2': 2,
            'referee': 0,
            'unknown': 0
        }
        return team_numbers.get(team, 0)
    
    def _get_team_display_name(self, team: str) -> str:
        """Get simplified team display name."""
        display_names = {
            'team_1': 'T1',
            'team_2': 'T2',
            'referee': 'REF',
            'unknown': '?'
        }
        return display_names.get(team, '?')
    
    def draw_team_legend(self, frame, players: List[Dict[str, Any]], 
                        position: Tuple[int, int] = (10, 50)):
        """Draw a team legend showing colors and player counts."""
        if not players:
            return
        
        # Count players per team
        team_counts = {}
        for player in players:
            team = player.get('team', 'unknown')
            team_counts[team] = team_counts.get(team, 0) + 1
        
        # Draw legend background
        legend_width = 200
        legend_height = len(team_counts) * 30 + 20
        x, y = position
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + legend_width, y + legend_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Border
        cv2.rectangle(frame, (x, y), (x + legend_width, y + legend_height), (255, 255, 255), 2)
        
        # Title
        cv2.putText(frame, "TEAM LEGEND", (x + 10, y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Team entries
        y_offset = y + 35
        for team, count in team_counts.items():
            if team in TEAM_COLORS:
                color = TEAM_COLORS[team]
                
                # Color box
                cv2.rectangle(frame, (x + 10, y_offset), (x + 25, y_offset + 15), color, -1)
                cv2.rectangle(frame, (x + 10, y_offset), (x + 25, y_offset + 15), (255, 255, 255), 1)
                
                # Team info
                team_text = f"{self._get_team_display_name(team)}: {count}"
                cv2.putText(frame, team_text, (x + 35, y_offset + 12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                y_offset += 25
    
    def get_team_statistics(self) -> Dict[str, Any]:
        """Get comprehensive team and player statistics."""
        stats = {
            'total_players': len(self.players),
            'active_players': len([p for p in self.players.values() 
                                 if p.disappeared <= MAX_DISAPPEARED]),
            'teams': {},
            'average_speeds': {},
            'player_details': []
        }
        
        team_counts = {}
        team_speeds = {}
        
        for player in self.players.values():
            if player.disappeared <= MAX_DISAPPEARED:
                team = player.team or 'unknown'
                
                # Count players per team
                team_counts[team] = team_counts.get(team, 0) + 1
                
                # Calculate average speeds per team
                avg_speed = player.get_average_speed()
                if team not in team_speeds:
                    team_speeds[team] = []
                team_speeds[team].append(avg_speed)
                
                # Player details
                stats['player_details'].append({
                    'player_id': player.player_id,
                    'team': team,
                    'average_speed': avg_speed,
                    'total_frames': player.total_frames,
                    'current_position': player.center
                })
        
        stats['teams'] = team_counts
        for team, speeds in team_speeds.items():
            stats['average_speeds'][team] = np.mean(speeds) if speeds else 0
        
        return stats
    
    def reset_tracking(self):
        """Reset all tracking data."""
        self.players.clear()
        self.next_player_id = 0
        self.frame_count = 0
        print("🔄 Player tracking reset")