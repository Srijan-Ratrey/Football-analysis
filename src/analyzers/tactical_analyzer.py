"""
FIFA-Quality Tactical Analysis Module
Advanced tactical analysis for passing, vision, and decision making.
"""

import numpy as np
import cv2
from typing import List, Dict, Any, Tuple, Optional
from collections import deque
import math

class TacticalAnalyzer:
    """FIFA-quality tactical analysis for football players."""
    
    def __init__(self):
        """Initialize tactical analyzer."""
        # FIFA tactical parameters
        self.field_length = 105.0  # meters
        self.field_width = 68.0    # meters
        
        # Tactical analysis parameters
        self.passing_window_frames = 60  # 2 seconds at 30fps
        self.vision_radius = 15.0  # meters
        self.press_distance = 3.0  # meters
        self.marking_distance = 2.0  # meters
        
        # Analysis history
        self.player_positions_history = deque(maxlen=300)  # 10 seconds
        self.ball_positions_history = deque(maxlen=300)
        self.pass_events = []
        self.decision_points = []
        
    def analyze_frame(self, frame_data: Dict[str, Any], frame_idx: int) -> Dict[str, Any]:
        """
        Analyze tactical aspects of current frame.
        
        Args:
            frame_data: Frame analysis data with players and ball
            frame_idx: Current frame index
            
        Returns:
            Tactical analysis results
        """
        players = frame_data.get('player_detections', [])
        ball_data = frame_data.get('ball_detections', [])
        
        # Store positions for historical analysis
        self._store_positions(players, ball_data, frame_idx)
        
        analysis = {}
        
        # Passing analysis
        analysis['passing'] = self._analyze_passing_opportunities(players, ball_data, frame_idx)
        
        # Vision and awareness analysis
        analysis['vision'] = self._analyze_player_vision(players, ball_data)
        
        # Defensive analysis
        analysis['defensive'] = self._analyze_defensive_play(players, ball_data)
        
        # Positioning analysis
        analysis['positioning'] = self._analyze_team_positioning(players)
        
        # Decision making analysis
        analysis['decision_making'] = self._analyze_decision_making(players, ball_data, frame_idx)
        
        return analysis
    
    def _store_positions(self, players: List[Dict], ball_data: List[Dict], frame_idx: int):
        """Store player and ball positions for historical analysis."""
        player_positions = {}
        for player in players:
            player_id = player.get('player_id', 'unknown')
            center = player.get('center', (0, 0))
            team = player.get('team', 'unknown')
            
            player_positions[player_id] = {
                'position': center,
                'team': team,
                'frame_idx': frame_idx
            }
        
        self.player_positions_history.append(player_positions)
        
        # Store ball position
        if ball_data:
            ball_center = ball_data[0].get('center', (0, 0))
            self.ball_positions_history.append({
                'position': ball_center,
                'frame_idx': frame_idx
            })
    
    def _analyze_passing_opportunities(self, players: List[Dict], ball_data: List[Dict], 
                                     frame_idx: int) -> Dict[str, Any]:
        """Analyze passing opportunities and decision making."""
        if not players or not ball_data:
            return {'opportunities': [], 'quality_score': 0.0}
        
        ball_pos = ball_data[0].get('center', (0, 0))
        
        # Find player with ball
        ball_carrier = self._find_ball_carrier(players, ball_pos)
        if not ball_carrier:
            return {'opportunities': [], 'quality_score': 0.0}
        
        ball_carrier_team = ball_carrier.get('team', 'unknown')
        teammates = [p for p in players if p.get('team') == ball_carrier_team and p != ball_carrier]
        opponents = [p for p in players if p.get('team') != ball_carrier_team]
        
        opportunities = []
        
        for teammate in teammates:
            opportunity = self._evaluate_pass_opportunity(ball_carrier, teammate, opponents)
            if opportunity['feasible']:
                opportunities.append(opportunity)
        
        # Calculate overall passing quality score
        quality_score = self._calculate_passing_quality_score(opportunities, opponents, ball_carrier)
        
        # Detect pass events
        self._detect_pass_events(ball_carrier, opportunities, frame_idx)
        
        return {
            'opportunities': opportunities,
            'quality_score': quality_score,
            'ball_carrier': ball_carrier.get('player_id', 'unknown'),
            'pressure_level': self._calculate_pressure_level(ball_carrier, opponents)
        }
    
    def _find_ball_carrier(self, players: List[Dict], ball_pos: Tuple[int, int]) -> Optional[Dict]:
        """Find player closest to ball (ball carrier)."""
        min_distance = float('inf')
        ball_carrier = None
        
        for player in players:
            player_pos = player.get('center', (0, 0))
            distance = math.sqrt((player_pos[0] - ball_pos[0])**2 + (player_pos[1] - ball_pos[1])**2)
            
            if distance < min_distance and distance < 50:  # 50 pixel threshold
                min_distance = distance
                ball_carrier = player
        
        return ball_carrier
    
    def _evaluate_pass_opportunity(self, ball_carrier: Dict, teammate: Dict, 
                                 opponents: List[Dict]) -> Dict[str, Any]:
        """Evaluate quality of pass opportunity to teammate."""
        carrier_pos = ball_carrier.get('center', (0, 0))
        teammate_pos = teammate.get('center', (0, 0))
        
        # Calculate pass distance and angle
        pass_vector = (teammate_pos[0] - carrier_pos[0], teammate_pos[1] - carrier_pos[1])
        pass_distance = math.sqrt(pass_vector[0]**2 + pass_vector[1]**2)
        pass_angle = math.atan2(pass_vector[1], pass_vector[0])
        
        # Check for obstacles (opponents in pass line)
        obstacles = self._check_pass_obstacles(carrier_pos, teammate_pos, opponents)
        
        # Calculate teammate's space (distance to nearest opponent)
        teammate_space = self._calculate_player_space(teammate, opponents)
        
        # Evaluate pass quality
        quality_factors = {
            'distance_score': max(0, 1 - pass_distance / 300),  # Prefer closer passes
            'obstacle_score': max(0, 1 - len(obstacles) * 0.3),  # Fewer obstacles better
            'space_score': min(1, teammate_space / 100),  # More space better
            'angle_score': self._evaluate_pass_angle(pass_angle, carrier_pos, teammate_pos)
        }
        
        overall_score = np.mean(list(quality_factors.values()))
        
        return {
            'teammate_id': teammate.get('player_id', 'unknown'),
            'teammate_position': teammate_pos,
            'pass_distance': pass_distance,
            'pass_angle': math.degrees(pass_angle),
            'obstacles': len(obstacles),
            'teammate_space': teammate_space,
            'quality_score': overall_score,
            'feasible': overall_score > 0.3 and len(obstacles) < 2,
            'quality_factors': quality_factors
        }
    
    def _check_pass_obstacles(self, start_pos: Tuple[int, int], end_pos: Tuple[int, int], 
                            opponents: List[Dict]) -> List[Dict]:
        """Check for opponents blocking pass line."""
        obstacles = []
        
        for opponent in opponents:
            opponent_pos = opponent.get('center', (0, 0))
            
            # Calculate distance from opponent to pass line
            line_distance = self._point_to_line_distance(opponent_pos, start_pos, end_pos)
            
            # If opponent is close to pass line, it's an obstacle
            if line_distance < 30:  # 30 pixel threshold
                obstacles.append(opponent)
        
        return obstacles
    
    def _point_to_line_distance(self, point: Tuple[int, int], line_start: Tuple[int, int], 
                              line_end: Tuple[int, int]) -> float:
        """Calculate distance from point to line segment."""
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Vector from line_start to line_end
        A = x2 - x1
        B = y2 - y1
        
        # Vector from line_start to point
        C = x0 - x1
        D = y0 - y1
        
        # Calculate distance
        dot = A * C + B * D
        len_sq = A * A + B * B
        
        if len_sq == 0:
            return math.sqrt(C * C + D * D)
        
        param = dot / len_sq
        
        if param < 0:
            xx, yy = x1, y1
        elif param > 1:
            xx, yy = x2, y2
        else:
            xx = x1 + param * A
            yy = y1 + param * B
        
        dx = x0 - xx
        dy = y0 - yy
        
        return math.sqrt(dx * dx + dy * dy)
    
    def _calculate_player_space(self, player: Dict, opponents: List[Dict]) -> float:
        """Calculate space around player (distance to nearest opponent)."""
        player_pos = player.get('center', (0, 0))
        min_distance = float('inf')
        
        for opponent in opponents:
            opponent_pos = opponent.get('center', (0, 0))
            distance = math.sqrt((player_pos[0] - opponent_pos[0])**2 + (player_pos[1] - opponent_pos[1])**2)
            min_distance = min(min_distance, distance)
        
        return min_distance if min_distance != float('inf') else 200
    
    def _evaluate_pass_angle(self, pass_angle: float, carrier_pos: Tuple[int, int], 
                           teammate_pos: Tuple[int, int]) -> float:
        """Evaluate pass angle quality (forward passes preferred)."""
        # Assume forward direction is positive X (right side of screen)
        forward_angle = 0  # 0 radians = straight right
        
        angle_diff = abs(pass_angle - forward_angle)
        angle_diff = min(angle_diff, 2 * math.pi - angle_diff)  # Use smaller angle
        
        # Score based on how close to forward the pass is
        return max(0, 1 - angle_diff / math.pi)
    
    def _calculate_passing_quality_score(self, opportunities: List[Dict], opponents: List[Dict], 
                                       ball_carrier: Dict) -> float:
        """Calculate overall passing quality score."""
        if not opportunities:
            return 0.0
        
        # Average quality of available opportunities
        avg_opportunity_quality = np.mean([opp['quality_score'] for opp in opportunities])
        
        # Number of options bonus
        options_bonus = min(0.3, len(opportunities) * 0.1)
        
        # Pressure penalty
        pressure_level = self._calculate_pressure_level(ball_carrier, opponents)
        pressure_penalty = pressure_level * 0.2
        
        return max(0, min(1, avg_opportunity_quality + options_bonus - pressure_penalty))
    
    def _calculate_pressure_level(self, player: Dict, opponents: List[Dict]) -> float:
        """Calculate pressure level on player (0-1 scale)."""
        player_pos = player.get('center', (0, 0))
        pressure = 0
        
        for opponent in opponents:
            opponent_pos = opponent.get('center', (0, 0))
            distance = math.sqrt((player_pos[0] - opponent_pos[0])**2 + (player_pos[1] - opponent_pos[1])**2)
            
            # Pressure increases as opponents get closer
            if distance < 100:  # 100 pixel pressure radius
                pressure += max(0, 1 - distance / 100)
        
        return min(1, pressure)
    
    def _detect_pass_events(self, ball_carrier: Dict, opportunities: List[Dict], frame_idx: int):
        """Detect and evaluate pass events."""
        # Look for ball position changes indicating passes
        if len(self.ball_positions_history) < 10:
            return
        
        current_ball_pos = self.ball_positions_history[-1]['position']
        previous_ball_pos = self.ball_positions_history[-10]['position']
        
        # Check if ball moved significantly (potential pass)
        movement = math.sqrt((current_ball_pos[0] - previous_ball_pos[0])**2 + 
                           (current_ball_pos[1] - previous_ball_pos[1])**2)
        
        if movement > 80:  # 80 pixel threshold for pass detection
            # Evaluate pass decision
            best_opportunity = max(opportunities, key=lambda x: x['quality_score']) if opportunities else None
            
            pass_event = {
                'frame_idx': frame_idx,
                'ball_carrier': ball_carrier.get('player_id', 'unknown'),
                'movement_distance': movement,
                'opportunities_available': len(opportunities),
                'best_opportunity_score': best_opportunity['quality_score'] if best_opportunity else 0,
                'decision_quality': self._evaluate_pass_decision(opportunities, movement)
            }
            
            self.pass_events.append(pass_event)
    
    def _evaluate_pass_decision(self, opportunities: List[Dict], movement_distance: float) -> str:
        """Evaluate quality of pass decision."""
        if not opportunities:
            if movement_distance > 100:
                return 'risky_pass'  # Long pass with no clear opportunities
            return 'forced_pass'
        
        best_score = max(opp['quality_score'] for opp in opportunities)
        
        if best_score > 0.7:
            return 'excellent_decision'
        elif best_score > 0.5:
            return 'good_decision'
        elif best_score > 0.3:
            return 'acceptable_decision'
        else:
            return 'poor_decision'
    
    def _analyze_player_vision(self, players: List[Dict], ball_data: List[Dict]) -> Dict[str, Any]:
        """Analyze player vision and awareness."""
        if not players:
            return {'vision_score': 0.0, 'awareness_level': 'poor'}
        
        vision_analysis = {}
        
        for player in players:
            player_id = player.get('player_id', 'unknown')
            player_pos = player.get('center', (0, 0))
            team = player.get('team', 'unknown')
            
            # Find teammates and opponents in vision radius
            teammates_in_vision = []
            opponents_in_vision = []
            
            for other_player in players:
                if other_player == player:
                    continue
                    
                other_pos = other_player.get('center', (0, 0))
                distance = math.sqrt((player_pos[0] - other_pos[0])**2 + (player_pos[1] - other_pos[1])**2)
                
                if distance < 150:  # Vision radius in pixels
                    if other_player.get('team') == team:
                        teammates_in_vision.append(other_player)
                    else:
                        opponents_in_vision.append(other_player)
            
            # Calculate vision score
            vision_score = self._calculate_vision_score(teammates_in_vision, opponents_in_vision)
            
            vision_analysis[player_id] = {
                'vision_score': vision_score,
                'teammates_visible': len(teammates_in_vision),
                'opponents_visible': len(opponents_in_vision),
                'awareness_level': self._categorize_awareness(vision_score)
            }
        
        return vision_analysis
    
    def _calculate_vision_score(self, teammates: List[Dict], opponents: List[Dict]) -> float:
        """Calculate vision score based on visible players."""
        # Good vision means seeing many teammates and being aware of opponents
        teammate_score = min(1.0, len(teammates) / 5)  # Normalize to 5 teammates
        opponent_awareness = min(1.0, len(opponents) / 3)  # Normalize to 3 opponents
        
        # Combine scores (weighted toward teammate awareness)
        return (teammate_score * 0.7 + opponent_awareness * 0.3)
    
    def _categorize_awareness(self, vision_score: float) -> str:
        """Categorize awareness level."""
        if vision_score > 0.8:
            return 'excellent'
        elif vision_score > 0.6:
            return 'good'
        elif vision_score > 0.4:
            return 'fair'
        else:
            return 'poor'
    
    def _analyze_defensive_play(self, players: List[Dict], ball_data: List[Dict]) -> Dict[str, Any]:
        """Analyze defensive positioning and pressing."""
        if not players or not ball_data:
            return {'press_intensity': 0.0, 'marking_effectiveness': 0.0}
        
        ball_pos = ball_data[0].get('center', (0, 0))
        ball_carrier = self._find_ball_carrier(players, ball_pos)
        
        if not ball_carrier:
            return {'press_intensity': 0.0, 'marking_effectiveness': 0.0}
        
        attacking_team = ball_carrier.get('team', 'unknown')
        defenders = [p for p in players if p.get('team') != attacking_team]
        attackers = [p for p in players if p.get('team') == attacking_team]
        
        # Analyze pressing
        press_intensity = self._analyze_pressing(defenders, ball_carrier)
        
        # Analyze marking
        marking_effectiveness = self._analyze_marking(defenders, attackers)
        
        return {
            'press_intensity': press_intensity,
            'marking_effectiveness': marking_effectiveness,
            'defenders_count': len(defenders),
            'attackers_count': len(attackers)
        }
    
    def _analyze_pressing(self, defenders: List[Dict], ball_carrier: Dict) -> float:
        """Analyze pressing intensity."""
        ball_pos = ball_carrier.get('center', (0, 0))
        pressing_defenders = 0
        total_press_distance = 0
        
        for defender in defenders:
            defender_pos = defender.get('center', (0, 0))
            distance = math.sqrt((ball_pos[0] - defender_pos[0])**2 + (ball_pos[1] - defender_pos[1])**2)
            
            if distance < 100:  # Press radius
                pressing_defenders += 1
                total_press_distance += distance
        
        if pressing_defenders == 0:
            return 0.0
        
        # Calculate press intensity (closer = more intense)
        avg_press_distance = total_press_distance / pressing_defenders
        press_intensity = max(0, 1 - avg_press_distance / 100)
        
        return press_intensity
    
    def _analyze_marking(self, defenders: List[Dict], attackers: List[Dict]) -> float:
        """Analyze marking effectiveness."""
        if not attackers:
            return 1.0
        
        marked_attackers = 0
        
        for attacker in attackers:
            attacker_pos = attacker.get('center', (0, 0))
            
            # Find closest defender
            min_distance = float('inf')
            for defender in defenders:
                defender_pos = defender.get('center', (0, 0))
                distance = math.sqrt((attacker_pos[0] - defender_pos[0])**2 + (attacker_pos[1] - defender_pos[1])**2)
                min_distance = min(min_distance, distance)
            
            # Consider attacker marked if defender is close enough
            if min_distance < 80:  # Marking distance threshold
                marked_attackers += 1
        
        return marked_attackers / len(attackers)
    
    def _analyze_team_positioning(self, players: List[Dict]) -> Dict[str, Any]:
        """Analyze team formation and positioning."""
        if len(players) < 4:
            return {'formation_compactness': 0.0, 'width_utilization': 0.0}
        
        # Group players by team
        teams = {}
        for player in players:
            team = player.get('team', 'unknown')
            if team not in teams:
                teams[team] = []
            teams[team].append(player)
        
        positioning_analysis = {}
        
        for team_name, team_players in teams.items():
            if len(team_players) < 3:
                continue
                
            positions = [p.get('center', (0, 0)) for p in team_players]
            
            # Calculate formation compactness
            compactness = self._calculate_formation_compactness(positions)
            
            # Calculate width utilization
            width_utilization = self._calculate_width_utilization(positions)
            
            positioning_analysis[team_name] = {
                'formation_compactness': compactness,
                'width_utilization': width_utilization,
                'player_count': len(team_players)
            }
        
        return positioning_analysis
    
    def _calculate_formation_compactness(self, positions: List[Tuple[int, int]]) -> float:
        """Calculate how compact the team formation is."""
        if len(positions) < 2:
            return 0.0
        
        # Calculate average distance between all players
        total_distance = 0
        pair_count = 0
        
        for i, pos1 in enumerate(positions):
            for j, pos2 in enumerate(positions):
                if i < j:
                    distance = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                    total_distance += distance
                    pair_count += 1
        
        avg_distance = total_distance / pair_count if pair_count > 0 else 0
        
        # Normalize compactness (closer = more compact)
        return max(0, 1 - avg_distance / 200)
    
    def _calculate_width_utilization(self, positions: List[Tuple[int, int]]) -> float:
        """Calculate how well the team utilizes field width."""
        if not positions:
            return 0.0
        
        x_positions = [pos[0] for pos in positions]
        x_range = max(x_positions) - min(x_positions)
        
        # Normalize width utilization
        return min(1.0, x_range / 400)  # Assume 400 pixels = good width utilization
    
    def _analyze_decision_making(self, players: List[Dict], ball_data: List[Dict], 
                               frame_idx: int) -> Dict[str, Any]:
        """Analyze decision making quality."""
        if not self.pass_events:
            return {'overall_decision_quality': 'insufficient_data'}
        
        recent_decisions = [event for event in self.pass_events if frame_idx - event['frame_idx'] < 300]
        
        if not recent_decisions:
            return {'overall_decision_quality': 'no_recent_decisions'}
        
        # Analyze recent decision quality
        decision_scores = {
            'excellent_decision': 1.0,
            'good_decision': 0.8,
            'acceptable_decision': 0.6,
            'poor_decision': 0.3,
            'risky_pass': 0.2,
            'forced_pass': 0.4
        }
        
        total_score = sum(decision_scores.get(event['decision_quality'], 0.5) for event in recent_decisions)
        avg_score = total_score / len(recent_decisions)
        
        if avg_score > 0.8:
            quality_level = 'excellent'
        elif avg_score > 0.6:
            quality_level = 'good'
        elif avg_score > 0.4:
            quality_level = 'fair'
        else:
            quality_level = 'poor'
        
        return {
            'overall_decision_quality': quality_level,
            'recent_decisions_count': len(recent_decisions),
            'average_decision_score': avg_score,
            'decision_breakdown': {decision: sum(1 for event in recent_decisions if event['decision_quality'] == decision) 
                                 for decision in decision_scores.keys()}
        }
    
    def get_tactical_summary(self) -> Dict[str, Any]:
        """Get comprehensive tactical analysis summary."""
        return {
            'total_pass_events': len(self.pass_events),
            'decision_patterns': self._analyze_decision_patterns(),
            'tactical_strengths': self._identify_tactical_strengths(),
            'areas_for_improvement': self._identify_areas_for_improvement()
        }
    
    def _analyze_decision_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in decision making."""
        if not self.pass_events:
            return {}
        
        decision_types = [event['decision_quality'] for event in self.pass_events]
        
        return {
            'most_common_decision': max(set(decision_types), key=decision_types.count),
            'decision_consistency': len(set(decision_types)),
            'improvement_trend': self._calculate_improvement_trend()
        }
    
    def _calculate_improvement_trend(self) -> str:
        """Calculate if decision making is improving over time."""
        if len(self.pass_events) < 10:
            return 'insufficient_data'
        
        # Compare first half vs second half of decisions
        mid_point = len(self.pass_events) // 2
        first_half = self.pass_events[:mid_point]
        second_half = self.pass_events[mid_point:]
        
        first_avg = np.mean([event.get('best_opportunity_score', 0) for event in first_half])
        second_avg = np.mean([event.get('best_opportunity_score', 0) for event in second_half])
        
        if second_avg > first_avg + 0.1:
            return 'improving'
        elif second_avg < first_avg - 0.1:
            return 'declining'
        else:
            return 'stable'
    
    def _identify_tactical_strengths(self) -> List[str]:
        """Identify tactical strengths."""
        strengths = []
        
        if len(self.pass_events) > 0:
            excellent_decisions = sum(1 for event in self.pass_events if event['decision_quality'] == 'excellent_decision')
            if excellent_decisions / len(self.pass_events) > 0.3:
                strengths.append("Excellent decision making under pressure")
        
        return strengths
    
    def _identify_areas_for_improvement(self) -> List[str]:
        """Identify areas needing improvement."""
        improvements = []
        
        if len(self.pass_events) > 0:
            poor_decisions = sum(1 for event in self.pass_events if event['decision_quality'] in ['poor_decision', 'risky_pass'])
            if poor_decisions / len(self.pass_events) > 0.3:
                improvements.append("Reduce risky passing decisions")
        
        return improvements