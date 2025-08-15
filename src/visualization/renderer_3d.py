"""
FIFA-Quality 3D Visualization Renderer
Professional 3D rendering for football analysis with corrective feedback.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle
import cv2
from typing import List, Dict, Any, Tuple, Optional
import json
from pathlib import Path

class FIFA3DRenderer:
    """FIFA-quality 3D visualization renderer for football analysis."""
    
    def __init__(self, output_dir: str = "analysis_results"):
        """Initialize 3D renderer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # FIFA color scheme
        self.colors = {
            'field': '#1e8b3e',  # FIFA green
            'lines': '#ffffff',   # White lines
            'team_1': '#ff0000',  # Red team
            'team_2': '#0000ff',  # Blue team
            'referee': '#ffff00', # Yellow referee
            'ball': '#ffffff',    # White ball
            'corrective': '#ff6600',  # Orange for corrections
            'excellent': '#00ff00',   # Green for good technique
            'poor': '#ff0000'     # Red for poor technique
        }
        
        # 3D scene setup
        self.fig = None
        self.ax = None
        self.frame_data = []
        
    def initialize_3d_scene(self, field_data: Dict[str, np.ndarray]) -> Tuple[plt.Figure, Axes3D]:
        """Initialize 3D scene with FIFA-standard field."""
        self.fig = plt.figure(figsize=(16, 12))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Set FIFA field appearance
        self.ax.set_facecolor('black')
        self.fig.patch.set_facecolor('black')
        
        # Draw field
        self._draw_3d_field(field_data)
        
        # Set viewing angle for optimal visualization
        self.ax.view_init(elev=25, azim=45)
        
        # Set axis properties
        self.ax.set_xlim(-60, 60)
        self.ax.set_ylim(-40, 40)
        self.ax.set_zlim(0, 3)
        
        # Remove axis for clean look
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])
        
        # Add professional title
        self.fig.suptitle('FIFA-Quality Football Analysis - 3D Biomechanical Evaluation', 
                         fontsize=16, color='white', fontweight='bold')
        
        return self.fig, self.ax
    
    def _draw_3d_field(self, field_data: Dict[str, np.ndarray]):
        """Draw FIFA-standard 3D field."""
        # Field surface
        x = [-52.5, 52.5, 52.5, -52.5, -52.5]
        y = [-34, -34, 34, 34, -34]
        z = [0, 0, 0, 0, 0]
        self.ax.plot(x, y, z, color=self.colors['lines'], linewidth=2)
        
        # Center line
        self.ax.plot([0, 0], [-34, 34], [0, 0], color=self.colors['lines'], linewidth=2)
        
        # Center circle
        theta = np.linspace(0, 2*np.pi, 64)
        circle_x = 9.15 * np.cos(theta)
        circle_y = 9.15 * np.sin(theta)
        circle_z = np.zeros_like(circle_x)
        self.ax.plot(circle_x, circle_y, circle_z, color=self.colors['lines'], linewidth=1.5)
        
        # Penalty areas
        penalty_areas = [
            # Left penalty area
            [[-52.5, -20.15, 0], [-36, -20.15, 0], [-36, 20.15, 0], [-52.5, 20.15, 0], [-52.5, -20.15, 0]],
            # Right penalty area
            [[52.5, -20.15, 0], [36, -20.15, 0], [36, 20.15, 0], [52.5, 20.15, 0], [52.5, -20.15, 0]]
        ]
        
        for area in penalty_areas:
            area_array = np.array(area)
            self.ax.plot(area_array[:, 0], area_array[:, 1], area_array[:, 2], 
                        color=self.colors['lines'], linewidth=1.5)
        
        # Goal areas
        goal_areas = [
            # Left goal area
            [[-52.5, -9.15, 0], [-47, -9.15, 0], [-47, 9.15, 0], [-52.5, 9.15, 0], [-52.5, -9.15, 0]],
            # Right goal area
            [[52.5, -9.15, 0], [47, -9.15, 0], [47, 9.15, 0], [52.5, 9.15, 0], [52.5, -9.15, 0]]
        ]
        
        for area in goal_areas:
            area_array = np.array(area)
            self.ax.plot(area_array[:, 0], area_array[:, 1], area_array[:, 2], 
                        color=self.colors['lines'], linewidth=1.5)
        
        # Goals
        goal_posts = [
            # Left goal
            [[-52.5, -3.66, 0], [-52.5, -3.66, 2.44], [-52.5, 3.66, 2.44], [-52.5, 3.66, 0]],
            # Right goal
            [[52.5, -3.66, 0], [52.5, -3.66, 2.44], [52.5, 3.66, 2.44], [52.5, 3.66, 0]]
        ]
        
        for goal in goal_posts:
            goal_array = np.array(goal)
            self.ax.plot(goal_array[:, 0], goal_array[:, 1], goal_array[:, 2], 
                        color=self.colors['lines'], linewidth=3)
    
    def render_3d_pose(self, pose_3d: np.ndarray, player_id: str, team: str, 
                      biomech_analysis: Dict[str, Any]) -> None:
        """Render 3D pose with biomechanical analysis."""
        if pose_3d is None or len(pose_3d) == 0:
            return
            
        # Get team color
        color = self.colors.get(team, self.colors['team_1'])
        
        # Draw skeleton
        self._draw_3d_skeleton(pose_3d, color)
        
        # Draw biomechanical indicators
        self._draw_biomech_indicators(pose_3d, biomech_analysis, player_id)
        
        # Add player label
        if len(pose_3d) > 0:
            head_pos = pose_3d[0] if len(pose_3d) > 0 else [0, 0, 2]
            self.ax.text(head_pos[0], head_pos[1], head_pos[2] + 0.3, 
                        f'{player_id}', fontsize=12, color=color, fontweight='bold')
    
    def _draw_3d_skeleton(self, pose_3d: np.ndarray, color: str):
        """Draw 3D skeleton from pose landmarks."""
        # Define pose connections for 3D visualization
        connections = [
            # Head and neck
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
            # Arms
            (11, 13), (13, 15), (12, 14), (14, 16),
            # Torso
            (11, 12), (11, 23), (12, 24), (23, 24),
            # Legs
            (23, 25), (25, 27), (27, 29), (24, 26), (26, 28), (28, 30)
        ]
        
        # Draw connections
        for start_idx, end_idx in connections:
            if start_idx < len(pose_3d) and end_idx < len(pose_3d):
                start_pos = pose_3d[start_idx]
                end_pos = pose_3d[end_idx]
                
                self.ax.plot([start_pos[0], end_pos[0]], 
                           [start_pos[1], end_pos[1]], 
                           [start_pos[2], end_pos[2]], 
                           color=color, linewidth=2, alpha=0.8)
        
        # Draw joints as spheres
        for i, joint in enumerate(pose_3d):
            self.ax.scatter(joint[0], joint[1], joint[2], 
                          color=color, s=30, alpha=0.9)
    
    def _draw_biomech_indicators(self, pose_3d: np.ndarray, analysis: Dict[str, Any], player_id: str):
        """Draw biomechanical analysis indicators."""
        if not analysis:
            return
            
        # Body lean indicator
        if 'body_lean' in analysis:
            lean_data = analysis['body_lean']
            lean_angle = lean_data.get('lean_angle', 0)
            
            # Color code based on lean angle (FIFA coaching standards)
            if lean_angle < 10:
                lean_color = self.colors['excellent']  # Good posture
            elif lean_angle < 20:
                lean_color = '#ffaa00'  # Acceptable
            else:
                lean_color = self.colors['poor']  # Poor posture
                
            # Draw lean indicator
            if len(pose_3d) > 24:
                hip_center = (pose_3d[23] + pose_3d[24]) / 2
                self.ax.text(hip_center[0] + 1, hip_center[1], hip_center[2] + 0.5,
                           f'Lean: {lean_angle:.1f}°', fontsize=10, color=lean_color, fontweight='bold')
        
        # Balance indicator
        if 'balance' in analysis:
            balance_score = analysis['balance']['balance_score']
            
            # Color code balance
            if balance_score > 0.8:
                balance_color = self.colors['excellent']
            elif balance_score > 0.6:
                balance_color = '#ffaa00'
            else:
                balance_color = self.colors['poor']
                
            # Draw balance indicator
            if len(pose_3d) > 24:
                hip_center = (pose_3d[23] + pose_3d[24]) / 2
                self.ax.text(hip_center[0] - 1, hip_center[1], hip_center[2] + 0.5,
                           f'Balance: {balance_score:.2f}', fontsize=10, color=balance_color, fontweight='bold')
        
        # Shooting mechanics indicator
        if 'shooting_mechanics' in analysis:
            shoot_data = analysis['shooting_mechanics']
            if shoot_data.get('contact_imminent', False):
                contact_dist = shoot_data.get('contact_distance', 0)
                approach_angle = shoot_data.get('approach_angle', 0)
                
                # Draw shooting analysis
                if len(pose_3d) > 32:
                    foot_pos = pose_3d[32] if shoot_data.get('shooting_foot') == 'right' else pose_3d[31]
                    self.ax.text(foot_pos[0], foot_pos[1], foot_pos[2] + 0.3,
                               f'Shot: {approach_angle:.0f}°', fontsize=10, 
                               color=self.colors['corrective'], fontweight='bold')
    
    def render_ball_3d(self, ball_position: np.ndarray, trajectory: List[np.ndarray] = None):
        """Render 3D ball with trajectory."""
        if ball_position is not None:
            # Draw ball
            self.ax.scatter(ball_position[0], ball_position[1], ball_position[2], 
                          color=self.colors['ball'], s=100, alpha=0.9, edgecolors='black')
            
            # Draw trajectory if available
            if trajectory and len(trajectory) > 1:
                traj_array = np.array(trajectory)
                self.ax.plot(traj_array[:, 0], traj_array[:, 1], traj_array[:, 2], 
                           color=self.colors['ball'], linewidth=2, alpha=0.6, linestyle='--')
    
    def add_corrective_feedback(self, player_pos: np.ndarray, feedback_type: str, 
                              feedback_text: str, correction_vector: np.ndarray = None):
        """Add 3D corrective feedback visualization."""
        # Add feedback text
        self.ax.text(player_pos[0], player_pos[1], player_pos[2] + 2,
                    feedback_text, fontsize=12, color=self.colors['corrective'], 
                    fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", 
                    facecolor='black', alpha=0.7))
        
        # Add correction arrow if vector provided
        if correction_vector is not None:
            end_pos = player_pos + correction_vector
            self.ax.quiver(player_pos[0], player_pos[1], player_pos[2],
                         correction_vector[0], correction_vector[1], correction_vector[2],
                         color=self.colors['corrective'], arrow_length_ratio=0.1, linewidth=3)
    
    def add_performance_metrics(self, metrics: Dict[str, Any]):
        """Add performance metrics overlay."""
        metrics_text = []
        
        if 'ball_touches' in metrics:
            metrics_text.append(f"Ball Touches: {metrics['ball_touches']}")
        if 'touch_frequency' in metrics:
            metrics_text.append(f"Touch Frequency: {metrics['touch_frequency']:.1f}/min")
        if 'average_speed' in metrics:
            metrics_text.append(f"Average Speed: {metrics['average_speed']:.1f} km/h")
        if 'distance_covered' in metrics:
            metrics_text.append(f"Distance: {metrics['distance_covered']:.0f}m")
        
        # Add metrics panel
        metrics_str = "\\n".join(metrics_text)
        self.ax.text2D(0.02, 0.98, metrics_str, transform=self.ax.transAxes,
                      fontsize=11, color='white', fontweight='bold',
                      verticalalignment='top',
                      bbox=dict(boxstyle="round,pad=0.5", facecolor='black', alpha=0.8))
    
    def add_coaching_insights(self, insights: List[str]):
        """Add FIFA-quality coaching insights."""
        insights_text = "COACHING INSIGHTS:\\n" + "\\n".join([f"• {insight}" for insight in insights])
        
        self.ax.text2D(0.02, 0.5, insights_text, transform=self.ax.transAxes,
                      fontsize=10, color='white', fontweight='bold',
                      verticalalignment='top',
                      bbox=dict(boxstyle="round,pad=0.5", facecolor='darkblue', alpha=0.8))
    
    def save_3d_frame(self, frame_idx: int, analysis_data: Dict[str, Any]):
        """Save current 3D frame."""
        self.frame_data.append({
            'frame_idx': frame_idx,
            'analysis_data': analysis_data.copy()
        })
        
        # Save static frame
        frame_path = self.output_dir / f"3d_frame_{frame_idx:06d}.png"
        self.fig.savefig(frame_path, dpi=150, bbox_inches='tight', 
                        facecolor='black', edgecolor='none')
    
    def create_3d_animation(self, output_path: str, fps: int = 30):
        """Create 3D animation video."""
        if not self.frame_data:
            print("No frame data available for animation")
            return
            
        def animate(frame_idx):
            """Animation function."""
            self.ax.clear()
            
            # Redraw field for each frame
            self._draw_3d_field({})
            
            # Get frame data
            if frame_idx < len(self.frame_data):
                frame_data = self.frame_data[frame_idx]
                analysis_data = frame_data['analysis_data']
                
                # Render poses and analysis for this frame
                # (This would be populated with actual frame data)
                
            # Set title with frame info
            self.ax.set_title(f'Frame {frame_idx + 1}/{len(self.frame_data)}', 
                            color='white', fontsize=14)
            
            return []
        
        # Create animation
        anim = animation.FuncAnimation(self.fig, animate, frames=len(self.frame_data),
                                     interval=1000//fps, blit=False, repeat=True)
        
        # Save animation
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='FIFA Football Analysis'), bitrate=1800)
        anim.save(output_path, writer=writer)
        
        print(f"3D animation saved to: {output_path}")
    
    def generate_3d_report(self, analysis_summary: Dict[str, Any]) -> str:
        """Generate comprehensive 3D analysis report."""
        report_path = self.output_dir / "fifa_3d_analysis_report.json"
        
        # Enhanced report with 3D metrics
        enhanced_summary = analysis_summary.copy()
        enhanced_summary['3d_analysis'] = {
            'total_frames_analyzed': len(self.frame_data),
            'biomechanical_insights': {
                'posture_analysis': 'FIFA-standard biomechanical evaluation',
                'balance_assessment': 'Dynamic stability analysis',
                'movement_quality': '3D movement pattern evaluation'
            },
            '3d_visualization': {
                'field_mapping': 'FIFA-standard 3D field projection',
                'pose_reconstruction': '3D pose from MediaPipe landmarks',
                'corrective_feedback': '3D visual corrections and improvements'
            }
        }
        
        # Save report
        with open(report_path, 'w') as f:
            json.dump(enhanced_summary, f, indent=2)
            
        return str(report_path)
    
    def cleanup(self):
        """Clean up resources."""
        if self.fig:
            plt.close(self.fig)
        self.frame_data.clear()

class CorrectiveFeedback3D:
    """FIFA-quality 3D corrective feedback generator."""
    
    def __init__(self):
        """Initialize corrective feedback system."""
        self.feedback_templates = {
            'posture': {
                'excellent': "Excellent posture! Maintain this balanced stance.",
                'good': "Good posture. Minor adjustments recommended.",
                'fair': "Moderate posture issues. Focus on core stability.",
                'poor': "Poor posture detected. Major corrections needed."
            },
            'balance': {
                'excellent': "Perfect balance! Great stability.",
                'good': "Good balance. Slightly adjust foot positioning.",
                'fair': "Balance needs improvement. Widen stance.",
                'poor': "Poor balance. Focus on core strength and foot placement."
            },
            'shooting': {
                'excellent': "Excellent shooting technique! Perfect approach angle.",
                'good': "Good shooting form. Minor angle adjustment recommended.",
                'fair': "Shooting technique needs work. Adjust body position.",
                'poor': "Poor shooting technique. Major form corrections needed."
            }
        }
    
    def generate_corrective_feedback(self, biomech_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate FIFA-quality corrective feedback."""
        feedback_list = []
        
        # Posture feedback
        if 'body_lean' in biomech_analysis:
            lean_angle = biomech_analysis['body_lean']['lean_angle']
            posture_rating = self._rate_posture(lean_angle)
            
            feedback_list.append({
                'type': 'posture',
                'rating': posture_rating,
                'message': self.feedback_templates['posture'][posture_rating],
                'correction_vector': self._get_posture_correction(biomech_analysis['body_lean'])
            })
        
        # Balance feedback
        if 'balance' in biomech_analysis:
            balance_score = biomech_analysis['balance']['balance_score']
            balance_rating = self._rate_balance(balance_score)
            
            feedback_list.append({
                'type': 'balance',
                'rating': balance_rating,
                'message': self.feedback_templates['balance'][balance_rating],
                'correction_vector': self._get_balance_correction(biomech_analysis['balance'])
            })
        
        # Shooting feedback
        if 'shooting_mechanics' in biomech_analysis:
            shooting_data = biomech_analysis['shooting_mechanics']
            if shooting_data.get('contact_imminent', False):
                shooting_rating = self._rate_shooting(shooting_data)
                
                feedback_list.append({
                    'type': 'shooting',
                    'rating': shooting_rating,
                    'message': self.feedback_templates['shooting'][shooting_rating],
                    'correction_vector': self._get_shooting_correction(shooting_data)
                })
        
        return feedback_list
    
    def _rate_posture(self, lean_angle: float) -> str:
        """Rate posture based on lean angle."""
        if lean_angle < 5:
            return 'excellent'
        elif lean_angle < 10:
            return 'good'
        elif lean_angle < 20:
            return 'fair'
        else:
            return 'poor'
    
    def _rate_balance(self, balance_score: float) -> str:
        """Rate balance based on balance score."""
        if balance_score > 0.9:
            return 'excellent'
        elif balance_score > 0.7:
            return 'good'
        elif balance_score > 0.5:
            return 'fair'
        else:
            return 'poor'
    
    def _rate_shooting(self, shooting_data: Dict[str, Any]) -> str:
        """Rate shooting technique."""
        approach_angle = abs(shooting_data.get('approach_angle', 0))
        
        if approach_angle < 15:
            return 'excellent'
        elif approach_angle < 30:
            return 'good'
        elif approach_angle < 45:
            return 'fair'
        else:
            return 'poor'
    
    def _get_posture_correction(self, lean_data: Dict[str, Any]) -> np.ndarray:
        """Get posture correction vector."""
        lean_angle = lean_data['lean_angle']
        lean_direction = lean_data['lean_direction']
        
        # Simple correction vector (opposite to lean)
        correction_magnitude = min(lean_angle / 20.0, 1.0)
        
        if lean_direction == 'left':
            return np.array([correction_magnitude, 0, 0])
        elif lean_direction == 'right':
            return np.array([-correction_magnitude, 0, 0])
        elif lean_direction == 'forward':
            return np.array([0, -correction_magnitude, 0])
        elif lean_direction == 'backward':
            return np.array([0, correction_magnitude, 0])
        
        return np.array([0, 0, 0])
    
    def _get_balance_correction(self, balance_data: Dict[str, Any]) -> np.ndarray:
        """Get balance correction vector."""
        horizontal_offset = balance_data.get('horizontal_offset', 0)
        
        # Correction toward center
        correction_magnitude = min(horizontal_offset / 0.5, 1.0)
        return np.array([0, 0, correction_magnitude])
    
    def _get_shooting_correction(self, shooting_data: Dict[str, Any]) -> np.ndarray:
        """Get shooting correction vector."""
        approach_angle = shooting_data.get('approach_angle', 0)
        
        # Correction toward optimal angle (0 degrees)
        correction_magnitude = min(abs(approach_angle) / 45.0, 1.0)
        correction_direction = -1 if approach_angle > 0 else 1
        
        return np.array([0, correction_direction * correction_magnitude, 0])