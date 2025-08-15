"""
FIFA-Quality Football Analysis System
Complete integration of 3D visualization, biomechanics, and tactical analysis.
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm

from ..visualization.pose_3d import Pose3DReconstructor, Field3DMapper
from ..visualization.renderer_3d import FIFA3DRenderer, CorrectiveFeedback3D
from ..analyzers.tactical_analyzer import TacticalAnalyzer
from ..analyzers import BallControlAnalyzer, PossessionAnalyzer
from ..analyzers.video_processor import VideoProcessor
from ..detectors import BallTracker, PoseDetector, PlayerDetector
from ..config import *

class FIFAAnalysisSystem:
    """
    FIFA-Quality Football Analysis System
    Comprehensive 3D analysis with biomechanics, tactics, and corrective feedback.
    """
    
    def __init__(self, max_frames: int = 1000, skip_frames: int = 1, resize_width: int = 640):
        """Initialize FIFA analysis system."""
        self.max_frames = max_frames
        self.skip_frames = skip_frames
        self.resize_width = resize_width
        
        # Initialize core detectors
        self.ball_tracker = BallTracker()
        self.pose_detector = PoseDetector()
        self.player_detector = PlayerDetector()
        
        # Initialize analyzers (video_processor will be initialized per video)
        self.video_processor = None
        self.ball_control_analyzer = BallControlAnalyzer()
        self.possession_analyzer = PossessionAnalyzer()
        self.tactical_analyzer = TacticalAnalyzer()
        
        # Initialize 3D components
        self.pose_3d_reconstructor = Pose3DReconstructor()
        self.field_3d_mapper = Field3DMapper()
        self.fifa_3d_renderer = FIFA3DRenderer()
        self.corrective_feedback = CorrectiveFeedback3D()
        
        # Analysis data storage
        self.frame_analyses = []
        self.biomech_analyses = []
        self.tactical_analyses = []
        self.corrective_feedbacks = []
        
        # Performance tracking
        self.fifa_metrics = {
            'total_frames_analyzed': 0,
            'pose_3d_reconstructions': 0,
            'biomech_evaluations': 0,
            'tactical_insights': 0,
            'corrective_recommendations': 0,
            'fifa_quality_score': 0.0
        }
    
    def analyze_video(self, video_path: Path, output_dir: Path = None) -> Dict[str, Any]:
        """
        Perform complete FIFA-quality video analysis.
        
        Args:
            video_path: Path to input video
            output_dir: Output directory for results
            
        Returns:
            Comprehensive analysis results
        """
        if output_dir is None:
            output_dir = Path("analysis_results")
        output_dir.mkdir(exist_ok=True)
        
        print(f"🏆 Starting FIFA-Quality Football Analysis...")
        print(f"📹 Video: {video_path.name}")
        print(f"🎯 Max frames: {self.max_frames}")
        print(f"📊 Skip frames: {self.skip_frames}")
        
        # Initialize video processor for this specific video
        self.video_processor = VideoProcessor(video_path)
        
        # Initialize video processing
        cap = cv2.VideoCapture(str(video_path))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize 3D scene
        field_data = self.field_3d_mapper.create_3d_field()
        fig, ax = self.fifa_3d_renderer.initialize_3d_scene(field_data)
        
        # Setup output video
        output_video_path = output_dir / f"fifa_3d_analysis_{video_path.stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        # Resize if needed
        if self.resize_width != width:
            aspect_ratio = height / width
            height = int(self.resize_width * aspect_ratio)
            width = self.resize_width
        
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps // self.skip_frames, (width, height))
        
        frame_idx = 0
        processed_frames = 0
        
        print("🔄 Processing video frames with FIFA-quality analysis...")
        
        with tqdm(total=min(self.max_frames, total_frames), desc="FIFA Analysis") as pbar:
            while cap.isOpened() and processed_frames < self.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames if needed
                if frame_idx % self.skip_frames != 0:
                    frame_idx += 1
                    continue
                
                # Resize frame
                if frame.shape[1] != width:
                    frame = cv2.resize(frame, (width, height))
                
                # Perform comprehensive analysis
                frame_analysis = self._analyze_single_frame(frame, frame_idx)
                
                # 3D reconstruction and biomechanical analysis
                biomech_analysis = self._perform_3d_biomech_analysis(frame_analysis, frame.shape)
                
                # Tactical analysis
                tactical_analysis = self.tactical_analyzer.analyze_frame(frame_analysis, frame_idx)
                
                # Generate corrective feedback
                corrective_feedback = self._generate_corrective_feedback(biomech_analysis, tactical_analysis)
                
                # Render 3D visualization
                annotated_frame = self._render_fifa_3d_frame(frame, frame_analysis, biomech_analysis, 
                                                           tactical_analysis, corrective_feedback)
                
                # Store analyses
                self.frame_analyses.append(frame_analysis)
                self.biomech_analyses.append(biomech_analysis)
                self.tactical_analyses.append(tactical_analysis)
                self.corrective_feedbacks.append(corrective_feedback)
                
                # Write frame
                out.write(annotated_frame)
                
                # Update metrics
                self._update_fifa_metrics(frame_analysis, biomech_analysis, tactical_analysis)
                
                processed_frames += 1
                frame_idx += 1
                pbar.update(1)
        
        # Cleanup
        cap.release()
        out.release()
        
        # Generate comprehensive FIFA report
        fifa_report = self._generate_fifa_report(video_path, output_dir)
        
        # Create 3D animation
        animation_path = output_dir / f"fifa_3d_animation_{video_path.stem}.mp4"
        self.fifa_3d_renderer.create_3d_animation(str(animation_path), fps=fps//self.skip_frames)
        
        print(f"✅ FIFA-Quality Analysis Complete!")
        print(f"📹 Enhanced video: {output_video_path}")
        print(f"🎬 3D animation: {animation_path}")
        print(f"📊 FIFA report: {fifa_report}")
        
        return {
            'video_path': str(output_video_path),
            'animation_path': str(animation_path),
            'report_path': fifa_report,
            'fifa_metrics': self.fifa_metrics,
            'analysis_summary': self._get_analysis_summary()
        }
    
    def _analyze_single_frame(self, frame: np.ndarray, frame_idx: int) -> Dict[str, Any]:
        """Analyze single frame with all detectors."""
        frame_data = {'frame_idx': frame_idx}
        
        # Ball detection
        ball_detections = self.ball_tracker.detect_ball(frame)
        frame_data['ball_detections'] = ball_detections
        
        # Pose detection
        pose_landmarks, pose_results = self.pose_detector.detect_pose(frame)
        pose_detections = [{'landmarks': pose_landmarks, 'results': pose_results}] if pose_landmarks else []
        frame_data['pose_detections'] = pose_detections
        
        # Player detection
        player_detections = self.player_detector.detect_players(frame)
        player_detections = self.player_detector.assign_teams(frame, player_detections)
        frame_data['player_detections'] = player_detections
        
        # Ball control analysis (simplified for FIFA system)
        ball_control_data = {
            'frame_idx': frame_idx,
            'ball_detected': len(ball_detections) > 0,
            'pose_detected': len(pose_detections) > 0,
            'total_touches': self.ball_control_analyzer.performance_metrics['total_touches']
        }
        frame_data['ball_control'] = ball_control_data
        
        # Possession analysis
        possession_data = self.possession_analyzer.analyze_possession(
            ball_detections, player_detections, frame_idx
        )
        frame_data['possession'] = possession_data
        
        return frame_data
    
    def _perform_3d_biomech_analysis(self, frame_analysis: Dict[str, Any], 
                                   frame_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Perform 3D pose reconstruction and biomechanical analysis."""
        biomech_data = {}
        
        pose_detections = frame_analysis.get('pose_detections', [])
        ball_detections = frame_analysis.get('ball_detections', [])
        
        # Get ball position for shooting analysis
        ball_position_3d = None
        if ball_detections:
            ball_2d = ball_detections[0].get('center', (0, 0))
            ball_position_3d = self.field_3d_mapper.map_player_to_field(
                np.array(ball_2d), frame_shape
            )
        
        # Process each detected pose
        for i, pose_data in enumerate(pose_detections):
            landmarks_2d = pose_data.get('landmarks')
            if landmarks_2d is None:
                continue
            
            # Convert MediaPipe normalized coordinates to pixel coordinates
            landmarks_2d_pixels = np.array([
                [lm.x * frame_shape[1], lm.y * frame_shape[0]] 
                for lm in landmarks_2d.landmark
            ])
            
            # 3D reconstruction
            pose_3d = self.pose_3d_reconstructor.reconstruct_3d_pose(
                landmarks_2d_pixels, frame_shape
            )
            
            if pose_3d is not None:
                # Biomechanical analysis
                biomech_analysis = self.pose_3d_reconstructor.analyze_body_mechanics(
                    pose_3d, ball_position_3d
                )
                
                biomech_data[f'pose_{i}'] = {
                    'pose_3d': pose_3d.tolist(),
                    'biomech_analysis': biomech_analysis,
                    'landmarks_2d': landmarks_2d_pixels.tolist()
                }
        
        return biomech_data
    
    def _generate_corrective_feedback(self, biomech_analysis: Dict[str, Any], 
                                    tactical_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate FIFA-quality corrective feedback."""
        all_feedback = []
        
        # Biomechanical feedback
        for pose_key, pose_data in biomech_analysis.items():
            biomech_data = pose_data.get('biomech_analysis', {})
            if biomech_data:
                feedback = self.corrective_feedback.generate_corrective_feedback(biomech_data)
                all_feedback.extend(feedback)
        
        # Tactical feedback
        passing_data = tactical_analysis.get('passing', {})
        if passing_data:
            quality_score = passing_data.get('quality_score', 0)
            if quality_score < 0.5:
                all_feedback.append({
                    'type': 'tactical',
                    'rating': 'poor' if quality_score < 0.3 else 'fair',
                    'message': f"Improve passing options awareness. Quality score: {quality_score:.2f}",
                    'correction_vector': np.array([0, 0, 0.5])  # Look up for better vision
                })
        
        return all_feedback
    
    def _render_fifa_3d_frame(self, frame: np.ndarray, frame_analysis: Dict[str, Any],
                            biomech_analysis: Dict[str, Any], tactical_analysis: Dict[str, Any],
                            corrective_feedback: List[Dict[str, Any]]) -> np.ndarray:
        """Render FIFA-quality 3D visualization frame."""
        # Clear previous 3D content
        self.fifa_3d_renderer.ax.clear()
        
        # Redraw field
        field_data = self.field_3d_mapper.create_3d_field()
        self.fifa_3d_renderer._draw_3d_field(field_data)
        
        # Render players with 3D poses
        player_detections = frame_analysis.get('player_detections', [])
        for i, player in enumerate(player_detections):
            player_id = player.get('player_id', f'P{i+1}')
            team = player.get('team', 'unknown')
            
            # Check if we have 3D pose for this player
            pose_key = f'pose_{i}' if f'pose_{i}' in biomech_analysis else None
            if pose_key:
                pose_data = biomech_analysis[pose_key]
                pose_3d = np.array(pose_data['pose_3d'])
                biomech_data = pose_data['biomech_analysis']
                
                # Map to field coordinates
                player_pos_2d = player.get('center', (0, 0))
                field_pos = self.field_3d_mapper.map_player_to_field(
                    np.array(player_pos_2d), frame.shape[:2]
                )
                
                # Adjust 3D pose to field position
                pose_3d_adjusted = pose_3d + field_pos
                
                # Render 3D pose
                self.fifa_3d_renderer.render_3d_pose(
                    pose_3d_adjusted, player_id, team, biomech_data
                )
        
        # Render ball in 3D
        ball_detections = frame_analysis.get('ball_detections', [])
        if ball_detections:
            ball_2d = ball_detections[0].get('center', (0, 0))
            ball_3d = self.field_3d_mapper.map_player_to_field(
                np.array(ball_2d), frame.shape[:2]
            )
            ball_3d[2] = 0.1  # Ball height
            self.fifa_3d_renderer.render_ball_3d(ball_3d)
        
        # Add corrective feedback
        for feedback in corrective_feedback:
            if 'correction_vector' in feedback:
                # Use first player position for feedback (simplified)
                if player_detections:
                    player_pos_2d = player_detections[0].get('center', (0, 0))
                    field_pos = self.field_3d_mapper.map_player_to_field(
                        np.array(player_pos_2d), frame.shape[:2]
                    )
                    
                    self.fifa_3d_renderer.add_corrective_feedback(
                        field_pos, feedback['type'], feedback['message'],
                        feedback.get('correction_vector')
                    )
        
        # Add performance metrics
        ball_control = frame_analysis.get('ball_control', {})
        metrics = {
            'ball_touches': ball_control.get('total_touches', 0),
            'touch_frequency': ball_control.get('touch_frequency', 0),
        }
        self.fifa_3d_renderer.add_performance_metrics(metrics)
        
        # Add coaching insights
        insights = self._generate_frame_insights(tactical_analysis, biomech_analysis)
        self.fifa_3d_renderer.add_coaching_insights(insights)
        
        # Save 3D frame
        frame_idx = frame_analysis.get('frame_idx', 0)
        self.fifa_3d_renderer.save_3d_frame(frame_idx, {
            'biomech': biomech_analysis,
            'tactical': tactical_analysis,
            'feedback': corrective_feedback
        })
        
        # Convert matplotlib figure to OpenCV image
        self.fifa_3d_renderer.fig.canvas.draw()
        img = np.frombuffer(self.fifa_3d_renderer.fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(self.fifa_3d_renderer.fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Resize to match output frame size
        img = cv2.resize(img, (frame.shape[1], frame.shape[0]))
        
        return img
    
    def _generate_frame_insights(self, tactical_analysis: Dict[str, Any], 
                               biomech_analysis: Dict[str, Any]) -> List[str]:
        """Generate frame-specific coaching insights."""
        insights = []
        
        # Tactical insights
        passing_data = tactical_analysis.get('passing', {})
        if passing_data:
            quality_score = passing_data.get('quality_score', 0)
            if quality_score > 0.8:
                insights.append("Excellent passing opportunities available")
            elif quality_score < 0.3:
                insights.append("Limited passing options - improve positioning")
        
        # Biomechanical insights
        for pose_key, pose_data in biomech_analysis.items():
            biomech_data = pose_data.get('biomech_analysis', {})
            
            if 'body_lean' in biomech_data:
                lean_angle = biomech_data['body_lean']['lean_angle']
                if lean_angle > 20:
                    insights.append("Improve posture - excessive body lean detected")
            
            if 'balance' in biomech_data:
                balance_score = biomech_data['balance']['balance_score']
                if balance_score > 0.8:
                    insights.append("Excellent balance and stability")
                elif balance_score < 0.4:
                    insights.append("Focus on balance - unstable stance")
        
        return insights[:3]  # Limit to top 3 insights
    
    def _update_fifa_metrics(self, frame_analysis: Dict[str, Any], 
                           biomech_analysis: Dict[str, Any], 
                           tactical_analysis: Dict[str, Any]):
        """Update FIFA quality metrics."""
        self.fifa_metrics['total_frames_analyzed'] += 1
        
        if biomech_analysis:
            self.fifa_metrics['pose_3d_reconstructions'] += len(biomech_analysis)
            self.fifa_metrics['biomech_evaluations'] += len(biomech_analysis)
        
        if tactical_analysis.get('passing'):
            self.fifa_metrics['tactical_insights'] += 1
        
        if self.corrective_feedbacks and self.corrective_feedbacks[-1]:
            self.fifa_metrics['corrective_recommendations'] += len(self.corrective_feedbacks[-1])
    
    def _generate_fifa_report(self, video_path: Path, output_dir: Path) -> str:
        """Generate comprehensive FIFA-quality analysis report."""
        # Calculate FIFA quality score
        self.fifa_metrics['fifa_quality_score'] = self._calculate_fifa_quality_score()
        
        # Generate tactical summary
        tactical_summary = self.tactical_analyzer.get_tactical_summary()
        
        # Generate biomechanical summary
        biomech_summary = self._generate_biomech_summary()
        
        # Create comprehensive report
        fifa_report = {
            'video_info': {
                'filename': video_path.name,
                'total_frames_analyzed': self.fifa_metrics['total_frames_analyzed'],
                'analysis_type': 'FIFA-Quality Professional Football Analysis'
            },
            'fifa_metrics': self.fifa_metrics,
            'biomechanical_analysis': biomech_summary,
            'tactical_analysis': tactical_summary,
            'corrective_feedback_summary': self._generate_feedback_summary(),
            '3d_visualization': {
                'total_3d_reconstructions': self.fifa_metrics['pose_3d_reconstructions'],
                'field_mapping': 'FIFA-standard 105x68m field projection',
                'corrective_models': 'Professional 3D feedback generation'
            },
            'fifa_compliance': {
                '3d_visualization': True,
                'biomechanical_analysis': True,
                'tactical_evaluation': True,
                'corrective_feedback': True,
                'professional_quality': self.fifa_metrics['fifa_quality_score'] > 0.7
            }
        }
        
        # Save report
        report_path = output_dir / "fifa_professional_analysis_report.json"
        with open(report_path, 'w') as f:
            json.dump(fifa_report, f, indent=2)
        
        return str(report_path)
    
    def _calculate_fifa_quality_score(self) -> float:
        """Calculate overall FIFA quality score."""
        if self.fifa_metrics['total_frames_analyzed'] == 0:
            return 0.0
        
        # Component scores
        pose_3d_ratio = min(1.0, self.fifa_metrics['pose_3d_reconstructions'] / self.fifa_metrics['total_frames_analyzed'])
        biomech_ratio = min(1.0, self.fifa_metrics['biomech_evaluations'] / self.fifa_metrics['total_frames_analyzed'])
        tactical_ratio = min(1.0, self.fifa_metrics['tactical_insights'] / self.fifa_metrics['total_frames_analyzed'])
        feedback_ratio = min(1.0, self.fifa_metrics['corrective_recommendations'] / (self.fifa_metrics['total_frames_analyzed'] * 2))
        
        # Weighted FIFA quality score
        fifa_score = (pose_3d_ratio * 0.3 + biomech_ratio * 0.3 + tactical_ratio * 0.25 + feedback_ratio * 0.15)
        
        return fifa_score
    
    def _generate_biomech_summary(self) -> Dict[str, Any]:
        """Generate biomechanical analysis summary."""
        if not self.biomech_analyses:
            return {}
        
        all_lean_angles = []
        all_balance_scores = []
        shooting_events = 0
        
        for analysis in self.biomech_analyses:
            for pose_key, pose_data in analysis.items():
                biomech_data = pose_data.get('biomech_analysis', {})
                
                if 'body_lean' in biomech_data:
                    all_lean_angles.append(biomech_data['body_lean']['lean_angle'])
                
                if 'balance' in biomech_data:
                    all_balance_scores.append(biomech_data['balance']['balance_score'])
                
                if 'shooting_mechanics' in biomech_data:
                    shooting_events += 1
        
        return {
            'average_body_lean': np.mean(all_lean_angles) if all_lean_angles else 0,
            'average_balance_score': np.mean(all_balance_scores) if all_balance_scores else 0,
            'total_shooting_events': shooting_events,
            'posture_quality': 'excellent' if np.mean(all_lean_angles) < 10 else 'needs_improvement',
            'balance_quality': 'excellent' if np.mean(all_balance_scores) > 0.8 else 'needs_improvement'
        }
    
    def _generate_feedback_summary(self) -> Dict[str, Any]:
        """Generate corrective feedback summary."""
        all_feedback = [fb for frame_fb in self.corrective_feedbacks for fb in frame_fb]
        
        if not all_feedback:
            return {}
        
        feedback_types = {}
        ratings = {}
        
        for feedback in all_feedback:
            fb_type = feedback.get('type', 'unknown')
            rating = feedback.get('rating', 'unknown')
            
            feedback_types[fb_type] = feedback_types.get(fb_type, 0) + 1
            ratings[rating] = ratings.get(rating, 0) + 1
        
        return {
            'total_feedback_items': len(all_feedback),
            'feedback_by_type': feedback_types,
            'feedback_by_rating': ratings,
            'most_common_issue': max(feedback_types.items(), key=lambda x: x[1])[0] if feedback_types else None
        }
    
    def _get_analysis_summary(self) -> Dict[str, Any]:
        """Get comprehensive analysis summary."""
        return {
            'frames_analyzed': len(self.frame_analyses),
            'biomech_analyses': len(self.biomech_analyses),
            'tactical_analyses': len(self.tactical_analyses),
            'corrective_feedbacks': len(self.corrective_feedbacks),
            'fifa_quality_score': self.fifa_metrics['fifa_quality_score'],
            'analysis_complete': True,
            'fifa_compliance': self.fifa_metrics['fifa_quality_score'] > 0.7
        }
    
    def cleanup(self):
        """Cleanup resources."""
        self.fifa_3d_renderer.cleanup()
        plt.close('all')