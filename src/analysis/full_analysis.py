"""
Full video analysis module that orchestrates the complete football analysis pipeline.
"""

import cv2
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

from ..config import (
    VIDEO_DIR, OUTPUT_DIR, MAX_FRAMES, SKIP_FRAMES, RESIZE_WIDTH,
    ANALYSIS_UPDATE_FREQ, SAVE_ANNOTATED_VIDEO, VIDEO_CODEC,
    GOOD_POSE_DETECTION_RATE, FAIR_POSE_DETECTION_RATE, GOOD_BALL_TOUCHES,
    FOOT_DOMINANCE_THRESHOLD, FEEDBACK_MESSAGES, TOUCH_THRESHOLD
)
from ..detectors import PoseDetector, BallTracker, PlayerDetector
from ..analyzers import VideoProcessor, BallControlAnalyzer, PossessionAnalyzer
from ..utils.visualization import (
    create_real_time_dashboard, update_dashboard_display, clear_output_and_display,
    create_progress_display, create_coaching_feedback_display, 
    create_analysis_report_display, draw_touch_indicator
)


class FullVideoAnalysis:
    """
    Complete football video analysis system.
    Coordinates all components for comprehensive analysis.
    """
    
    def __init__(self, max_frames: int = MAX_FRAMES, 
                 skip_frames: int = SKIP_FRAMES,
                 resize_width: int = RESIZE_WIDTH,
                 save_video: bool = SAVE_ANNOTATED_VIDEO):
        """
        Initialize the full analysis system.
        
        Args:
            max_frames: Maximum number of frames to process
            skip_frames: Process every Nth frame
            resize_width: Width for frame resizing
            save_video: Whether to save annotated video
        """
        self.max_frames = max_frames
        self.skip_frames = skip_frames
        self.resize_width = resize_width
        self.save_video = save_video
        
        # Initialize components
        self.pose_detector = PoseDetector()
        self.ball_tracker = BallTracker()
        self.player_detector = PlayerDetector()
        self.ball_control_analyzer = BallControlAnalyzer()
        self.possession_analyzer = PossessionAnalyzer()
        # Speed estimation removed for cleaner analysis
        
        # Analysis tracking
        self.performance_metrics = {
            'total_frames': 0,
            'frames_with_pose': 0,
            'frames_with_ball': 0,
            'frames_with_players': 0,
            'ball_touches': 0,
            'avg_ball_confidence': 0,
            'foot_contacts': {'left': 0, 'right': 0},
            'team_possession': {'team_1': 0, 'team_2': 0, 'unknown': 0},
            'total_players_detected': 0,
            # Speed metrics removed
        }
        
        # Data storage
        self.pose_data = []
        self.ball_data = []
        self.player_data = []
        self.possession_data = []
        # Speed data storage removed
        self.ball_trajectory_x = []
        self.ball_trajectory_y = []
        self.touch_times = []
        
        print("🚀 Football Analysis System initialized!")
    
    def analyze_video(self, video_path: Path, 
                     show_realtime: bool = True) -> Dict[str, Any]:
        """
        Perform complete video analysis.
        
        Args:
            video_path: Path to video file
            show_realtime: Whether to show real-time dashboard
            
        Returns:
            Complete analysis results dictionary
        """
        print(f"🎬 Starting Full Football Analysis...")
        print("=" * 70)
        
        start_time = time.time()
        
        # Initialize video processor
        with VideoProcessor(video_path) as processor:
            # Setup video writer if needed
            video_writer = None
            if self.save_video:
                annotated_video_path = OUTPUT_DIR / f"annotated_{video_path.stem}.mp4"
                video_writer = processor.setup_video_writer(
                    annotated_video_path, self.resize_width
                )
                print(f"📹 Will save annotated video to: {annotated_video_path}")
            
            # Setup real-time dashboard
            fig, axes = None, None
            if show_realtime:
                fig, axes = create_real_time_dashboard(video_path.name)
            
            # Process video
            results = self._process_video_frames(
                processor, video_writer, fig, axes, start_time
            )
            
            # Close video writer
            if video_writer:
                video_writer.release()
                print(f"\n📹 Annotated video saved: {annotated_video_path}")
        
        # Generate final analysis
        total_time = time.time() - start_time
        final_results = self._generate_final_analysis(
            video_path, total_time, results
        )
        
        # Save comprehensive report
        report_path = self._save_analysis_report(video_path, final_results)
        
        # Display final summary
        self._display_final_summary(final_results, report_path)
        
        return final_results
    
    def _process_video_frames(self, processor: VideoProcessor, 
                             video_writer, fig, axes, 
                             start_time: float) -> Dict[str, Any]:
        """
        Process video frames with real-time analysis.
        
        Args:
            processor: VideoProcessor instance
            video_writer: OpenCV video writer
            fig: Matplotlib figure for dashboard
            axes: Matplotlib axes for dashboard
            start_time: Analysis start time
            
        Returns:
            Processing results dictionary
        """
        print(f"\n🔄 Processing video frames...")
        if fig is not None:
            print("📊 Real-time analysis will be displayed below")
        
        # Reset to beginning
        processor.seek_to_frame(0)
        frame_idx, processed = 0, 0
        
        while processed < self.max_frames:
            # Read frame
            success, frame = processor.get_frame()
            if not success:
                break
            
            if frame_idx % self.skip_frames == 0:
                # Process this frame
                processed_frame_data = self._process_single_frame(
                    frame, processor, frame_idx, video_writer
                )
                
                # Update real-time display
                if fig is not None and (processed % ANALYSIS_UPDATE_FREQ == 0 or processed < 5):
                    self._update_realtime_display(
                        fig, axes, processed_frame_data, start_time, processed
                    )
                
                processed += 1
            
            frame_idx += 1
        
        return {
            'total_processed': processed,
            'last_frame_idx': frame_idx - 1
        }
    
    def _process_single_frame(self, frame, processor: VideoProcessor, 
                             frame_idx: int, video_writer) -> Dict[str, Any]:
        """
        Process a single video frame.
        
        Args:
            frame: Input video frame
            processor: VideoProcessor instance
            frame_idx: Current frame index
            video_writer: OpenCV video writer
            
        Returns:
            Frame processing results
        """
        # Resize frame
        h, w = frame.shape[:2]
        scale = self.resize_width / w
        new_h = int(h * scale)
        frame_resized = processor.resize_frame(frame, self.resize_width)
        
        # Process frame
        frame_data = processor.process_frame(
            frame_resized, self.pose_detector, self.ball_tracker, frame_idx,
            self.player_detector
        )
        
        # Create annotated frame
        annotated_frame = frame_resized.copy()
        processor.annotate_frame(
            annotated_frame, frame_data, self.pose_detector, self.ball_tracker,
            self.player_detector
        )
        
        # Analyze ball touches
        self._analyze_frame_touches(frame_data, annotated_frame, new_h)
        
        # Advanced analysis
        self._process_advanced_analysis(frame_data, annotated_frame, frame_idx)
        
        # Add info overlay
        additional_info = {
            'Touches': self.performance_metrics['ball_touches'],
            'Players': len(frame_data.get('player_detections', [])),
            'Possession': self.possession_analyzer.current_possession or 'None'
        }
        processor.add_info_overlay(annotated_frame, frame_data, additional_info)
        
        # Update performance metrics
        self._update_performance_metrics(frame_data)
        
        # Update trajectory
        if frame_data['ball_detections']:
            best_ball = self.ball_tracker.get_best_detection(frame_data['ball_detections'])
            if best_ball:
                self.ball_trajectory_x.append(best_ball['center'][0])
                self.ball_trajectory_y.append(best_ball['center'][1])
        
        # Save to video
        if video_writer:
            video_writer.write(annotated_frame)
        
        return {
            'frame_data': frame_data,
            'annotated_frame': annotated_frame,
            'new_h': new_h
        }
    
    def _analyze_frame_touches(self, frame_data: Dict[str, Any], 
                              annotated_frame, frame_height: int):
        """
        Analyze ball touches for current frame.
        
        Args:
            frame_data: Frame detection data
            annotated_frame: Frame to draw touch indicators on
            frame_height: Frame height for coordinate conversion
        """
        if not (frame_data['landmarks'] and frame_data['ball_detections'] 
                and frame_data['key_points']):
            return
        
        # Check for ball touches
        for ball in frame_data['ball_detections']:
            ball_pos = ball['center']
            
            for foot_name in ['left_foot', 'right_foot']:
                if foot_name in frame_data['key_points']:
                    foot_pos = frame_data['key_points'][foot_name]
                    foot_pixel = (
                        int(foot_pos[0] * self.resize_width),
                        int(foot_pos[1] * frame_height)
                    )
                    
                    distance = np.sqrt(
                        (ball_pos[0] - foot_pixel[0])**2 + 
                        (ball_pos[1] - foot_pixel[1])**2
                    )
                    
                    if distance < TOUCH_THRESHOLD:  # Use updated touch threshold
                        # Only count as touch if this is a new touch event (debounced)
                        current_time = frame_data['timestamp'] 
                        if not self.touch_times or (current_time - self.touch_times[-1]) > 0.5:  # 0.5 second debounce
                            self.performance_metrics['ball_touches'] += 1
                            foot_side = foot_name.split('_')[0]
                            self.performance_metrics['foot_contacts'][foot_side] += 1
                            self.touch_times.append(current_time)
                        
                        # Draw touch indicator
                        draw_touch_indicator(annotated_frame, foot_pixel)
    
    def _process_advanced_analysis(self, frame_data: Dict[str, Any], 
                                  annotated_frame, frame_idx: int):
        """Process advanced analysis features."""
        player_detections = frame_data.get('player_detections', [])
        ball_detections = frame_data.get('ball_detections', [])
        
        # Speed estimation removed for cleaner analysis
        enhanced_players = player_detections
        if enhanced_players:
            frame_data['player_detections'] = enhanced_players
        
        # Possession analysis
        possession_result = self.possession_analyzer.analyze_possession(
            ball_detections, player_detections, frame_idx
        )
        
        # Draw possession information
        self.possession_analyzer.draw_possession_info(
            annotated_frame, possession_result, ball_detections
        )
        
        # Store possession data
        self.possession_data.append(possession_result)
        
        # Store player data
        if player_detections:
            self.player_data.append({
                'frame_idx': frame_idx,
                'players': player_detections
            })
    
    def _update_performance_metrics(self, frame_data: Dict[str, Any]):
        """Update performance metrics based on frame data."""
        self.performance_metrics['total_frames'] += 1
        
        if frame_data['landmarks']:
            self.performance_metrics['frames_with_pose'] += 1
        
        if frame_data['ball_detections']:
            self.performance_metrics['frames_with_ball'] += 1
            confidences = [b['confidence'] for b in frame_data['ball_detections']]
            self.performance_metrics['avg_ball_confidence'] = np.mean(confidences)
    
    def _update_realtime_display(self, fig, axes, processed_frame_data: Dict, 
                                start_time: float, processed: int):
        """Update the real-time analysis dashboard."""
        clear_output_and_display()
        
        frame_data = processed_frame_data['frame_data']
        annotated_frame = processed_frame_data['annotated_frame']
        new_h = processed_frame_data['new_h']
        
        # Prepare metrics data
        total_frames = max(1, self.performance_metrics['total_frames'])
        metrics_data = [
            self.performance_metrics['frames_with_pose'] / total_frames * 100,
            self.performance_metrics['frames_with_ball'] / total_frames * 100,
            self.performance_metrics['avg_ball_confidence'] * 100,
            min(self.performance_metrics['ball_touches'] * 10, 100)  # Scale touches
        ]
        
        # Update dashboard
        update_dashboard_display(
            fig, axes, annotated_frame, frame_data['frame_idx'], 
            frame_data['timestamp'], self.ball_trajectory_x, self.ball_trajectory_y,
            self.touch_times, metrics_data, self.resize_width, new_h
        )
        
        # Print progress
        elapsed = time.time() - start_time
        progress_str = create_progress_display(
            processed, self.max_frames, elapsed, self.performance_metrics
        )
        print(progress_str)
    
    def _generate_final_analysis(self, video_path: Path, 
                                total_time: float, 
                                processing_results: Dict) -> Dict[str, Any]:
        """Generate comprehensive final analysis results."""
        print(f"\n🎉 VIDEO ANALYSIS COMPLETE!")
        print(f"⏱️  Total processing time: {total_time//60:.0f}min {total_time%60:.1f}s")
        print(f"📊 Processed {self.performance_metrics['total_frames']} frames")
        
        # Generate ball control analysis
        ball_control_analysis = None
        if len(self.touch_times) >= 2:
            # Create touch events for analysis
            touch_events = []
            for i, touch_time in enumerate(self.touch_times):
                touch_events.append({
                    'timestamp': touch_time,
                    'foot': 'left_foot' if i % 2 == 0 else 'right_foot',  # Simplified
                    'quality_score': 0.8  # Simplified
                })
            
            ball_trajectory = list(zip(self.ball_trajectory_x, self.ball_trajectory_y))
            ball_control_analysis = self.ball_control_analyzer.analyze_dribbling_pattern(
                ball_trajectory, touch_events
            )
        
        # Compile final results
        final_results = {
            'video_info': {
                'filename': video_path.name,
                'total_frames_processed': self.performance_metrics['total_frames'],
                'processing_time_seconds': total_time,
                'fps': (self.performance_metrics['total_frames'] / max(1e-6, total_time))
            },
            'detection_performance': {
                'pose_detection_rate': self.performance_metrics['frames_with_pose'] / max(1, self.performance_metrics['total_frames']),
                'ball_detection_rate': self.performance_metrics['frames_with_ball'] / max(1, self.performance_metrics['total_frames']),
                'average_ball_confidence': self.performance_metrics['avg_ball_confidence']
            },
            'ball_control_analysis': {
                'total_ball_touches': self.performance_metrics['ball_touches'],
                'touch_frequency': self.performance_metrics['ball_touches'] / max(1, self.performance_metrics['total_frames'] / 30),
                'foot_preference': self.performance_metrics['foot_contacts'],
                'ball_trajectory_points': len(self.ball_trajectory_x),
                'detailed_analysis': ball_control_analysis
            }
        }
        
        return final_results
    
    def _save_analysis_report(self, video_path: Path, 
                             results: Dict[str, Any]) -> Path:
        """Save comprehensive analysis report to JSON."""
        report_path = OUTPUT_DIR / f"analysis_report_{video_path.stem}.json"
        
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return report_path
    
    def _display_final_summary(self, results: Dict[str, Any], report_path: Path):
        """Display final analysis summary."""
        # Analysis report
        report_display = create_analysis_report_display(results)
        print(report_display)
        
        # Generate and display feedback
        feedback = self._generate_coaching_feedback(results)
        feedback_display = create_coaching_feedback_display(feedback)
        print(f"\n{feedback_display}")
        
        # Files generated
        print(f"\n📁 FILES GENERATED:")
        if self.save_video:
            print(f"   🎬 Annotated video: {OUTPUT_DIR}/annotated_{Path(results['video_info']['filename']).stem}.mp4")
        print(f"   📄 Analysis report: {report_path}")
        print(f"   📊 Performance charts: Displayed above")
        
        print(f"\n🏆 FOOTBALL ANALYSIS SYSTEM COMPLETE!")
    
    def _generate_coaching_feedback(self, results: Dict[str, Any]) -> List[str]:
        """Generate coaching feedback based on analysis results."""
        feedback = []
        
        detection_perf = results['detection_performance']
        ball_control = results['ball_control_analysis']
        
        # Pose detection feedback
        pose_rate = detection_perf['pose_detection_rate']
        if pose_rate > GOOD_POSE_DETECTION_RATE:
            feedback.append(FEEDBACK_MESSAGES['excellent_pose'])
        elif pose_rate > FAIR_POSE_DETECTION_RATE:
            feedback.append(FEEDBACK_MESSAGES['good_pose'])
        else:
            feedback.append(FEEDBACK_MESSAGES['poor_pose'])
        
        # Ball control feedback
        total_touches = ball_control['total_ball_touches']
        if total_touches > GOOD_BALL_TOUCHES:
            feedback.append(FEEDBACK_MESSAGES['good_ball_control'])
        else:
            feedback.append(FEEDBACK_MESSAGES['limited_ball_control'])
        
        # Foot preference feedback
        left_touches = ball_control['foot_preference']['left']
        right_touches = ball_control['foot_preference']['right']
        if abs(left_touches - right_touches) > FOOT_DOMINANCE_THRESHOLD:
            dominant_foot = 'left' if left_touches > right_touches else 'right'
            feedback.append(FEEDBACK_MESSAGES['foot_dominance'].format(foot=dominant_foot.capitalize()))
        else:
            feedback.append(FEEDBACK_MESSAGES['balanced_feet'])
        
        return feedback
    
    def analyze_video_from_directory(self, video_index: int = 0, 
                                   show_realtime: bool = True) -> Optional[Dict[str, Any]]:
        """
        Analyze video from the configured video directory.
        
        Args:
            video_index: Index of video to analyze (0 for first, 1 for second, etc.)
            show_realtime: Whether to show real-time dashboard
            
        Returns:
            Analysis results or None if no videos found
        """
        if not VIDEO_DIR.exists():
            print("❌ Video directory not found!")
            return None
        
        video_files = list(VIDEO_DIR.glob("*.mp4"))
        if not video_files:
            print("❌ No video files found in 'video' directory!")
            print("📁 Please add .mp4 files to analyze")
            return None
        
        if video_index >= len(video_files):
            print(f"❌ Video index {video_index} out of range. Found {len(video_files)} videos.")
            return None
        
        selected_video = video_files[video_index]
        print(f"🎬 Analyzing: {selected_video.name} ({selected_video.stat().st_size // 1024 // 1024}MB)")
        
        try:
            return self.analyze_video(selected_video, show_realtime)
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def reset_analysis(self):
        """Reset all analysis data for new analysis."""
        self.performance_metrics = {
            'total_frames': 0,
            'frames_with_pose': 0,
            'frames_with_ball': 0,
            'ball_touches': 0,
            'avg_ball_confidence': 0,
            'foot_contacts': {'left': 0, 'right': 0}
        }
        
        self.pose_data.clear()
        self.ball_data.clear()
        self.ball_trajectory_x.clear()
        self.ball_trajectory_y.clear()
        self.touch_times.clear()
        
        self.ball_control_analyzer.reset_metrics()