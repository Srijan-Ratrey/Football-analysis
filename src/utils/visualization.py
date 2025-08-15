"""
Visualization utilities for football analysis system.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
from IPython.display import display, clear_output
from ..config import (
    FIGURE_SIZE, BALL_TRAJECTORY_COLOR, BALL_TRAJECTORY_ALPHA, 
    BALL_TRAJECTORY_MARKERSIZE, METRICS_COLORS, TOUCH_COLOR
)


def create_real_time_dashboard(video_name: str) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Create a real-time analysis dashboard with multiple panels.
    
    Args:
        video_name: Name of the video being analyzed
        
    Returns:
        Tuple of (figure, axes_list)
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=FIGURE_SIZE)
    fig.suptitle(f'Real-Time Football Analysis: {video_name}', fontsize=14)
    
    # Initialize plot elements
    ax1.set_title('Current Frame with Detection')
    ax1.axis('off')
    
    ax2.set_title('Ball Trajectory')
    ax2.grid(True, alpha=0.3)
    
    ax3.set_title('Touch Frequency Over Time')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Touch Count')
    
    ax4.set_title('Performance Metrics')
    ax4.set_ylim(0, 100)
    
    return fig, [ax1, ax2, ax3, ax4]


def update_current_frame_display(ax: plt.Axes, annotated_frame, 
                                frame_idx: int, timestamp: float):
    """
    Update the current frame display panel.
    
    Args:
        ax: Matplotlib axes to update
        annotated_frame: Annotated video frame
        frame_idx: Current frame index
        timestamp: Current timestamp
    """
    ax.clear()
    # Convert BGR to RGB for matplotlib
    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    ax.imshow(frame_rgb)
    ax.set_title(f'Frame {frame_idx} - Time: {timestamp:.1f}s')
    ax.axis('off')


def update_ball_trajectory_display(ax: plt.Axes, trajectory_x: List[int], 
                                  trajectory_y: List[int], 
                                  frame_width: int, frame_height: int):
    """
    Update the ball trajectory display panel.
    
    Args:
        ax: Matplotlib axes to update
        trajectory_x: List of x coordinates
        trajectory_y: List of y coordinates
        frame_width: Frame width for setting limits
        frame_height: Frame height for setting limits
    """
    ax.clear()
    
    if len(trajectory_x) > 1:
        ax.plot(trajectory_x, trajectory_y, BALL_TRAJECTORY_COLOR, 
                alpha=BALL_TRAJECTORY_ALPHA, markersize=BALL_TRAJECTORY_MARKERSIZE)
        ax.set_xlim(0, frame_width)
        ax.set_ylim(0, frame_height)
        ax.invert_yaxis()  # Invert Y-axis to match image coordinates
    
    ax.set_title(f'Ball Trajectory ({len(trajectory_x)} points)')
    ax.grid(True, alpha=0.3)


def update_touch_frequency_display(ax: plt.Axes, touch_times: List[float]):
    """
    Update the touch frequency histogram display.
    
    Args:
        ax: Matplotlib axes to update
        touch_times: List of touch timestamps
    """
    ax.clear()
    
    if touch_times:
        bins = max(1, len(touch_times) // 3)
        ax.hist(touch_times, bins=bins, alpha=0.7, color='orange')
    
    ax.set_title(f'Ball Touches Over Time ({len(touch_times)} total)')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Touch Count')


def update_performance_metrics_display(ax: plt.Axes, metrics_data: List[float], 
                                     metrics_labels: List[str]):
    """
    Update the performance metrics bar chart.
    
    Args:
        ax: Matplotlib axes to update
        metrics_data: List of metric values (0-100)
        metrics_labels: List of metric labels
    """
    ax.clear()
    
    bars = ax.bar(metrics_labels, metrics_data, color=METRICS_COLORS)
    ax.set_ylim(0, 100)
    ax.set_title('Real-Time Performance Metrics')
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_data):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.1f}', ha='center', va='bottom', fontsize=9)


def create_analysis_plots(analysis_results: Dict[str, Any], 
                         output_path: str = None) -> plt.Figure:
    """
    Create comprehensive analysis plots from results.
    
    Args:
        analysis_results: Analysis results dictionary
        output_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=FIGURE_SIZE)
    fig.suptitle('Football Performance Analysis Results', fontsize=16)
    
    # Plot 1: Detection rates
    detection_data = analysis_results.get('detection_performance', {})
    pose_rate = detection_data.get('pose_detection_rate', 0) * 100
    ball_rate = detection_data.get('ball_detection_rate', 0) * 100
    
    ax1.bar(['Pose Detection', 'Ball Detection'], [pose_rate, ball_rate], 
            color=['skyblue', 'lightgreen'])
    ax1.set_ylim(0, 100)
    ax1.set_ylabel('Detection Rate (%)')
    ax1.set_title('Detection Performance')
    
    # Plot 2: Ball control metrics
    ball_control = analysis_results.get('ball_control_analysis', {})
    total_touches = ball_control.get('total_ball_touches', 0)
    touch_freq = ball_control.get('touch_frequency', 0)
    
    ax2.bar(['Total Touches', 'Touch Frequency'], [total_touches, touch_freq * 10], 
            color=['coral', 'gold'])
    ax2.set_ylabel('Count / Frequency x10')
    ax2.set_title('Ball Control Metrics')
    
    # Plot 3: Foot preference
    foot_pref = ball_control.get('foot_preference', {})
    left_touches = foot_pref.get('left', 0)
    right_touches = foot_pref.get('right', 0)
    
    if left_touches + right_touches > 0:
        ax3.pie([left_touches, right_touches], labels=['Left Foot', 'Right Foot'], 
                autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
    ax3.set_title('Foot Usage Distribution')
    
    # Plot 4: Performance timeline (if available)
    timeline_data = analysis_results.get('timeline_data', [])
    if timeline_data:
        times = [t['timestamp'] for t in timeline_data]
        touches = [t['cumulative_touches'] for t in timeline_data]
        ax4.plot(times, touches, 'b-', linewidth=2)
        ax4.set_xlabel('Time (seconds)')
        ax4.set_ylabel('Cumulative Touches')
        ax4.set_title('Performance Timeline')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Timeline data\nnot available', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Performance Timeline')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Analysis plots saved to: {output_path}")
    
    return fig


def draw_touch_indicator(frame, foot_position: Tuple[int, int], 
                        color=TOUCH_COLOR, radius: int = 15):
    """
    Draw a touch indicator on the frame.
    
    Args:
        frame: Frame to draw on
        foot_position: (x, y) position of the foot
        color: BGR color tuple
        radius: Circle radius
    """
    cv2.circle(frame, foot_position, radius, color, 3)
    cv2.putText(frame, 'TOUCH!', 
               (foot_position[0]-20, foot_position[1]-20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def create_progress_display(processed: int, total: int, elapsed_time: float, 
                           performance_metrics: Dict[str, int]) -> str:
    """
    Create formatted progress display string.
    
    Args:
        processed: Number of frames processed
        total: Total frames to process
        elapsed_time: Time elapsed so far
        performance_metrics: Current performance metrics
        
    Returns:
        Formatted progress string
    """
    progress = processed / total if total > 0 else 0
    eta = (elapsed_time / progress - elapsed_time) if progress > 0 else 0
    
    progress_str = (
        f"📊 Progress: {processed}/{total} ({progress*100:.1f}%) | "
        f"Elapsed: {elapsed_time:.0f}s | ETA: {eta:.0f}s\n"
        f"⚽ Detections: Pose={performance_metrics.get('frames_with_pose', 0)}, "
        f"Ball={performance_metrics.get('frames_with_ball', 0)}, "
        f"Touches={performance_metrics.get('ball_touches', 0)}"
    )
    
    return progress_str


def create_coaching_feedback_display(feedback: List[str]) -> str:
    """
    Create formatted coaching feedback display.
    
    Args:
        feedback: List of feedback messages
        
    Returns:
        Formatted feedback string
    """
    if not feedback:
        return "No feedback available."
    
    feedback_str = "💬 COACHING FEEDBACK:\n"
    for i, fb in enumerate(feedback, 1):
        feedback_str += f"   {i}. {fb}\n"
    
    return feedback_str


def create_analysis_report_display(report: Dict[str, Any]) -> str:
    """
    Create formatted analysis report display.
    
    Args:
        report: Analysis report dictionary
        
    Returns:
        Formatted report string
    """
    video_info = report.get('video_info', {})
    detection_perf = report.get('detection_performance', {})
    ball_control = report.get('ball_control_analysis', {})
    
    report_str = (
        f"📄 ANALYSIS REPORT:\n"
        f"   🎬 Video: {video_info.get('filename', 'Unknown')}\n"
        f"   📊 Frames processed: {video_info.get('total_frames_processed', 0)}\n"
        f"   🤸 Pose detection: {detection_perf.get('pose_detection_rate', 0)*100:.1f}%\n"
        f"   ⚽ Ball detection: {detection_perf.get('ball_detection_rate', 0)*100:.1f}%\n"
        f"   🏃 Ball touches: {ball_control.get('total_ball_touches', 0)}\n"
        f"   🦶 Foot preference: L:{ball_control.get('foot_preference', {}).get('left', 0)} "
        f"R:{ball_control.get('foot_preference', {}).get('right', 0)}"
    )
    
    return report_str


def update_dashboard_display(fig: plt.Figure, axes: List[plt.Axes], 
                           annotated_frame, frame_idx: int, timestamp: float,
                           trajectory_x: List[int], trajectory_y: List[int],
                           touch_times: List[float], metrics_data: List[float],
                           frame_width: int, frame_height: int):
    """
    Update all dashboard panels in one call.
    
    Args:
        fig: Matplotlib figure
        axes: List of axes [ax1, ax2, ax3, ax4]
        annotated_frame: Current annotated frame
        frame_idx: Current frame index
        timestamp: Current timestamp
        trajectory_x: Ball trajectory x coordinates
        trajectory_y: Ball trajectory y coordinates
        touch_times: List of touch timestamps
        metrics_data: Performance metrics data
        frame_width: Frame width
        frame_height: Frame height
    """
    ax1, ax2, ax3, ax4 = axes
    
    # Update all panels
    update_current_frame_display(ax1, annotated_frame, frame_idx, timestamp)
    update_ball_trajectory_display(ax2, trajectory_x, trajectory_y, frame_width, frame_height)
    update_touch_frequency_display(ax3, touch_times)
    
    # Performance metrics
    metrics_labels = ['Pose\nDetection %', 'Ball\nDetection %', 'Ball\nConfidence %', 'Touch\nActivity']
    update_performance_metrics_display(ax4, metrics_data, metrics_labels)
    
    plt.tight_layout()
    plt.show()


def clear_output_and_display():
    """Clear output and prepare for new display."""
    clear_output(wait=True)