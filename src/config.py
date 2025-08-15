"""
Configuration settings for the Football Analysis System.
"""

from pathlib import Path

# Project paths
VIDEO_DIR = Path("video")
OUTPUT_DIR = Path("analysis_results")
MODELS_DIR = Path("models")

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)

# Video processing settings
MAX_FRAMES = 1000          # Process up to N frames for demo
SKIP_FRAMES = 1          # Process every Nth frame for speed
RESIZE_WIDTH = 640       # Video display resolution
ANALYSIS_UPDATE_FREQ = 15 # Update analysis every N frames
SAVE_ANNOTATED_VIDEO = True  # Save video with annotations

# Detection settings
POSE_CONFIDENCE = 0.3     # MediaPipe pose detection confidence (lowered for better detection)
BALL_CONFIDENCE = 0.2     # YOLOv11 ball detection confidence (lowered to capture low-confidence balls)
BALL_CLASS_ID = 32        # 'sports ball' class in COCO dataset
PLAYER_CLASS_ID = 0       # 'person' class in COCO dataset
TOUCH_THRESHOLD = 30      # Distance threshold for ball-foot contact (pixels) - more realistic
TOUCH_DEBOUNCE_FRAMES = 10  # Minimum frames between separate touch events
MIN_TOUCH_DURATION = 3      # Minimum frames for a valid touch
USE_PITCH_MASK = True      # Use green pitch mask to filter ball detections

# Advanced tracking settings
PLAYER_CONFIDENCE = 0.3   # Player detection confidence
TRACKING_CONF_THRESHOLD = 0.5  # Tracking confidence threshold
MAX_DISAPPEARED = 30      # Maximum frames a track can disappear
EUCLIDEAN_DIST_THRESHOLD = 50   # Distance threshold for tracking

# Field analysis settings
FIELD_LENGTH_M = 105      # FIFA standard field length in meters
FIELD_WIDTH_M = 68        # FIFA standard field width in meters
MAX_PLAYER_SPEED_KMH = 40 # Maximum realistic player speed in km/h

# Simplified team colors (BGR values for OpenCV)
TEAM_COLORS = {
    'team_1': (0, 0, 255),      # Red team (BGR format)
    'team_2': (255, 0, 0),      # Blue team (BGR format)
    'referee': (255, 255, 255), # White referee (BGR format)
    'unknown': (128, 128, 128)  # Gray for unassigned (BGR format)
}

# Simplified team detection colors (RGB values for analysis)
JERSEY_COLOR_RANGES = {
    'red': {'lower': (100, 0, 0), 'upper': (255, 100, 100)},
    'blue': {'lower': (0, 0, 100), 'upper': (100, 100, 255)},
    'white': {'lower': (180, 180, 180), 'upper': (255, 255, 255)},
    'black': {'lower': (0, 0, 0), 'upper': (80, 80, 80)}
}

# YOLOv11 model settings
YOLO_MODEL_PATH = 'yolo11n.pt'  # YOLOv11 nano model

# Display settings
FIGURE_SIZE = (15, 10)    # Matplotlib figure size
BALL_TRAJECTORY_COLOR = 'ro-'
BALL_TRAJECTORY_ALPHA = 0.6
BALL_TRAJECTORY_MARKERSIZE = 3

# Video codec settings
VIDEO_CODEC = 'mp4v'      # Video codec for output
VIDEO_EXTENSION = '.mp4'  # Output video extension

# Performance metrics colors
METRICS_COLORS = ['skyblue', 'lightgreen', 'gold', 'coral']

# Touch detection colors (BGR format for OpenCV)
POSE_COLOR = (255, 0, 0)      # Blue for pose landmarks
BALL_COLOR = (0, 255, 0)      # Green for ball detection
TOUCH_COLOR = (255, 0, 255)   # Magenta for ball touches
INFO_TEXT_COLOR = (255, 255, 255)  # White for info text
INFO_SHADOW_COLOR = (0, 0, 0)      # Black for text shadow

# Analysis thresholds
GOOD_POSE_DETECTION_RATE = 0.8
FAIR_POSE_DETECTION_RATE = 0.5
GOOD_BALL_TOUCHES = 5
FOOT_DOMINANCE_THRESHOLD = 3

# Feedback messages
FEEDBACK_MESSAGES = {
    'excellent_pose': "✅ Excellent pose visibility throughout the video",
    'good_pose': "⚠️ Good pose detection - consider better camera angle for improvement",
    'poor_pose': "❌ Poor pose detection - check camera position and lighting",
    'good_ball_control': "⚽ Good ball control activity detected",
    'limited_ball_control': "⚽ Limited ball contact detected - focus on ball control drills",
    'balanced_feet': "🦶 Good balanced use of both feet",
    'foot_dominance': "🦶 {foot} foot dominant - practice with both feet"
}