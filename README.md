# 🏆 Football Performance Analysis System

An advanced AI-powered sports analytics platform for football/soccer performance analysis using **YOLOv11** and **MediaPipe**. This system provides both standard analysis and FIFA-quality professional analysis with 3D visualization and biomechanical insights.

## 🎯 Features

### 🤖 AI-Powered Detection
- **Pose Detection**: MediaPipe for human pose estimation and body tracking
- **Ball Tracking**: YOLOv11 (latest) for fast and accurate ball detection
- **Player Detection**: Advanced player tracking with team identification
- **Touch Detection**: Automatic detection of ball-foot contact events

### 📊 Performance Analytics
- **Standard Analysis**: Basic performance metrics and ball control analysis
- **FIFA-Quality Analysis**: Professional 3D biomechanical and tactical evaluation
- **Real-time Processing**: Live video analysis with interactive dashboard
- **Ball Control Metrics**: Touch frequency, foot preference, control consistency
- **Movement Analysis**: Ball trajectory tracking and movement patterns
- **Performance Scoring**: Quality assessment and coaching insights

### 🎬 Video Processing
- **Annotated Output**: Videos with pose landmarks and ball detection overlays
- **3D Visualization**: FIFA-quality 3D pose reconstruction and field mapping
- **Progress Tracking**: Real-time progress with ETA calculations
- **Optimized Performance**: Configurable frame skipping and resolution scaling

### 📄 Comprehensive Reporting
- **JSON Reports**: Detailed analysis results with metrics and statistics
- **Coaching Feedback**: Automated insights and improvement suggestions
- **Visual Charts**: Performance graphs and trajectory plots
- **3D Animations**: Professional 3D corrective feedback visualization

## 🚀 Quick Start

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Football
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add video files**
   ```bash
   mkdir -p video
   # Copy your .mp4 video files to the video/ directory
   ```

### Basic Usage

#### Standard Analysis
```bash
# Analyze first video in video/ directory
python main.py

# Analyze specific video
python main.py --video path/to/your/video.mp4

# List available videos
python main.py --list-videos

# Analyze with custom settings
python main.py --max-frames 500 --video-index 1 --no-realtime
```

#### FIFA-Quality Analysis
```bash
# FIFA-quality analysis of first video
python fifa_main.py

# FIFA analysis of specific video
python fifa_main.py --video path/to/your/video.mp4

# FIFA analysis with custom settings
python fifa_main.py --max-frames 1000 --video-index 1
```

## 📂 Project Structure

```
Football/
├── src/                    # Source code
│   ├── detectors/         # Detection modules
│   │   ├── pose_detector.py      # MediaPipe pose detection
│   │   ├── ball_tracker.py       # YOLOv11 ball tracking
│   │   └── player_detector.py    # Player detection and tracking
│   ├── analyzers/         # Analysis modules
│   │   ├── video_processor.py    # Video I/O and processing
│   │   ├── ball_control_analyzer.py  # Performance analysis
│   │   ├── tactical_analyzer.py      # Tactical analysis
│   │   ├── speed_estimator.py        # Speed and movement analysis
│   │   └── possession_analyzer.py    # Possession and control analysis
│   ├── utils/             # Utility modules
│   ├── analysis/          # Main analysis pipeline
│   │   ├── full_analysis.py      # Standard analysis orchestration
│   │   └── fifa_analysis.py      # FIFA-quality analysis system
│   ├── visualization/     # 3D visualization modules
│   │   ├── pose_3d.py            # 3D pose reconstruction
│   │   └── renderer_3d.py        # 3D rendering and visualization
│   └── config.py          # Configuration settings
├── video/                 # Input video directory
├── analysis_results/      # Output directory
├── main.py               # Standard analysis entry point
├── fifa_main.py          # FIFA-quality analysis entry point
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🛠️ Configuration

Edit `src/config.py` to customize:

```python
# Processing settings
MAX_FRAMES = 1000         # Maximum frames to process
SKIP_FRAMES = 1          # Process every Nth frame
RESIZE_WIDTH = 640       # Video resolution for processing

# Detection settings  
POSE_CONFIDENCE = 0.3     # MediaPipe confidence threshold
BALL_CONFIDENCE = 0.2     # YOLOv11 confidence threshold
TOUCH_THRESHOLD = 30      # Ball-foot contact distance (pixels)

# Output settings
SAVE_ANNOTATED_VIDEO = True  # Save annotated video files
```

## 📊 Analysis Types

### Standard Analysis (`main.py`)
- Basic pose and ball detection
- Ball control metrics
- Touch frequency analysis
- Performance scoring
- Real-time dashboard

### FIFA-Quality Analysis (`fifa_main.py`)
- Professional 3D pose reconstruction
- FIFA-standard field mapping
- Biomechanical analysis (posture, balance, shooting)
- Tactical intelligence (passing, vision, decisions)
- 3D corrective feedback visualization
- Professional coaching insights

## 📈 Analysis Outputs

### Generated Files

- **Annotated Video**: `annotated_{video_name}.mp4` - Video with all detections
- **Analysis Report**: `analysis_report_{video_name}.json` - Detailed metrics
- **3D Visualization**: FIFA-quality 3D analysis videos
- **Performance Charts**: Real-time visualization dashboard

### Analysis Metrics

- **Detection Performance**: Pose and ball detection rates
- **Ball Control**: Touch frequency, foot preference, consistency
- **Movement Analysis**: Ball trajectory, speed, direction changes
- **Quality Scores**: Touch quality and performance ratings
- **FIFA Metrics**: Professional biomechanical and tactical evaluation

### Sample Report Structure

```json
{
  "video_info": {
    "filename": "training_session.mp4",
    "total_frames_processed": 500,
    "processing_time_seconds": 120.5,
    "fps": 30
  },
  "detection_performance": {
    "pose_detection_rate": 0.85,
    "ball_detection_rate": 0.78,
    "average_ball_confidence": 0.75
  },
  "ball_control_analysis": {
    "total_ball_touches": 25,
    "touch_frequency": 3.2,
    "foot_preference": {"left": 12, "right": 13},
    "ball_trajectory_points": 312
  },
  "fifa_metrics": {
    "fifa_quality_score": 0.82,
    "biomech_evaluations": 156,
    "tactical_insights": 89,
    "corrective_recommendations": 23
  }
}
```

## 💡 Coaching Insights

The system automatically generates coaching feedback:

- **Pose Visibility**: Camera angle and lighting recommendations
- **Ball Control**: Touch frequency and consistency analysis
- **Foot Usage**: Balance between left and right foot usage
- **Touch Quality**: Proximity and control quality assessment
- **Biomechanical**: Posture, balance, and shooting mechanics
- **Tactical**: Passing decisions, vision, and positioning

## 🔧 Command Line Options

### Standard Analysis (`main.py`)
```bash
usage: main.py [-h] [--video VIDEO] [--video-index VIDEO_INDEX]
               [--max-frames MAX_FRAMES] [--skip-frames SKIP_FRAMES]
               [--resize-width RESIZE_WIDTH] [--no-save-video]
               [--no-realtime] [--list-videos]

optional arguments:
  -h, --help            show this help message and exit
  --video VIDEO         Path to specific video file to analyze
  --video-index VIDEO_INDEX
                        Index of video in video/ directory (default: 0)
  --max-frames MAX_FRAMES
                        Maximum number of frames to process (default: 1000)
  --skip-frames SKIP_FRAMES
                        Process every Nth frame (default: 1)
  --resize-width RESIZE_WIDTH
                        Width for frame resizing (default: 640)
  --no-save-video       Disable saving of annotated video
  --no-realtime         Disable real-time dashboard display
  --list-videos         List available videos in video/ directory
```

### FIFA-Quality Analysis (`fifa_main.py`)
```bash
usage: fifa_main.py [-h] [--video VIDEO] [--video-index VIDEO_INDEX]
                    [--max-frames MAX_FRAMES] [--skip-frames SKIP_FRAMES]
                    [--resize-width RESIZE_WIDTH] [--list-videos]

optional arguments:
  -h, --help            show this help message and exit
  --video VIDEO         Path to specific video file to analyze
  --video-index VIDEO_INDEX
                        Index of video in video/ directory (default: 0)
  --max-frames MAX_FRAMES
                        Maximum number of frames to process (default: 500)
  --skip-frames SKIP_FRAMES
                        Process every Nth frame (default: 1)
  --resize-width RESIZE_WIDTH
                        Width for frame resizing (default: 640)
  --list-videos         List available videos in video/ directory
```

## 🎥 Supported Video Formats

- **Input**: MP4 video files (`.mp4`)
- **Resolution**: Any resolution (automatically resized for processing)
- **Frame Rate**: Any frame rate (automatically detected)
- **Content**: Football/soccer training or match footage

## ⚡ Performance Optimization

- **YOLOv11**: Latest YOLO version for fastest ball detection
- **Frame Skipping**: Configurable frame processing for speed vs. quality
- **Resolution Scaling**: Automatic resizing for optimal performance
- **Progress Tracking**: Real-time progress with ETA calculations
- **Parallel Processing**: Efficient video processing pipeline

## 🔬 Technical Details

### Detection Technologies
- **MediaPipe**: Google's pose estimation solution
- **YOLOv11**: Ultralytics' latest object detection model
- **OpenCV**: Computer vision and video processing
- **NumPy/Matplotlib**: Data analysis and visualization

### Key Components
- **PoseDetector**: Human pose estimation and keypoint extraction
- **BallTracker**: Football detection and trajectory tracking
- **PlayerDetector**: Player detection and team identification
- **VideoProcessor**: Video I/O and frame processing pipeline
- **BallControlAnalyzer**: Performance metrics and touch analysis
- **FIFAAnalysisSystem**: Professional 3D analysis and visualization

## 🐛 Troubleshooting

### Common Issues

1. **"No module named 'cv2'"**
   ```bash
   pip install opencv-python
   ```

2. **YOLOv11 model download issues**
   - Ensure internet connection
   - Model downloads automatically on first run

3. **Video not found**
   ```bash
   python main.py --list-videos  # Check available videos
   ```

4. **Performance issues**
   - Increase `--skip-frames` value
   - Reduce `--resize-width`
   - Decrease `--max-frames`

### System Requirements

- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space for models and output
- **GPU**: Optional but recommended for faster processing

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📧 Support

For questions, issues, or feature requests, please:
1. Check the troubleshooting section above
2. Search existing issues
3. Create a new issue with detailed information

## 🏆 Acknowledgments

- **MediaPipe** team at Google for pose estimation
- **Ultralytics** for YOLOv11 object detection
- **OpenCV** community for computer vision tools

---

**Ready to analyze your football performance? Choose your analysis level:**

- **Standard Analysis**: `python main.py` - Basic performance metrics
- **FIFA-Quality Analysis**: `python fifa_main.py` - Professional 3D insights

⚽🚀

