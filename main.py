#!/usr/bin/env python3
"""
Football Performance Analysis System - Main Entry Point

This script provides the main interface for running football video analysis.
"""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.analysis.full_analysis import FullVideoAnalysis
from src.config import VIDEO_DIR, OUTPUT_DIR


def main():
    """Main function to run football analysis."""
    parser = argparse.ArgumentParser(
        description="Football Performance Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Analyze first video in video/ directory
  %(prog)s --video-index 1          # Analyze second video
  %(prog)s --max-frames 500         # Process up to 500 frames
  %(prog)s --no-realtime            # Disable real-time dashboard
  %(prog)s --video path/to/video.mp4 # Analyze specific video file
        """
    )
    
    parser.add_argument(
        '--video', 
        type=str, 
        help='Path to specific video file to analyze'
    )
    
    parser.add_argument(
        '--video-index', 
        type=int, 
        default=0,
        help='Index of video in video/ directory (0=first, 1=second, etc.) [default: 0]'
    )
    
    parser.add_argument(
        '--max-frames', 
        type=int, 
        default=1000,
        help='Maximum number of frames to process [default: 200]'
    )
    
    parser.add_argument(
        '--skip-frames', 
        type=int, 
        default=1,
        help='Process every Nth frame [default: 2]'
    )
    
    parser.add_argument(
        '--resize-width', 
        type=int, 
        default=640,
        help='Width for frame resizing [default: 640]'
    )
    
    parser.add_argument(
        '--no-save-video', 
        action='store_true',
        help='Disable saving of annotated video'
    )
    
    parser.add_argument(
        '--no-realtime', 
        action='store_true',
        help='Disable real-time dashboard display'
    )
    
    parser.add_argument(
        '--list-videos', 
        action='store_true',
        help='List available videos in video/ directory'
    )
    
    args = parser.parse_args()
    
    # Handle list videos option
    if args.list_videos:
        list_available_videos()
        return
    
    # Print system banner
    print_banner()
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Initialize analysis system
    analysis_system = FullVideoAnalysis(
        max_frames=args.max_frames,
        skip_frames=args.skip_frames,
        resize_width=args.resize_width,
        save_video=not args.no_save_video
    )
    
    # Run analysis
    try:
        if args.video:
            # Analyze specific video file
            video_path = Path(args.video)
            if not video_path.exists():
                print(f"❌ Video file not found: {video_path}")
                sys.exit(1)
            
            print(f"🎬 Analyzing specific video: {video_path.name}")
            results = analysis_system.analyze_video(
                video_path, 
                show_realtime=not args.no_realtime
            )
        else:
            # Analyze video from directory
            results = analysis_system.analyze_video_from_directory(
                video_index=args.video_index,
                show_realtime=not args.no_realtime
            )
        
        if results:
            print(f"\n✅ Analysis completed successfully!")
            print(f"📊 Results saved to: {OUTPUT_DIR}")
        else:
            print(f"\n❌ Analysis failed or no videos found.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n⚠️ Analysis interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def print_banner():
    """Print system banner."""
    banner = """
🏆 ============================================================
   FOOTBALL PERFORMANCE ANALYSIS SYSTEM
   AI-Powered Sports Analytics with YOLOv11 & MediaPipe
============================================================ ⚽
    """
    print(banner)


def list_available_videos():
    """List available videos in the video directory."""
    print("📁 Available videos in video/ directory:")
    
    if not VIDEO_DIR.exists():
        print(f"❌ Video directory '{VIDEO_DIR}' not found!")
        print("   Create the directory and add .mp4 files to analyze.")
        return
    
    video_files = list(VIDEO_DIR.glob("*.mp4"))
    
    if not video_files:
        print("❌ No .mp4 files found in video/ directory!")
        print("   Add video files to begin analysis.")
        return
    
    for i, video_file in enumerate(video_files):
        size_mb = video_file.stat().st_size / (1024 * 1024)
        print(f"   {i}: {video_file.name} ({size_mb:.1f}MB)")
    
    print(f"\nUse --video-index N to select a specific video (e.g., --video-index 0)")


def check_dependencies():
    """Check if required dependencies are installed."""
    required_modules = [
        'cv2', 'mediapipe', 'ultralytics', 'numpy', 'matplotlib', 'pandas'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print("❌ Missing required dependencies:")
        for module in missing_modules:
            print(f"   - {module}")
        print("\nInstall dependencies with:")
        print("   pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    # Check dependencies before running
    check_dependencies()
    
    # Run main function
    main()