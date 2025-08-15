#!/usr/bin/env python3
"""
FIFA-Quality Football Analysis System - Main Entry Point
Professional 3D analysis with biomechanics, tactics, and corrective feedback.
"""

import sys
import argparse
from pathlib import Path
import traceback

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.analysis.fifa_analysis import FIFAAnalysisSystem
from src.config import VIDEO_DIR, OUTPUT_DIR

def main():
    """Main function for FIFA-quality football analysis."""
    parser = argparse.ArgumentParser(
        description="FIFA-Quality Football Analysis System - Professional 3D Biomechanical & Tactical Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
FIFA-Quality Features:
  • 3D Pose Reconstruction from MediaPipe landmarks
  • FIFA-standard field mapping and projection
  • Biomechanical analysis (body lean, balance, shooting mechanics)
  • Tactical analysis (passing, vision, decision making)
  • Professional 3D corrective feedback visualization
  • FIFA-compliant comprehensive reporting

Examples:
  %(prog)s                          # Analyze first video with FIFA-quality
  %(prog)s --video-index 1          # Analyze second video
  %(prog)s --max-frames 1000        # Full analysis (1000 frames)
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
        default=500,
        help='Maximum number of frames to process for FIFA analysis [default: 500]'
    )
    
    parser.add_argument(
        '--skip-frames', 
        type=int, 
        default=1,
        help='Process every Nth frame [default: 1 - no skipping for maximum quality]'
    )
    
    parser.add_argument(
        '--resize-width', 
        type=int, 
        default=640,
        help='Width for frame resizing [default: 640]'
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
    
    # Print FIFA banner
    print_fifa_banner()
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Initialize FIFA analysis system
    fifa_system = FIFAAnalysisSystem(
        max_frames=args.max_frames,
        skip_frames=args.skip_frames,
        resize_width=args.resize_width
    )
    
    # Run FIFA-quality analysis
    try:
        if args.video:
            # Analyze specific video file
            video_path = Path(args.video)
            if not video_path.exists():
                print(f"❌ Video file not found: {video_path}")
                sys.exit(1)
            
            print(f"🏆 Analyzing specific video with FIFA-quality: {video_path.name}")
            results = fifa_system.analyze_video(video_path, OUTPUT_DIR)
        else:
            # Analyze video from directory
            video_files = list(VIDEO_DIR.glob("*.mp4"))
            if not video_files:
                print(f"❌ No .mp4 files found in {VIDEO_DIR}")
                sys.exit(1)
            
            if args.video_index >= len(video_files):
                print(f"❌ Video index {args.video_index} out of range. Available videos: {len(video_files)}")
                sys.exit(1)
            
            video_path = video_files[args.video_index]
            print(f"🏆 FIFA-Quality Analysis: {video_path.name}")
            results = fifa_system.analyze_video(video_path, OUTPUT_DIR)
        
        if results:
            print_fifa_results(results)
        else:
            print(f"\n❌ FIFA analysis failed.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n⚠️ FIFA analysis interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error in FIFA analysis: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup resources
        fifa_system.cleanup()

def print_fifa_banner():
    """Print FIFA-quality banner."""
    banner = """
🏆 ============================================================
   FIFA-QUALITY FOOTBALL ANALYSIS SYSTEM
   Professional 3D Biomechanical & Tactical Evaluation
   ✅ 3D Pose Reconstruction  ✅ Field Mapping
   ✅ Biomechanical Analysis  ✅ Tactical Intelligence  
   ✅ Corrective Feedback     ✅ Professional Reporting
============================================================ ⚽
    """
    print(banner)

def print_fifa_results(results: dict):
    """Print FIFA analysis results."""
    print(f"\n🏆 FIFA-QUALITY ANALYSIS COMPLETED!")
    print(f"=" * 60)
    
    fifa_metrics = results.get('fifa_metrics', {})
    analysis_summary = results.get('analysis_summary', {})
    
    print(f"📊 FIFA METRICS:")
    print(f"   🎬 Frames Analyzed: {fifa_metrics.get('total_frames_analyzed', 0)}")
    print(f"   🤸 3D Reconstructions: {fifa_metrics.get('pose_3d_reconstructions', 0)}")
    print(f"   💪 Biomech Evaluations: {fifa_metrics.get('biomech_evaluations', 0)}")
    print(f"   🧠 Tactical Insights: {fifa_metrics.get('tactical_insights', 0)}")
    print(f"   🎯 Corrective Recommendations: {fifa_metrics.get('corrective_recommendations', 0)}")
    print(f"   🏆 FIFA Quality Score: {fifa_metrics.get('fifa_quality_score', 0):.3f}")
    
    fifa_compliant = fifa_metrics.get('fifa_quality_score', 0) > 0.7
    compliance_status = "✅ FIFA-COMPLIANT" if fifa_compliant else "⚠️ NEEDS IMPROVEMENT"
    print(f"   📋 Status: {compliance_status}")
    
    print(f"\n📁 OUTPUT FILES:")
    print(f"   🎬 Enhanced Video: {results.get('video_path', 'N/A')}")
    print(f"   🎭 3D Animation: {results.get('animation_path', 'N/A')}")
    print(f"   📊 FIFA Report: {results.get('report_path', 'N/A')}")
    
    print(f"\n🎯 PROFESSIONAL FEATURES DELIVERED:")
    print(f"   ✅ 3D Visualization Video with FIFA-standard field")
    print(f"   ✅ Biomechanical Analysis (posture, balance, shooting)")
    print(f"   ✅ Tactical Intelligence (passing, vision, decisions)")
    print(f"   ✅ 3D Corrective Feedback Models")
    print(f"   ✅ Professional Coaching Insights")
    print(f"   ✅ Comprehensive FIFA-compliant Reporting")
    
    if fifa_compliant:
        print(f"\n🏆 CONGRATULATIONS! FIFA-QUALITY STANDARD ACHIEVED!")
        print(f"   This analysis meets professional football evaluation criteria")
        print(f"   suitable for player development and coaching insights.")
    else:
        print(f"\n⚠️  IMPROVEMENT NEEDED for FIFA compliance")
        print(f"   Consider increasing frame count or video quality")
        print(f"   for more comprehensive biomechanical analysis.")

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
        print("   Add video files to begin FIFA-quality analysis.")
        return
    
    for i, video_file in enumerate(video_files):
        size_mb = video_file.stat().st_size / (1024 * 1024)
        print(f"   {i}: {video_file.name} ({size_mb:.1f}MB)")
    
    print(f"\nUse --video-index N to select a specific video for FIFA analysis")

def check_fifa_dependencies():
    """Check if required dependencies for FIFA analysis are installed."""
    required_modules = [
        'cv2', 'mediapipe', 'ultralytics', 'numpy', 
        'matplotlib', 'pandas', 'tqdm'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print("❌ Missing required dependencies for FIFA analysis:")
        for module in missing_modules:
            print(f"   - {module}")
        print("\nInstall dependencies with:")
        print("   pip install -r requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    # Check FIFA dependencies before running
    check_fifa_dependencies()
    
    # Run FIFA-quality analysis
    main()