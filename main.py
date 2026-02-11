"""
Main Entry Point for Pose Estimation Module
Professional project structure with easy integration
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from pose_estimation import PoseEstimator
    from config import Config
    print("✅ Successfully imported modules")
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)

def main():
    """Main function with professional project structure"""
    print("🚀 POSE ESTIMATION MODULE")
    print("🏃 WITH HUMAN ACTIVITY DETECTION")
    print("=" * 50)
    
    # Show current configuration
    print(f"📹 Video Source: {Config.VIDEO.VIDEO_SOURCE}")
    print(f"💾 Save Video: {Config.VIDEO.SAVE_VIDEO}")
    print(f"📊 Save Pose Data: {Config.POSE.SAVE_POSE_DATA}")
    print(f"⚡ Confidence Threshold: {Config.POSE.CONFIDENCE_THRESHOLD}")
    print("=" * 50)
    
    # Check if model exists
    model_path = Config.PROJECT.MODEL_PATH
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("The model will be downloaded automatically on first run.")
    
    # Create outputs directory only if saving is enabled
    if Config.VIDEO.SAVE_VIDEO or Config.POSE.SAVE_POSE_DATA:
        outputs_dir = Path("outputs")
        outputs_dir.mkdir(exist_ok=True)
        print("💾 Outputs will be saved to 'outputs/' directory")
    else:
        print("💾 Output saving is disabled")
    
    # Initialize pose estimator
    try:
        estimator = PoseEstimator(model_path)
    except Exception as e:
        print(f"❌ Error initializing pose estimator: {e}")
        sys.exit(1)
    
    # Import required libraries for video processing
    try:
        import cv2
        import time
    except ImportError as e:
        print(f"❌ Missing required libraries: {e}")
        print("Please install: pip install opencv-python")
        sys.exit(1)
    
    # Use webcam
    cap = cv2.VideoCapture(Config.VIDEO.VIDEO_SOURCE)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        print("Please check if your webcam is connected and not being used by another application.")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📹 Video: {width}x{height} @ {fps} FPS")
    print("🏃 Activities: STANDING, SITTING, WALKING, RUNNING")
    print("Press 'q' to quit")
    
    # Setup video writer only if saving is enabled
    out = None
    if Config.VIDEO.SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*Config.VIDEO.OUTPUT_CODEC)
        output_video_path = Config.PROJECT.OUTPUT_VIDEO
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        print(f"🎬 Video will be saved to: {output_video_path}")
    
    frame_count = 0
    start_time = time.time()
    activity_stats = {}
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            annotated_frame, pose_data = estimator.process_frame(frame)
            
            # Save pose data only if enabled
            if pose_data and Config.POSE.SAVE_POSE_DATA:
                estimator.save_pose_data(pose_data, frame_count)
                
                # Update activity statistics
                for person in pose_data:
                    activity = person['activity']
                    if activity not in activity_stats:
                        activity_stats[activity] = 0
                    activity_stats[activity] += 1
            
            # Write output video only if enabled
            if out is not None:
                out.write(annotated_frame)
            
            # Display frame
            cv2.imshow(Config.VIDEO.WINDOW_NAME, annotated_frame)
            
            # Calculate FPS
            frame_count += 1
            if frame_count % Config.VIDEO.FPS_UPDATE_INTERVAL == 0:
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed
                
                # Display activity statistics
                print(f"🚀 FPS: {current_fps:.2f} | 🏃 Activities: {activity_stats}")
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n⏹️ Processing interrupted by user")
    
    except Exception as e:
        print(f"❌ Error during processing: {e}")
    
    finally:
        # Cleanup
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
        
        # Final statistics
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time
        
        print(f"\n🎉 PROCESSING COMPLETE!")
        print(f"📊 Total frames: {frame_count}")
        print(f"⏱️ Average FPS: {avg_fps:.2f}")
        
        if Config.POSE.SAVE_POSE_DATA:
            print(f"📁 Pose data: {Config.PROJECT.POSE_DATA_CSV}")
        else:
            print("📁 Pose data saving was disabled")
            
        if Config.VIDEO.SAVE_VIDEO and out is not None:
            print(f"🎬 Output video: {Config.PROJECT.OUTPUT_VIDEO}")
        else:
            print("🎬 Video saving was disabled")
        
        print(f"\n🏃 ACTIVITY SUMMARY:")
        
        if activity_stats:
            for activity, count in sorted(activity_stats.items(), key=lambda x: x[1], reverse=True):
                print(f"  {activity.upper()}: {count} detections")
        else:
            print("  No activities detected")
        
        if not (Config.VIDEO.SAVE_VIDEO or Config.POSE.SAVE_POSE_DATA):
            print(f"\n💡 TIP: Enable saving in config.py if you want to save outputs!")

if __name__ == "__main__":
    main()
