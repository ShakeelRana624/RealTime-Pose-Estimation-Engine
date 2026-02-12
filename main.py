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
    
    # Check if model exists
    model_path = "models/yolov8n-pose.pt"
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("The model will be downloaded automatically on first run.")
    
    # Create outputs directory if it doesn't exist
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    
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
    cap = cv2.VideoCapture(0)
    
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
    print("💾 Outputs will be saved to 'outputs/' directory")
    print("Press 'q' to quit")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = "outputs/pose_activity_output.mp4"
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
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
            
            # Save pose data
            if pose_data:
                estimator.save_pose_data(pose_data, frame_count)
                
                # Update activity statistics
                for person in pose_data:
                    activity = person['activity']
                    if activity not in activity_stats:
                        activity_stats[activity] = 0
                    activity_stats[activity] += 1
            
            # Write output video
            out.write(annotated_frame)
            
            # Display frame
            cv2.imshow('Pose Estimation with Activity Detection', annotated_frame)
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
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
        out.release()
        cv2.destroyAllWindows()
        
        # Final statistics
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time
        
        print(f"\n🎉 PROCESSING COMPLETE!")
        print(f"📊 Total frames: {frame_count}")
        print(f"⏱️ Average FPS: {avg_fps:.2f}")
        print(f"📁 Pose data: outputs/pose_data.csv")
        print(f"🎬 Output video: {output_video_path}")
        print(f"\n🏃 ACTIVITY SUMMARY:")
        
        if activity_stats:
            for activity, count in sorted(activity_stats.items(), key=lambda x: x[1], reverse=True):
                print(f"  {activity.upper()}: {count} detections")
        else:
            print("  No activities detected")
        
        print(f"\n💡 TIP: Check the 'outputs/' directory for all generated files!")

if __name__ == "__main__":
    main()
