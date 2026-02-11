"""
YOLOv8 Pose Estimation Module
Real-time multi-person 2D pose estimation with Human Activity Detection
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
import csv
import math
from pathlib import Path
from enum import Enum

# Try to import config, fallback to defaults if not available
try:
    from config import Config
except ImportError:
    print("⚠️ Config not found, using default settings")
    class Config:
        class PROJECT:
            MODEL_PATH = "models/yolov8n-pose.pt"
            POSE_DATA_CSV = "outputs/pose_data.csv"
            OUTPUT_VIDEO = "outputs/pose_activity_output.mp4"
        class POSE:
            SAVE_POSE_DATA = False
        class VIDEO:
            SAVE_VIDEO = False

class HumanActivity(Enum):
    STANDING = "standing"
    SITTING = "sitting"
    WALKING = "walking"
    RUNNING = "running"
    UNKNOWN = "unknown"

class PoseEstimator:
    def __init__(self, model_path=None):
        """Initialize YOLOv8 Pose Estimator"""
        if model_path is None:
            model_path = Config.PROJECT.MODEL_PATH
            
        print(f"Loading YOLOv8 Pose model: {model_path}")
        self.model = YOLO(model_path)
        print("✅ Model loaded successfully!")
        
        # COCO keypoint names
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # COCO skeleton connections
        self.skeleton = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
            [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
            [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
            [2, 4], [3, 5], [4, 6], [5, 7]
        ]
        
        # Colors
        self.keypoint_color = (0, 255, 0)  # Green
        self.skeleton_color = (0, 0, 255)  # Red
        
        # Activity detection
        self.pose_history = {}  # Store pose history for temporal analysis
        self.frame_count = 0
        
        # COCO keypoint indices for easier access
        self.NOSE = 0
        self.LEFT_SHOULDER = 5
        self.RIGHT_SHOULDER = 6
        self.LEFT_ELBOW = 7
        self.RIGHT_ELBOW = 8
        self.LEFT_WRIST = 9
        self.RIGHT_WRIST = 10
        self.LEFT_HIP = 11
        self.RIGHT_HIP = 12
        self.LEFT_KNEE = 13
        self.RIGHT_KNEE = 14
        self.LEFT_ANKLE = 15
        self.RIGHT_ANKLE = 16
        
    def calculate_angle(self, p1, p2, p3):
        """Calculate angle between three points in degrees"""
        if np.any(p1 == 0) or np.any(p2 == 0) or np.any(p3 == 0):
            return 0
        
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = math.degrees(math.acos(cos_angle))
        
        return angle
    
    def detect_activity(self, person_id, keypoints, confidences):
        """Detect human activity based on pose analysis"""
        
        # Store current pose in history
        if person_id not in self.pose_history:
            self.pose_history[person_id] = []
        
        current_pose = {
            'left_ankle': keypoints[self.LEFT_ANKLE].copy() if confidences[self.LEFT_ANKLE] > 0.5 else np.array([0, 0]),
            'right_ankle': keypoints[self.RIGHT_ANKLE].copy() if confidences[self.RIGHT_ANKLE] > 0.5 else np.array([0, 0]),
            'left_knee': keypoints[self.LEFT_KNEE].copy() if confidences[self.LEFT_KNEE] > 0.5 else np.array([0, 0]),
            'right_knee': keypoints[self.RIGHT_KNEE].copy() if confidences[self.RIGHT_KNEE] > 0.5 else np.array([0, 0]),
            'left_hip': keypoints[self.LEFT_HIP].copy() if confidences[self.LEFT_HIP] > 0.5 else np.array([0, 0]),
            'right_hip': keypoints[self.RIGHT_HIP].copy() if confidences[self.RIGHT_HIP] > 0.5 else np.array([0, 0]),
            'frame': self.frame_count
        }
        
        self.pose_history[person_id].append(current_pose)
        
        # Keep only last 10 frames for analysis
        if len(self.pose_history[person_id]) > 10:
            self.pose_history[person_id].pop(0)
        
        # Need at least 3 frames for temporal analysis
        if len(self.pose_history[person_id]) < 3:
            return HumanActivity.UNKNOWN, 0.0
        
        # Calculate knee angles
        left_knee_angle = self.calculate_angle(
            keypoints[self.LEFT_HIP], keypoints[self.LEFT_KNEE], keypoints[self.LEFT_ANKLE]
        )
        right_knee_angle = self.calculate_angle(
            keypoints[self.RIGHT_HIP], keypoints[self.RIGHT_KNEE], keypoints[self.RIGHT_ANKLE]
        )
        
        avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
        
        # Standing detection
        if 160 <= avg_knee_angle <= 180:
            # Check for movement (walking/running)
            movement = self.calculate_movement(person_id)
            if movement > 20:  # High movement = running
                return HumanActivity.RUNNING, 0.8
            elif movement > 5:  # Moderate movement = walking
                return HumanActivity.WALKING, 0.7
            else:  # No movement = standing
                return HumanActivity.STANDING, 0.8
        
        # Sitting detection
        elif 70 <= avg_knee_angle <= 120:
            return HumanActivity.SITTING, 0.8
        
        return HumanActivity.UNKNOWN, 0.0
    
    def calculate_movement(self, person_id):
        """Calculate movement intensity from pose history"""
        if person_id not in self.pose_history or len(self.pose_history[person_id]) < 2:
            return 0.0
        
        poses = self.pose_history[person_id]
        total_movement = 0.0
        movement_count = 0
        
        for i in range(1, len(poses)):
            prev_pose = poses[i-1]
            curr_pose = poses[i]
            
            # Calculate ankle movement
            if np.any(prev_pose['left_ankle'] != 0) and np.any(curr_pose['left_ankle'] != 0):
                left_movement = np.linalg.norm(curr_pose['left_ankle'] - prev_pose['left_ankle'])
                total_movement += left_movement
                movement_count += 1
            
            if np.any(prev_pose['right_ankle'] != 0) and np.any(curr_pose['right_ankle'] != 0):
                right_movement = np.linalg.norm(curr_pose['right_ankle'] - prev_pose['right_ankle'])
                total_movement += right_movement
                movement_count += 1
        
        return total_movement / movement_count if movement_count > 0 else 0.0
        
    def process_frame(self, frame):
        """Process frame for pose estimation with activity detection"""
        self.frame_count += 1
        results = self.model(frame, verbose=False)
        
        annotated_frame = frame.copy()
        pose_data = []
        
        for result in results:
            if result.keypoints is not None:
                keypoints = result.keypoints.xy.cpu().numpy()
                confidences = result.keypoints.conf.cpu().numpy()
                boxes = result.boxes.xyxy.cpu().numpy()
                
                for person_idx in range(len(keypoints)):
                    person_keypoints = keypoints[person_idx]
                    person_confidences = confidences[person_idx]
                    person_box = boxes[person_idx]
                    
                    # Detect activity
                    activity, activity_confidence = self.detect_activity(person_idx, person_keypoints, person_confidences)
                    
                    # Store pose data with activity
                    person_pose = {
                        'person_id': person_idx,
                        'bbox': person_box.tolist(),
                        'keypoints': person_keypoints.tolist(),
                        'confidences': person_confidences.tolist(),
                        'activity': activity.value,
                        'activity_confidence': activity_confidence
                    }
                    pose_data.append(person_pose)
                    
                    # Draw skeleton
                    self._draw_skeleton(annotated_frame, person_keypoints, person_confidences)
                    
                    # Draw bounding box and activity info
                    x1, y1, x2, y2 = map(int, person_box)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), self.keypoint_color, 2)
                    
                    # Activity label
                    activity_text = f"Person {person_idx}: {activity.value.upper()}"
                    cv2.putText(annotated_frame, activity_text, 
                               (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.keypoint_color, 2)
                    
                    # Confidence score
                    conf_text = f"Conf: {activity_confidence:.2f}"
                    cv2.putText(annotated_frame, conf_text, 
                               (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        return annotated_frame, pose_data
    
    def _draw_skeleton(self, frame, keypoints, confidences, conf_threshold=0.5):
        """Draw skeleton on frame"""
        # Draw keypoints
        for i, (kp, conf) in enumerate(zip(keypoints, confidences)):
            if conf > conf_threshold:
                x, y = map(int, kp)
                cv2.circle(frame, (x, y), 4, self.keypoint_color, -1)
        
        # Draw skeleton connections
        for connection in self.skeleton:
            pt1_idx, pt2_idx = connection[0] - 1, connection[1] - 1
            
            if (pt1_idx < len(confidences) and pt2_idx < len(confidences) and
                confidences[pt1_idx] > conf_threshold and 
                confidences[pt2_idx] > conf_threshold):
                
                pt1 = tuple(map(int, keypoints[pt1_idx]))
                pt2 = tuple(map(int, keypoints[pt2_idx]))
                cv2.line(frame, pt1, pt2, self.skeleton_color, 2)
        
        return frame
    
    def save_pose_data(self, pose_data, frame_id, output_path=None):
        """Save pose data with activity to CSV file (only if enabled)"""
        if not Config.POSE.SAVE_POSE_DATA:
            return  # Skip saving if disabled
            
        if output_path is None:
            output_path = Config.PROJECT.POSE_DATA_CSV
        file_exists = Path(output_path).exists()
        
        with open(output_path, 'a', newline='') as csvfile:
            fieldnames = ['frame_id', 'person_id', 'joint_id', 'joint_name', 
                          'x', 'y', 'confidence', 'activity', 'activity_confidence']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            for person in pose_data:
                person_id = person['person_id']
                keypoints = person['keypoints']
                confidences = person['confidences']
                activity = person['activity']
                activity_confidence = person['activity_confidence']
                
                for joint_idx, (kp, conf) in enumerate(zip(keypoints, confidences)):
                    writer.writerow({
                        'frame_id': frame_id,
                        'person_id': person_id,
                        'joint_id': joint_idx,
                        'joint_name': self.keypoint_names[joint_idx],
                        'x': kp[0],
                        'y': kp[1],
                        'confidence': conf,
                        'activity': activity,
                        'activity_confidence': activity_confidence
                    })

def main():
    """Main function for pose estimation with activity detection"""
    print("🚀 YOLOv8 POSE ESTIMATION MODULE")
    print("🏃 WITH HUMAN ACTIVITY DETECTION")
    print("=" * 50)
    
    # Initialize pose estimator
    estimator = PoseEstimator()
    
    # Use webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {width}x{height} @ {fps} FPS")
    print("Activities: STANDING, SITTING, WALKING, RUNNING")
    print("Press 'q' to quit")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('outputs/pose_activity_output.mp4', fourcc, fps, (width, height))
    
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
            cv2.imshow('YOLOv8 Pose Estimation with Activity Detection', annotated_frame)
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed
                
                # Display activity statistics
                print(f"FPS: {current_fps:.2f} | Activities: {activity_stats}")
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nProcessing interrupted")
    
    finally:
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time
        print(f"\n🎉 Processing complete!")
        print(f"📊 Total frames: {frame_count}")
        print(f"⏱️ Average FPS: {avg_fps:.2f}")
        print(f"📁 Pose data saved to: pose_data.csv")
        print(f"🎬 Output video: pose_activity_output.mp4")
        print(f"\n🏃 Activity Summary:")
        for activity, count in activity_stats.items():
            print(f"  {activity.upper()}: {count} detections")

if __name__ == "__main__":
    main()
