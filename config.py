"""
Configuration file for Pose Estimation Module
Easy to modify settings for different use cases
"""

import os
from pathlib import Path

# Project Structure Configuration
class ProjectConfig:
    # Base directory
    BASE_DIR = Path(__file__).parent
    
    # Model configuration
    MODEL_DIR = BASE_DIR / "models"
    MODEL_NAME = "yolov8n-pose.pt"
    MODEL_PATH = str(MODEL_DIR / MODEL_NAME)
    
    # Source code directory
    SRC_DIR = BASE_DIR / "src"
    
    # Output directory
    OUTPUTS_DIR = BASE_DIR / "outputs"
    OUTPUTS_DIR.mkdir(exist_ok=True)  # Create if doesn't exist
    
    # Data files
    POSE_DATA_CSV = str(OUTPUTS_DIR / "pose_data.csv")
    OUTPUT_VIDEO = str(OUTPUTS_DIR / "pose_activity_output.mp4")

# Pose Estimation Configuration
class PoseConfig:
    # Model settings
    CONFIDENCE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.45
    INPUT_SIZE = 640
    
    # Activity detection settings
    ACTIVITY_CONFIDENCE_THRESHOLD = 0.6
    MOVEMENT_FRAMES_WINDOW = 10  # Frames to analyze for movement
    MOVEMENT_THRESHOLD_WALKING = 5.0  # pixels per frame
    MOVEMENT_THRESHOLD_RUNNING = 20.0  # pixels per frame
    
    # Knee angle thresholds (in degrees)
    KNEE_ANGLE_STANDING_MIN = 160
    KNEE_ANGLE_STANDING_MAX = 180
    KNEE_ANGLE_SITTING_MIN = 70
    KNEE_ANGLE_SITTING_MAX = 120
    
    # Data saving settings
    SAVE_POSE_DATA = False  # Set to False to disable CSV saving
    
    # Visualization settings
    KEYPOINT_COLOR = (0, 255, 0)  # Green (BGR format)
    SKELETON_COLOR = (0, 0, 255)  # Red (BGR format)
    BOUNDING_BOX_COLOR = (0, 255, 0)  # Green
    TEXT_COLOR = (255, 255, 255)  # White
    CONFIDENCE_TEXT_COLOR = (255, 255, 0)  # Yellow
    
    # Font settings
    FONT_SCALE = 0.6
    FONT_THICKNESS = 2
    CONFIDENCE_FONT_SCALE = 0.4
    CONFIDENCE_FONT_THICKNESS = 1

# Video Processing Configuration
class VideoConfig:
    # Video source options
    # Use 0 for webcam, or provide path to video file
    VIDEO_SOURCE = 0
    
    # Output video settings
    SAVE_VIDEO = False  # Set to False to disable video saving
    OUTPUT_FPS = 30  # Will use input FPS if available
    OUTPUT_CODEC = 'mp4v'
    
    # Display settings
    WINDOW_NAME = "Pose Estimation with Activity Detection"
    FPS_UPDATE_INTERVAL = 30  # Update FPS display every N frames
    
    # Processing settings
    MAX_FRAMES_TO_PROCESS = None  # Set to None for unlimited
    SAVE_EVERY_N_FRAMES = 1  # Save every frame (set to higher for less storage)

# COCO Keypoints Configuration
class COCOConfig:
    # Keypoint names (COCO format)
    KEYPOINT_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    # Keypoint indices for easier access
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16
    
    # Skeleton connections (COCO format)
    # Each pair represents a connection between keypoints
    SKELETON_CONNECTIONS = [
        [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],  # Lower body
        [6, 12], [7, 13], [6, 7],  # Torso
        [6, 8], [7, 9], [8, 10], [9, 11],  # Arms
        [2, 3], [1, 2], [1, 3],  # Face
        [2, 4], [3, 5], [4, 6], [5, 7]  # Head to shoulders
    ]

# Human Activity Configuration
class ActivityConfig:
    # Activity types
    ACTIVITIES = {
        'standing': {
            'name': 'STANDING',
            'color': (0, 255, 0),  # Green
            'description': 'Person is standing upright'
        },
        'sitting': {
            'name': 'SITTING', 
            'color': (255, 255, 0),  # Yellow
            'description': 'Person is sitting down'
        },
        'walking': {
            'name': 'WALKING',
            'color': (0, 255, 255),  # Cyan
            'description': 'Person is walking'
        },
        'running': {
            'name': 'RUNNING',
            'color': (0, 0, 255),  # Red
            'description': 'Person is running'
        },
        'unknown': {
            'name': 'UNKNOWN',
            'color': (128, 128, 128),  # Gray
            'description': 'Activity could not be determined'
        }
    }
    
    # Minimum confidence for activity detection
    MIN_ACTIVITY_CONFIDENCE = 0.5
    
    # Frames needed for stable activity detection
    MIN_FRAMES_FOR_DETECTION = 3

# Performance Configuration
class PerformanceConfig:
    # Performance optimization settings
    ENABLE_GPU = True  # Will use CUDA if available
    ENABLE_MODEL_OPTIMIZATION = True  # Fuse Conv2D + BatchNorm layers
    
    # Memory management
    MAX_POSE_HISTORY = 10  # Frames to keep in memory for temporal analysis
    
    # Processing settings
    BATCH_PROCESSING = False  # Set to True for batch processing (future feature)
    PARALLEL_PROCESSING = False  # Set to True for parallel processing (future feature)

# Logging Configuration
class LoggingConfig:
    # Log levels
    LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
    
    # Console output
    ENABLE_CONSOLE_LOGGING = True
    ENABLE_PROGRESS_BAR = True
    
    # File logging (future feature)
    ENABLE_FILE_LOGGING = False
    LOG_FILE_PATH = str(ProjectConfig.OUTPUTS_DIR / "pose_estimation.log")

# Easy access to all configurations
class Config:
    PROJECT = ProjectConfig
    POSE = PoseConfig
    VIDEO = VideoConfig
    COCO = COCOConfig
    ACTIVITY = ActivityConfig
    PERFORMANCE = PerformanceConfig
    LOGGING = LoggingConfig

# Utility function to print current configuration
def print_config():
    """Print current configuration settings"""
    print("🔧 CURRENT CONFIGURATION")
    print("=" * 50)
    print(f"📁 Project Base Directory: {Config.PROJECT.BASE_DIR}")
    print(f"🤖 Model Path: {Config.PROJECT.MODEL_PATH}")
    print(f"📊 Output Directory: {Config.PROJECT.OUTPUTS_DIR}")
    print(f"🎥 Video Source: {Config.VIDEO.VIDEO_SOURCE}")
    print(f"⚡ Confidence Threshold: {Config.POSE.CONFIDENCE_THRESHOLD}")
    print(f"🏃 Movement Threshold (Walking): {Config.POSE.MOVEMENT_THRESHOLD_WALKING}")
    print(f"🏃 Movement Threshold (Running): {Config.POSE.MOVEMENT_THRESHOLD_RUNNING}")
    print("=" * 50)

if __name__ == "__main__":
    print_config()
