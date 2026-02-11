# Pose Estimation Module

🚀 **Professional Real-time Multi-person Pose Estimation with Human Activity Detection**

## 🎯 Project Overview

Advanced computer vision system that combines YOLOv8-Pose with intelligent human activity detection for real-time surveillance and analysis applications.

## ✨ Features

- 🎥 **Real-time Multi-person Pose Estimation** using YOLOv8-Pose
- 🏃 **Human Activity Detection**: Standing, Sitting, Walking, Running
- 🦴 **COCO-17 Keypoints** (17 joints per person)
- 📊 **Structured Data Export** (CSV format with activities)
- 🎬 **Video Output** with activity labels and confidence scores
- ⚡ **Performance Optimized** for real-time applications
- 🔧 **Easy Integration** with professional project structure

## 📁 Professional Project Structure

```
Pose Estimation Module/
├── 📋 main.py                 # Main entry point
├── ⚙️ config.py              # Configuration settings
├── 📦 setup.py               # Installation script
├── 📖 README.md              # This file
├── 📄 requirements.txt       # Dependencies
├── 🤖 models/                # Model files
│   └── yolov8n-pose.pt      # YOLOv8 pose model
├── 🧬 src/                   # Source code
│   └── pose_estimation.py    # Core pose estimation module
└── 📊 outputs/               # Generated files
    ├── pose_data.csv         # Keypoint data with activities
    └── pose_activity_output.mp4  # Output video
```

## 🚀 Quick Start

### 1. Installation
```bash
# Clone or download the project
cd "Pose Estimation Module"

# Install dependencies
pip install -r requirements.txt

# Or install with setup.py
pip install -e .
```

### 2. Run the System
```bash
# Simple execution
python main.py

# Or using the installed command
pose-estimation
```

### 3. View Results
- 📹 **Video Output**: `outputs/pose_activity_output.mp4`
- 📊 **Activity Data**: `outputs/pose_data.csv`

## 🏃 Human Activities Detected

### **Activity Classification Algorithm**
- **🧍 STANDING**: Knee angle 160-180°, minimal movement
- **🪑 SITTING**: Knee angle 70-120°, hips lower than standing
- **🚶 WALKING**: Standing pose + moderate ankle movement (5-20 pixels/frame)
- **🏃 RUNNING**: Standing pose + high ankle movement (>20 pixels/frame)

### **Detection Method**
1. **Spatial Analysis**: Joint angle calculations
2. **Temporal Analysis**: Movement intensity over 10-frame window
3. **Confidence Scoring**: Activity confidence (0.0-1.0)

## 🧬 COCO Keypoints (17 Joints)

| Index | Joint | Index | Joint |
|-------|-------|-------|-------|
| 0 | Nose | 9 | Left Wrist |
| 1 | Left Eye | 10 | Right Wrist |
| 2 | Right Eye | 11 | Left Hip |
| 3 | Left Ear | 12 | Right Hip |
| 4 | Right Ear | 13 | Left Knee |
| 5 | Left Shoulder | 14 | Right Knee |
| 6 | Right Shoulder | 15 | Left Ankle |
| 7 | Left Elbow | 16 | Right Ankle |
| 8 | Right Elbow | | |

## � Performance Metrics

- **Model**: YOLOv8n-Pose (lightweight, real-time)
- **Activities**: 4 types (Standing, Sitting, Walking, Running)
- **Input**: Webcam/Video file
- **Output**: Annotated video + CSV data
- **Performance**: ~6 FPS on CPU (optimized)

## 🔧 Configuration

Easy configuration through `config.py`:

```python
# Model settings
CONFIDENCE_THRESHOLD = 0.5
INPUT_SIZE = 640

# Activity detection
MOVEMENT_THRESHOLD_WALKING = 5.0
MOVEMENT_THRESHOLD_RUNNING = 20.0

# Visualization
KEYPOINT_COLOR = (0, 255, 0)  # Green
SKELETON_COLOR = (0, 0, 255)  # Red
```

## 📝 Data Format (CSV)

| frame_id | person_id | joint_id | joint_name | x | y | confidence | activity | activity_confidence |
|----------|-----------|-----------|-------------|---|---|------------|-----------|-------------------|

## 🎯 Activity Detection Algorithm

### **Step 1: Pose Analysis**
- Calculate knee angles (hip-knee-ankle)
- Determine basic posture (standing vs sitting)

### **Step 2: Movement Analysis**
- Track ankle positions over 10 frames
- Calculate movement intensity

### **Step 3: Activity Classification**
- Apply spatial and temporal thresholds
- Assign confidence scores

## 🔌 Integration Examples

### **Basic Integration**
```python
from src.pose_estimation import PoseEstimator

# Initialize
estimator = PoseEstimator("models/yolov8n-pose.pt")

# Process frame
frame = cv2.imread("image.jpg")
annotated_frame, pose_data = estimator.process_frame(frame)

# Access activity data
for person in pose_data:
    print(f"Person {person['person_id']}: {person['activity']}")
```

### **Custom Configuration**
```python
from config import Config

# Modify settings
Config.POSE.CONFIDENCE_THRESHOLD = 0.7
Config.ACTIVITY.MIN_ACTIVITY_CONFIDENCE = 0.8

# Use custom config
estimator = PoseEstimator()
```

## 🛠️ Development

### **Project Structure Benefits**
- ✅ **Modular Design**: Easy to extend and modify
- ✅ **Professional Layout**: Industry-standard structure
- ✅ **Easy Integration**: Clear separation of concerns
- ✅ **Configuration Management**: Centralized settings
- ✅ **Scalable**: Ready for production deployment

### **Adding New Activities**
```python
# In config.py
ACTIVITIES['jumping'] = {
    'name': 'JUMPING',
    'color': (255, 0, 255),  # Magenta
    'description': 'Person is jumping'
}

# In pose_estimation.py
# Add detection logic in detect_activity() method
```

## 📦 Installation Options

### **Standard Installation**
```bash
pip install -r requirements.txt
python main.py
```

### **Development Installation**
```bash
pip install -e .
```

### **GPU Support**
```bash
pip install -e .[gpu]
```

## 🎯 Use Cases

- 🎥 **Video Surveillance**: Real-time activity monitoring
- 🏃 **Sports Analysis**: Athlete movement tracking
- 🏥 **Healthcare**: Patient monitoring and rehabilitation
- 🎮 **Gaming**: Motion capture and interaction
- 📊 **Research**: Human behavior analysis

## 📈 Performance Optimization

### **CPU Optimization**
- Reduced input size (416x416)
- Efficient skeleton drawing
- Optimized confidence filtering

### **GPU Acceleration** (Future)
- CUDA support
- Model optimization
- Batch processing

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Add tests
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Ultralytics**: YOLOv8 framework
- **COCO Dataset**: Keypoint annotations
- **OpenCV**: Computer vision operations
- **PyTorch**: Deep learning framework

## 📞 Support

For questions and support:
- 📧 Email: your.email@example.com
- 🐛 Issues: GitHub Issues
- 📖 Documentation: Project Wiki

---

**🚀 Ready for production use and easy integration!**
