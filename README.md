# HPB Eye Screening Module

A real-time computer vision system for monitoring eye test procedures, ensuring correct user posture, screen distance, and eye closure states.

## Overview

This system provides real-time monitoring and feedback for:
- User posture analysis
- Screen-to-face distance measurement
- Individual eye state tracking (left/right eye open/closed)
- Visual feedback and status indicators

## Project Structure

```
# Eye Test CV Project Structure

/eye_test_root_cv/
├── eye_test_cv/
│   ├── __pycache__/
│   ├── config/
│   │   ├── __pycache__/
│   │   ├── logging_config.py
│   │   └── settings.py
│   ├── models/
│   │   ├── __pycache__/
│   │   ├── calibration.py
│   │   ├── camera.py
│   │   ├── distance.py
│   │   ├── eye_tracker.py
│   │   ├── metrics.py
│   │   └── posture.py
│   ├── test_data/
│   │   ├── image.jpg
│   │   └── Test.mp4
│   └── views/
│       ├── __pycache__/
│       └── display.py
├── __init__.py
├── benchmarks.py
├── controller.py
├── main.py
├── run_benchmarks.py
├── eye_test_cv.egg-info/
├── benchmark_results.txt
├── class_diagram.md
├── class-diagram-2.md
├── LICENSE
├── README.md
├── requirements.txt
├── setup.py
└── test_import.py
```

## Features

### 1. Posture Analysis
- Head tilt detection
- Forward lean detection
- Uneven shoulders detection
- Real-time posture status feedback

### 2. Eye Tracking
- Individual eye state monitoring
- Dynamic calibration
- Temporal smoothing
- Real-time EAR (Eye Aspect Ratio) values

### 3. Distance Monitoring
- Screen-to-face distance measurement
- Configurable optimal distance range
- Real-time distance feedback

### 4. Visual Feedback
- Pose landmarks visualization
- Face mesh display
- Color-coded status indicators
- Real-time metrics display

## Requirements

- Python 3.9 or higher
- Dependencies (installed via requirements.txt):
  ```
  opencv-python >= 4.11.0
  mediapipe >= 0.10.21
  numpy >= 1.26.4
  pillow >= 11.2.1
  matplotlib >= 3.10.3
  ```

## Dependencies Details

### Core Functionality
- **OpenCV**: Camera handling, image processing
- **MediaPipe**: Pose estimation, face mesh detection
- **NumPy**: Numerical computations, array operations

### Visualization
- **Matplotlib**: Data visualization, status indicators
- **Pillow**: Image handling and processing

### Standard Library
- **logging**: Application logging
- **time**: Frame rate control
- **collections.deque**: Temporal smoothing
## Installation

1. Clone the repository:
```bash
git clone https://github.com/reallyeasy1/cv_govtech.git
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate # Unix/MacOS
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

1. Run the application:
```bash
python main.py
```

2. Follow the camera calibration prompts
3. Keep eyes open during initial calibration (~2 seconds)
4. Press 'q' to quit

## PostureDistanceDetector Class

### Class Overview
The `PostureDistanceDetector` class is the main controller that integrates various computer vision components for posture analysis, distance estimation, and eye tracking.

### Initialization
```python
detector = PostureDistanceDetector(
    auto_calibrate=False,  # Enable/disable automatic camera calibration
    gender='average',      # Gender for face width estimation ('male', 'female', 'average')
    face_width=None       # Known face width in cm (uses default if None)
)
```

### Example Configurations
```python
# Basic configuration with defaults
detector = PostureDistanceDetector()

# Full configuration with custom parameters
detector = PostureDistanceDetector(
    auto_calibrate=True,           # Enable automatic camera calibration
    gender='male',                 # Use male average face width
    face_width=14.5,              # Custom face width in cm
    camera_source=1              # Specify which camera to use (default: 1) 
)

# Configuration for known face width
detector = PostureDistanceDetector(
    auto_calibrate=False,
    gender='female',
    face_width=13.8,              # Specific face width measurement
    camera_source=0
)
```

### Class Methods

#### Configuration Methods
```python
@classmethod
def enableLogging():
    """Enable detailed logging for the application."""

@classmethod
def enableMetrics():
    """Enable detailed metrics tracking for the application."""
```

#### Core Methods

1. `setup_distance_estimation()`
   - Initializes the distance estimation system
   - Handles camera calibration (auto or manual)
   - Sets up focal length calculations

2. `calculate_focal_length_px()`
   - Calculates the focal length in pixels
   - Uses calibration data for accurate measurements

3. `run()`
   - Main application loop
   - Processes real-time video feed
   - Performs:
     - Eye tracking
     - Posture analysis
     - Distance estimation
   - Updates display with results
   - Handles performance metrics

4. `run_single_frame(frame)`
   - Processes a single frame
   - Returns a dictionary with all analysis results:
     ```python
     {
         'eye_tracking': {
             'status': str,
             'ear_values': float,
             'is_calibrated': bool
         },
         'posture': {
             'status': str,
             'vertical_difference': float,
             'horizontal_difference': float
         },
         'distance': tuple,  # (distance, status, color)
         'landmarks': {
             'face': landmarks_object,
             'pose': landmarks_object
         }
     }
     ```

5. `cleanup()`
   - Releases system resources
   - Closes camera connections
   - Logs final metrics

### Component Integration

*Note: To configure the settings, simply go to config/settings.py*
The class integrates several specialized components:

1. **Camera Handler** (`self.camera`)
   - Manages video capture
   - Handles frame acquisition
   - Resolution: 640x480 pixels

2. **Posture Analyzer** (`self.posture_analyzer`)
   - Detects head tilt
   - Monitors forward lean
   - Tracks shoulder alignment
   - Thresholds:
     - Head tilt: 0.005
     - Forward lean: 0.005
     - Shoulder difference: 0.01

3. **Eye Tracker** (`self.eye_tracker`)
   - Monitors individual eye states
   - Calculates Eye Aspect Ratio (EAR)
   - Provides calibrated eye measurements

4. **Distance Estimator** (`self.distance_estimator`)
   - Calculates user-to-screen distance
   - Optimal range: 145-155 cm
   - Uses face width for calibration (default: 14.0 cm)

5. **Display Handler** (`self.display`)
   - Renders visual feedback
   - Shows landmarks and measurements
   - Displays status indicators

## Configuration

### Camera Settings (35mm format equivalent)
- Default focal length: 35mm
- Default sensor width: 24mm
- Default image width: 1932px

*Note: These values can be customized in config/settings.py during startup for your specific camera.*

### Performance Monitoring

When metrics are enabled:
- Tracks FPS
- Measures processing times
- Monitors detection rates
- Logs performance statistics

### Usage Example

```python
from eye_test_cv import PostureDistanceDetector

# Initialize detector
detector = PostureDistanceDetector(
    auto_calibrate=True,
    gender='average'
)

# Enable detailed logging and metrics
detector.enableLogging()
detector.enableMetrics()

# Run the detection system
detector.run()
```

## Status Indicators

### Posture Status
- 🟢 "GOOD POSTURE"
- 🔴 "HEAD TILTED"
- 🟠 "LEANING FORWARD"
- 🟠 "UNEVEN SHOULDERS"

### Eye Status
- "EYES OPEN"
- "LEFT EYE CLOSED"
- "RIGHT EYE CLOSED"
- "BOTH EYES CLOSED"
- "CALIBRATING..."

### Distance Status
- "TOO CLOSE"
- "GOOD DISTANCE"
- "TOO FAR"

## Documentation

- **Class Diagram**: See [class_diagram.md](class_diagram.md)
- **Future Plans**: See [class-diagram-2.md](class-diagram-2.md)
- **Full Report**: [CV Project Report](https://docs.google.com/document/d/1JF74Isdvhy2UkuKJPs_AnoDaFP32D5ALjhGi5I3NC6M/edit?usp=sharing)

## Future Enhancements

### Camera Calibration
- Automated focal length calibration
- Computer vision-based calibration
- Camera parameter auto-detection
- Calibration profile storage

### User Experience
- Centralized input handling
- Intuitive configuration interface
- Customizable keyboard shortcuts
- Real-time parameter adjustments

### Distance Measurement
- Multi-factor distance estimation
- Face size prediction
- Machine learning integration
- Dynamic reference point adjustment

### Performance Optimizations
- Frame Rate Control
  - Cap framerate at 15-20 FPS for optimal performance
  - Adaptive frame rate based on device capabilities
  - Reduced power consumption on mobile devices

- Model Optimization (Expected Gains: 50-60% faster inference)
  - Model pruning to remove redundant neurons/weights
  - Integration with TensorFlow Model Optimization toolkit
  - Network architecture refinement for mobile deployment
  - Reduced model size (up to 4x smaller)
  - Lower RAM usage (approximately 40% reduction)

- Input Processing
  - Reduced input image dimensions (45% fewer pixels)
  - Optimized for tablet screens (e.g., Tab A8's 800x1280)
  - Prevention of auto-rotation overhead
  - Efficient frame buffering

- Mobile-First Optimizations
  - INT8/FP16 quantization using TensorFlow Lite
  - Model compression techniques
  - Battery-aware processing
  - Optimized memory management