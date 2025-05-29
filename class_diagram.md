# HPB Eye Test Monitoring System - Class Diagram

```mermaid
classDiagram
    %% Main Application
    class MainApp {
        +__init__()
        +run()
    }

    %% PostureDistanceDetector
    class PostureDistanceDetector {
        -camera: Camera
        -distance_estimator: DistanceEstimator
        -eye_tracker: EyeTracker
        -posture_detector: PostureDetector
        -display: Display
        -is_running: bool
        +__init__()
        +start()
        +stop()
        +process_frame()
        -handle_distance_warning()
        -handle_eye_warning()
        -handle_posture_warning()
    }

    %% Models
    class Camera {
        -cap: cv2.VideoCapture
        -frame_width: int
        -frame_height: int
        +__init__(camera_id: int)
        +start()
        +stop()
        +read_frame() -> np.ndarray
        +is_opened() -> bool
        +release()
    }

    class DistanceEstimator {
        -face_mesh: mp.solutions.face_mesh.FaceMesh
        -focal_length_px: float
        +__init__()
        +set_focal_length(focal_length_px: float)
        +estimate(frame_rgb: np.ndarray, frame_width: int) -> Tuple[float, str, Tuple[int, int, int]]
        +close()
    }

    class EyeTracker {
        -face_mesh: mp.solutions.face_mesh.FaceMesh
        -last_blink_time: float
        -blink_counter: int
        +__init__()
        +process_frame(frame: np.ndarray) -> Tuple[bool, float]
        +calculate_eye_aspect_ratio(eye_landmarks) -> float
        +detect_blink(ear: float) -> bool
        +get_blink_rate() -> float
        +reset_blink_counter()
        +close()
    }

    class PostureDetector {
        -face_mesh: mp.solutions.face_mesh.FaceMesh
        +__init__()
        +detect_posture(frame: np.ndarray) -> Tuple[bool, str]
        +calculate_head_pose(face_landmarks) -> Tuple[float, float, float]
        +close()
    }

    class CameraCalibrator {
        -face_mesh: mp.solutions.face_mesh.FaceMesh
        +__init__()
        +calibrate_focal_length_with_reference(...) -> float
        +calibrate_focal_length_with_aruco(...) -> float
        +estimate_face_width_statistical(...) -> float
        +estimate_face_width_multi_distance(...) -> float
        +estimate_face_width_with_reference(...) -> float
        +close()
    }

    %% Views
    class Display {
        -window_name: str
        -frame: np.ndarray
        -distance_text: str
        -warning_text: str
        -warning_color: Tuple[int, int, int]
        +__init__()
        +update(frame: np.ndarray)
        +show_distance(distance: float, status: str)
        +show_warning(text: str, color: Tuple[int, int, int])
        +render()
        +close()
    }

    %% Configuration
    class Settings {
        +MIN_DISTANCE_CM: float
        +MAX_DISTANCE_CM: float
        +KNOWN_FACE_WIDTH: float
        +BLINK_THRESHOLD: float
        +MIN_BLINK_RATE: float
        +MAX_HEAD_TILT: float
        +CAMERA_ID: int
    }

    class LoggingConfig {
        +configure_logging()
    }

    %% Class Relationships
    MainApp *-- PostureDistanceDetector
    PostureDistanceDetector *-- Camera
    PostureDistanceDetector *-- DistanceEstimator
    PostureDistanceDetector *-- EyeTracker
    PostureDistanceDetector *-- PostureDetector
    PostureDistanceDetector *-- Display
    PostureDistanceDetector o-- Settings
    DistanceEstimator ..> CameraCalibrator : uses for calibration
    Camera ..> Settings : uses configuration
    DistanceEstimator ..> Settings : uses configuration
    EyeTracker ..> Settings : uses configuration
    PostureDetector ..> Settings : uses configuration
    Display ..> Settings : uses configuration
```
