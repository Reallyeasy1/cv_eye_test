# HPB Eye Test Monitoring System - Class Diagram

```mermaid
classDiagram
    %% Core Application
    class Application {
        -config: ConfigurationManager
        -monitoringFacade: MonitoringFacade
        -uiManager: UIManager
        -eventBus: EventBus
        +__init__()
        +initialize()
        +start()
        +stop()
    }

    %% Facade Pattern for Monitoring
    class MonitoringFacade {
        -cameraManager: CameraManager
        -monitoringPipeline: MonitoringPipeline
        -analysisManager: AnalysisManager
        +__init__()
        +start_monitoring()
        +stop_monitoring()
        +process_frame()
        +get_monitoring_status()
    }

    %% Observer Pattern - Event System
    class EventBus {
        -subscribers: Dict
        +subscribe(event_type: str, callback: Callable)
        +unsubscribe(event_type: str, callback: Callable)
        +publish(event_type: str, data: Any)
    }

    %% Factory Pattern for Monitors
    class MonitorFactory {
        +create_distance_monitor() -> IMonitor
        +create_eye_monitor() -> IMonitor
        +create_posture_monitor() -> IMonitor
    }

    %% Strategy Pattern for Analysis
    class AnalysisManager {
        -active_analyzers: List[IAnalyzer]
        +add_analyzer(analyzer: IAnalyzer)
        +remove_analyzer(analyzer: IAnalyzer)
        +analyze_frame(frame: np.ndarray)
    }

    %% Camera Management
    class CameraManager {
        -camera_devices: List[Camera]
        -active_camera: Camera
        +initialize_cameras()
        +switch_camera(camera_id: int)
        +get_frame() -> np.ndarray
        +release_all()
    }

    %% Pipeline Management
    class MonitoringPipeline {
        -monitors: List[IMonitor]
        -preprocessors: List[IPreprocessor]
        +add_monitor(monitor: IMonitor)
        +add_preprocessor(preprocessor: IPreprocessor)
        +process_frame(frame: np.ndarray)
    }

    %% Interfaces
    class IMonitor {
        <<interface>>
        +process(frame: np.ndarray)
        +get_status()
    }

    class IAnalyzer {
        <<interface>>
        +analyze(data: Any)
        +get_results()
    }

    class IPreprocessor {
        <<interface>>
        +preprocess(frame: np.ndarray)
    }

    %% UI Components
    class UIManager {
        -mainWindow: MainWindow
        -overlayManager: OverlayManager
        -notificationManager: NotificationManager
        +initialize_ui()
        +update_display(frame: np.ndarray)
        +show_notification(message: str)
    }

    class OverlayManager {
        -active_overlays: List[IOverlay]
        +add_overlay(overlay: IOverlay)
        +remove_overlay(overlay: IOverlay)
        +render_overlays(frame: np.ndarray)
    }

    %% Configuration Management
    class ConfigurationManager {
        -settings: Dict
        +load_config(path: str)
        +save_config(path: str)
        +get_setting(key: str)
        +update_setting(key: str, value: Any)
    }

    %% Data Management
    class DataLogger {
        -storage: IStorage
        +log_event(event_type: str, data: Dict)
        +get_logs(start_time: datetime, end_time: datetime)
        +export_logs(format: str)
    }

    class IStorage {
        <<interface>>
        +save(data: Any)
        +load(query: Dict)
        +delete(query: Dict)
    }

    %% Model Classes
    class EyeTracker {
        -face_mesh: mp.solutions.face_mesh.FaceMesh
        -LEFT_EYE: List[int]
        -RIGHT_EYE: List[int]
        -left_ear_buffer: deque
        -right_ear_buffer: deque
        -baseline_ears: Dict
        -calibration_frames: int
        -EAR_THRESHOLD: float
        -calibrated: bool
        +__init__()
        +calculate_ear(landmarks, eye_indices)
        +update_calibration(left_ear, right_ear)
        +get_smoothed_ear(current_ear, buffer)
        +analyze(frame_rgb)
        +close()
    }

    class PostureAnalyzer {
        -pose: mp.solutions.pose.Pose
        +__init__()
        +analyze(frame_rgb, frame_width, frame_height)
        +close()
    }

    class DistanceEstimator {
        -face_mesh: mp.solutions.face_mesh.FaceMesh
        -focal_length_px: float
        -face_width_cm: float
        -auto_calibrate: bool
        -calibrator: CameraCalibrator
        +__init__()
        +estimate(frame_rgb, frame_width)
        +close()
    }

    class CameraCalibrator {
        -face_mesh: mp.solutions.face_mesh.FaceMesh
        +__init__()
        +calibrate_focal_length_with_reference(frame, known_width_cm, known_distance_cm, reference_points)
        +estimate_face_width_multi_distance(frames, distances_cm, focal_length)
        +estimate_face_width_with_reference(frame, reference_width_cm, reference_points)
        +close()
    }

    %% Relationships
    Application *-- MonitoringFacade
    Application *-- UIManager
    Application *-- EventBus
    Application *-- ConfigurationManager
    MonitoringFacade *-- CameraManager
    MonitoringFacade *-- MonitoringPipeline
    MonitoringFacade *-- AnalysisManager
    MonitoringPipeline o-- IMonitor
    MonitoringPipeline o-- IPreprocessor
    AnalysisManager o-- IAnalyzer
    MonitorFactory ..> IMonitor
    UIManager *-- OverlayManager
    DataLogger o-- IStorage
    
    %% Model Relationships
    EyeTracker ..|> IAnalyzer
    PostureAnalyzer ..|> IAnalyzer
    DistanceEstimator ..|> IAnalyzer
    DistanceEstimator *-- CameraCalibrator
    MonitorFactory ..> EyeTracker
    MonitorFactory ..> PostureAnalyzer
    MonitorFactory ..> DistanceEstimator
```