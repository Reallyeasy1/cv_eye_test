import time
import logging
from eye_test_cv.models.camera import Camera
from eye_test_cv.models.posture import PostureAnalyzer
from eye_test_cv.models.distance import DistanceEstimator
from eye_test_cv.models.eye_tracker import EyeTracker
from eye_test_cv.models.metrics import PerformanceMetrics
from eye_test_cv.views.display import Display
import cv2
from eye_test_cv.config.settings import (
    FOCAL_LENGTH_35MM, SENSOR_WIDTH_35MM, IMAGE_WIDTH_PX,
    KNOWN_FACE_WIDTH, FOCAL_LENGTH_PX
)

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

logger = logging.getLogger(__name__)

class PostureDistanceDetector:
    _logging_enabled = False
    _metrics_enabled = False

    @classmethod
    def enableLogging(cls):
        """Enable detailed logging for the application."""
        cls._logging_enabled = True
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Detailed logging enabled")

    @classmethod
    def enableMetrics(cls):
        """Enable detailed metrics tracking for the application."""
        cls._metrics_enabled = True
        logger.info("Detailed metrics enabled")

    def __init__(self, auto_calibrate=False, gender='average', face_width=None, camera_source=0):
        self.camera = Camera(camera_source)
        self.posture_analyzer = PostureAnalyzer()
        self.eye_tracker = EyeTracker()
        self.display = Display()
        
        # Initialize metrics based on class setting
        self.metrics = PerformanceMetrics(detailed=True) if self._metrics_enabled else None
        
        # Camera specs
        self.focal_length_mm = FOCAL_LENGTH_35MM
        self.sensor_width_mm = SENSOR_WIDTH_35MM
        self.image_width_px = IMAGE_WIDTH_PX
        
        # To improve in the future, set up the formula below
        # self.focal_length_px = self.calculate_focal_length_px()
        # logger.debug(f"Initial focal length calculation: {self.focal_length_px:.2f}px")
        
        
        self.focal_length_px = FOCAL_LENGTH_PX
        # Calibration settings
        self.auto_calibrate = auto_calibrate
        self.gender = gender
        self.face_width = face_width or KNOWN_FACE_WIDTH

        
        # Initialize distance estimator
        self.distance_estimator = DistanceEstimator()

    def setup_distance_estimation(self):
        """Handle the setup of the distance estimator."""
        if self.auto_calibrate:
            logger.info("Starting auto-calibration...")
            calibration_successful = False
            
            while not calibration_successful and not cv2.waitKey(1) & 0xFF == ord('q'):
                ret, frame = self.camera.read_frame()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    try:
                    
                        calibration_successful = True
                        logger.info("Auto-calibration successful!")
                    except Exception as e:
                        logger.error(f"Calibration error: {str(e)}")
                        cv2.imshow('Calibration', frame)
            
            cv2.destroyAllWindows()
        else:
            logger.info("Using manual calibration settings...")
            self.calculate_focal_length_px()
        
        self.distance_estimator.set_focal_length(self.focal_length_px)
        logger.info(f"Distance estimator initialized with focal length: {self.focal_length_px:.2f}px")
        logger.info(f"Using face width: {self.face_width}cm")

    def calculate_focal_length_px(self):
        logger.info(f"Calculated Focal Length: {self.focal_length_px:.2f} px")
        return (self.image_width_px * self.focal_length_mm) / self.sensor_width_mm

    def run(self):
        if not self.camera.initialize():
            return
            
        self.setup_distance_estimation()

        try:
            while True:
                frame_start = self.metrics.start_operation() if self.metrics else None
                
                # Frame capture and basic processing
                ret, frame = self.camera.read_frame()
                if not ret:
                    logger.error("Failed to capture frame")
                    break

                # Resize frame for better performance
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Update FPS if metrics enabled
                if self.metrics:
                    current_fps = self.metrics.update_fps()
                    if current_fps:
                        logger.debug(f"FPS: {current_fps:.1f}")

                # Process all models on every frame
                # Eye tracking
                eye_start = self.metrics.start_operation() if self.metrics else None
                eye_status, face_landmarks, ear_values, is_calibrated = self.eye_tracker.analyze(frame_rgb)
                if self.metrics:
                    self.metrics.end_operation(eye_start, 'eye_tracking')
                    self.metrics.update_detection_status('face', eye_status != "NO FACE")
                
                # Posture analysis
                posture_start = self.metrics.start_operation() if self.metrics else None
                posture_status, posture_color, vert_diff, horiz_diff, pose_landmarks = self.posture_analyzer.analyze(
                    frame_rgb, FRAME_WIDTH, FRAME_HEIGHT)
                if self.metrics:
                    self.metrics.end_operation(posture_start, 'posture')
                    self.metrics.update_detection_status('pose', posture_status != "No pose detected")
                
                # Distance estimation
                distance_start = self.metrics.start_operation() if self.metrics else None
                distance_data = self.distance_estimator.estimate(frame_rgb, FRAME_WIDTH)
                if self.metrics:
                    self.metrics.end_operation(distance_start, 'distance')

                # Use the results directly since we're processing every frame
                posture_data = (posture_status, posture_color, vert_diff, horiz_diff)

                # Update display with available data
                camera_specs = {
                    'focal_length': self.focal_length_mm,
                    'sensor_width': self.sensor_width_mm
                }
                
                # End total frame processing time
                if self.metrics:
                    total_time = self.metrics.end_operation(frame_start, 'total')
                    
                    # Log metrics periodically
                    if self.metrics.frame_count % 30 == 0:
                        self.metrics.log_metrics()

                # Update display with performance metrics
                metrics_summary = self.metrics.get_metrics_summary() if self.metrics else None
                self.display.update(
                    frame,
                    posture_data,
                    distance_data,
                    eye_status,
                    camera_specs,
                    pose_landmarks,
                    face_landmarks,
                    metrics_summary
                )

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            logger.exception("Error in main loop")
            raise e
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources and log final metrics."""
        self.camera.release()
        self.posture_analyzer.close()
        if self.distance_estimator:
            self.distance_estimator.close()
        self.eye_tracker.close()
        
        # Log final metrics if enabled
        if self.metrics:
            logger.info("\nFinal Performance Metrics:")
            self.metrics.log_metrics()
        
        self.display.close()
        logger.info("Application shutdown complete")

    def run_single_frame(self, frame):
        """Process a single frame and return the results.
        
        This method is primarily used for benchmarking and testing.
        It processes a single frame through all detection pipelines
        and returns the results without displaying them.
        
        Args:
            frame: A numpy array containing the image frame to process
            
        Returns:
            dict: A dictionary containing all detection results
        """
        frame_start = self.metrics.start_operation() if self.metrics else None
        
        # Resize frame for better performance
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Eye tracking
        eye_start = self.metrics.start_operation() if self.metrics else None
        eye_status, face_landmarks, ear_values, is_calibrated = self.eye_tracker.analyze(frame_rgb)
        if self.metrics:
            self.metrics.end_operation(eye_start, 'eye_tracking')
            self.metrics.update_detection_status('face', eye_status != "NO FACE")
        
        # Posture analysis
        posture_start = self.metrics.start_operation() if self.metrics else None
        posture_status, posture_color, vert_diff, horiz_diff, pose_landmarks = self.posture_analyzer.analyze(
            frame_rgb, FRAME_WIDTH, FRAME_HEIGHT)
        if self.metrics:
            self.metrics.end_operation(posture_start, 'posture')
            self.metrics.update_detection_status('pose', posture_status != "No pose detected")
        
        # Distance estimation
        distance_start = self.metrics.start_operation() if self.metrics else None
        distance_data = self.distance_estimator.estimate(frame_rgb, FRAME_WIDTH)
        if self.metrics:
            self.metrics.end_operation(distance_start, 'distance')
        
        # End total frame processing time
        if self.metrics:
            total_time = self.metrics.end_operation(frame_start, 'total')
        
        return {
            'eye_tracking': {
                'status': eye_status,
                'ear_values': ear_values,
                'is_calibrated': is_calibrated
            },
            'posture': {
                'status': posture_status,
                'vertical_difference': vert_diff,
                'horizontal_difference': horiz_diff
            },
            'distance': distance_data,
            'landmarks': {
                'face': face_landmarks,
                'pose': pose_landmarks
            }
        }