import time
import logging
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)

class PerformanceMetrics:
    def __init__(self, window_size=30, detailed=True):
        """Initialize performance metrics tracking."""
        self.window_size = window_size
        self.detailed = detailed
        
      
        self.frame_times = deque(maxlen=window_size)
        self.fps_history = deque(maxlen=window_size)
        self.last_fps_time = time.time()
        self.frame_count = 0
        
        if detailed:
            self.processing_times = {
                'total': deque(maxlen=window_size),
                'eye_tracking': deque(maxlen=window_size),
                'posture': deque(maxlen=window_size),
                'distance': deque(maxlen=window_size)
            }
            
            self.detection_counts = {
                'face': {'success': 0, 'total': 0},
                'pose': {'success': 0, 'total': 0},
                'eyes': {'success': 0, 'total': 0}
            }
            
            self.model_latencies = {
                'face_mesh': deque(maxlen=window_size),
                'pose': deque(maxlen=window_size)
            }
        else:
            self.processing_times = {}
            self.detection_counts = {}
            self.model_latencies = {}

    def start_operation(self):
        """Start timing an operation."""
        return time.time() if self.detailed else None

    def end_operation(self, start_time, operation_name):
        """End timing an operation and record its duration."""
        if not self.detailed or start_time is None:
            return None
            
        duration = time.time() - start_time
        if operation_name in self.processing_times:
            self.processing_times[operation_name].append(duration)
        return duration

    def update_fps(self):
        """Update FPS calculation."""
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_fps_time
        
        if elapsed >= 1.0:
            fps = self.frame_count / elapsed
            self.fps_history.append(fps)
            self.frame_count = 0
            self.last_fps_time = current_time
            return fps
        return None

    def update_detection_status(self, detection_type, success):
        """Update detection success rates."""
        if not self.detailed:
            return
            
        self.detection_counts[detection_type]['total'] += 1
        if success:
            self.detection_counts[detection_type]['success'] += 1

    def get_detection_rates(self):
        """Calculate detection success rates."""
        if not self.detailed:
            return {}
            
        rates = {}
        for det_type, counts in self.detection_counts.items():
            if counts['total'] > 0:
                rate = (counts['success'] / counts['total']) * 100
                rates[det_type] = rate
            else:
                rates[det_type] = 0.0
        return rates

    def get_average_latencies(self):
        """Calculate average latencies for different operations."""
        if not self.detailed:
            return {}
            
        latencies = {}
        for op_name, times in self.processing_times.items():
            if times:
                avg_time = np.mean(times) * 1000  
                latencies[op_name] = avg_time
            else:
                latencies[op_name] = 0.0
        return latencies

    def get_metrics_summary(self):
        """Get a comprehensive summary of all metrics."""
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0
        
        if not self.detailed:
            return {'fps': avg_fps}
            
        latencies = self.get_average_latencies()
        detection_rates = self.get_detection_rates()
        
        return {
            'fps': avg_fps,
            'latencies': latencies,
            'detection_rates': detection_rates
        }

    def log_metrics(self):
        """Log current metrics to the logger."""
        metrics = self.get_metrics_summary()
        
        logger.info("Performance Metrics Summary:")
        logger.info(f"FPS: {metrics['fps']:.1f}")
        
        if self.detailed:
            logger.info("Processing Latencies (ms):")
            for op, lat in metrics['latencies'].items():
                logger.info(f"  {op}: {lat:.1f}ms")
            
            logger.info("Detection Success Rates (%):")
            for det_type, rate in metrics['detection_rates'].items():
                logger.info(f"  {det_type}: {rate:.1f}%")

    def reset(self):
        """Reset all metrics."""
        self.__init__(self.window_size, self.detailed) 