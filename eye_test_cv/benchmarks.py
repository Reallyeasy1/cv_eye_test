import time
import psutil
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    cpu_percent: float
    memory_percent: float
    gpu_utilization: Optional[float]
    io_counters: Dict[str, int]

@dataclass
class ModelMetrics:
    inference_time: float
    detection_accuracy: float
    false_positive_rate: float
    false_negative_rate: float
    calibration_error: float

@dataclass
class RealTimeMetrics:
    frame_drop_rate: float
    processing_queue_length: int
    buffer_utilization: float
    end_to_end_latency: float

class PerformanceBenchmark:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.system_metrics_history: List[SystemMetrics] = []
        self.model_metrics_history: List[ModelMetrics] = []
        self.realtime_metrics_history: List[RealTimeMetrics] = []
        
    def measure_system_resources(self) -> SystemMetrics:
        """Measure current system resource utilization."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.Process().memory_percent()
        
        # GPU measurement would go here if GPU is used
        gpu_utilization = None
        
        io_counters = psutil.Process().io_counters()._asdict()
        
        metrics = SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            gpu_utilization=gpu_utilization,
            io_counters=io_counters
        )
        
        self.system_metrics_history.append(metrics)
        if len(self.system_metrics_history) > self.window_size:
            self.system_metrics_history.pop(0)
            
        return metrics
    
    def measure_model_performance(
        self, 
        inference_time: float,
        true_positives: int,
        false_positives: int,
        false_negatives: int,
        calibration_error: float
    ) -> ModelMetrics:
        """Measure model performance metrics."""
        total = true_positives + false_positives + false_negatives
        
        metrics = ModelMetrics(
            inference_time=inference_time,
            detection_accuracy=(true_positives / total) if total > 0 else 0,
            false_positive_rate=(false_positives / total) if total > 0 else 0,
            false_negative_rate=(false_negatives / total) if total > 0 else 0,
            calibration_error=calibration_error
        )
        
        self.model_metrics_history.append(metrics)
        if len(self.model_metrics_history) > self.window_size:
            self.model_metrics_history.pop(0)
            
        return metrics
    
    def measure_realtime_performance(
        self,
        total_frames: int,
        dropped_frames: int,
        queue_length: int,
        buffer_size: int,
        buffer_used: int,
        start_time: float,
        end_time: float
    ) -> RealTimeMetrics:
        """Measure real-time performance metrics."""
        metrics = RealTimeMetrics(
            frame_drop_rate=(dropped_frames / total_frames) if total_frames > 0 else 0,
            processing_queue_length=queue_length,
            buffer_utilization=(buffer_used / buffer_size) if buffer_size > 0 else 0,
            end_to_end_latency=(end_time - start_time) * 1000  # Convert to ms
        )
        
        self.realtime_metrics_history.append(metrics)
        if len(self.realtime_metrics_history) > self.window_size:
            self.realtime_metrics_history.pop(0)
            
        return metrics
    
    def get_performance_summary(self) -> Dict[str, float]:
        """Get a summary of all performance metrics."""
        if not any([self.system_metrics_history, 
                   self.model_metrics_history, 
                   self.realtime_metrics_history]):
            return {}
        
        # System metrics averages
        sys_metrics = np.mean([
            [m.cpu_percent, m.memory_percent] 
            for m in self.system_metrics_history
        ], axis=0)
        
        # Model metrics averages
        model_metrics = np.mean([
            [m.inference_time, m.detection_accuracy, 
             m.false_positive_rate, m.false_negative_rate,
             m.calibration_error]
            for m in self.model_metrics_history
        ], axis=0)
        
        # Real-time metrics averages
        rt_metrics = np.mean([
            [m.frame_drop_rate, m.processing_queue_length,
             m.buffer_utilization, m.end_to_end_latency]
            for m in self.realtime_metrics_history
        ], axis=0)
        
        return {
            "avg_cpu_percent": sys_metrics[0],
            "avg_memory_percent": sys_metrics[1],
            "avg_inference_time": model_metrics[0],
            "avg_detection_accuracy": model_metrics[1],
            "avg_false_positive_rate": model_metrics[2],
            "avg_false_negative_rate": model_metrics[3],
            "avg_calibration_error": model_metrics[4],
            "avg_frame_drop_rate": rt_metrics[0],
            "avg_queue_length": rt_metrics[1],
            "avg_buffer_utilization": rt_metrics[2],
            "avg_end_to_end_latency": rt_metrics[3]
        }
    
    def log_performance_summary(self):
        """Log a summary of performance metrics."""
        summary = self.get_performance_summary()
        if not summary:
            logger.warning("No performance metrics available to summarize")
            return
            
        logger.info("\nPerformance Summary:")
        logger.info("===================")
        
        logger.info("\nSystem Metrics:")
        logger.info(f"CPU Usage: {summary['avg_cpu_percent']:.1f}%")
        logger.info(f"Memory Usage: {summary['avg_memory_percent']:.1f}%")
        
        logger.info("\nModel Performance:")
        logger.info(f"Inference Time: {summary['avg_inference_time']:.2f}ms")
        logger.info(f"Detection Accuracy: {summary['avg_detection_accuracy']:.1f}%")
        logger.info(f"False Positive Rate: {summary['avg_false_positive_rate']:.1f}%")
        logger.info(f"False Negative Rate: {summary['avg_false_negative_rate']:.1f}%")
        logger.info(f"Calibration Error: {summary['avg_calibration_error']:.2f}")
        
        logger.info("\nReal-time Performance:")
        logger.info(f"Frame Drop Rate: {summary['avg_frame_drop_rate']:.1f}%")
        logger.info(f"Avg Queue Length: {summary['avg_queue_length']:.1f}")
        logger.info(f"Buffer Utilization: {summary['avg_buffer_utilization']:.1f}%")
        logger.info(f"End-to-end Latency: {summary['avg_end_to_end_latency']:.2f}ms") 