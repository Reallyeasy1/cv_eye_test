import os
import sys
import time
import logging
from typing import Dict, Any
from pathlib import Path
import cv2
import numpy as np
from eye_test_cv.controller import PostureDistanceDetector
from eye_test_cv.benchmarks import PerformanceBenchmark

logger = logging.getLogger(__name__)

class BenchmarkRunner:
    def __init__(self, test_data_dir: str = "test_data"):
        self.test_data_dir = Path(test_data_dir)
        self.benchmark = PerformanceBenchmark()
        
        if not self.test_data_dir.exists():
            logger.warning(f"Test data directory {self.test_data_dir} does not exist")
        
    def run_single_image_test(self, image_filename: str) -> Dict[str, Any]:
        """Run benchmark on a single image."""
        # Initialize detector
        detector = PostureDistanceDetector()
        # Enables logging
        detector.enableLogging() 

        # Enables metrics
        detector.enableMetrics()

        image_path = self.test_data_dir / image_filename
        
        logger.info(f"Attempting to load image from: {image_path.absolute()}")
       
        if not image_path.exists():
          
            alternative_paths = [
                Path(image_filename), 
                Path("main") / self.test_data_dir / image_filename, 
                Path("..") / self.test_data_dir / image_filename, 
            ]
            
            for alt_path in alternative_paths:
                logger.info(f"Trying alternative path: {alt_path.absolute()}")
                if alt_path.exists():
                    image_path = alt_path
                    break
            else:
                raise ValueError(f"Could not find image file. Tried paths:\n" + 
                               f"  - {image_path.absolute()}\n" + 
                               "\n".join([f"  - {p.absolute()}" for p in alternative_paths]))
        
        # Load test image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image from {image_path}. File exists but OpenCV failed to read it.")
            
        logger.info(f"Successfully loaded image: {image.shape}")
        
        # Warm-up run
        for _ in range(5):
            detector.run_single_frame(image)
            
        # Actual test runs
        num_runs = 50
        start_time = time.time()
        
        for i in range(num_runs):
            frame_start = time.time()
            
            # Process frame and collect metrics
            results = detector.run_single_frame(image)
            
            # Measure system resources
            self.benchmark.measure_system_resources()
            
            # Measure model performance
            # Note: These values would need to be calculated based on ground truth
            self.benchmark.measure_model_performance(
                inference_time=time.time() - frame_start,
                true_positives=1, 
                false_positives=0,
                false_negatives=0,
                calibration_error=0.1
            )
            
            # Measure real-time performance
            self.benchmark.measure_realtime_performance(
                total_frames=i + 1,
                dropped_frames=0,  
                queue_length=0,
                buffer_size=1,
                buffer_used=1,
                start_time=frame_start,
                end_time=time.time()
            )
            
        end_time = time.time()
        total_time = end_time - start_time
        
        # Get performance summary
        summary = self.benchmark.get_performance_summary()
        summary['total_time'] = total_time
        summary['average_fps'] = num_runs / total_time
        
        return summary

    def run_video_test(self, video_filename: str) -> Dict[str, Any]:
        """Run benchmark on a video file."""
        detector = PostureDistanceDetector()

        detector.enableLogging()
        detector.enableMetrics()
        
        video_path = self.test_data_dir / video_filename

        
        logger.info(f"Attempting to load video from: {video_path.absolute()}")
        
        
        if not video_path.exists():
           
            alternative_paths = [
                Path(video_filename),  
                Path("main") / self.test_data_dir / video_filename, 
                Path("..") / self.test_data_dir / video_filename, 
            ]
            
            for alt_path in alternative_paths:
                if alt_path.exists():
                    video_path = alt_path
                    break
            else:
                raise ValueError(f"Could not find video file. Tried paths:\n" + 
                               f"  - {video_path.absolute()}\n" + 
                               "\n".join([f"  - {p.absolute()}" for p in alternative_paths]))
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dropped_frames = 0
        processed_frames = 0
        
        start_time = time.time()
        
        while cap.isOpened():
            frame_start = time.time()
            ret, frame = cap.read()
            
            if not ret:
                break
                

            try:
                results = detector.run_single_frame(frame)
                processed_frames += 1
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                dropped_frames += 1
                continue
                

            self.benchmark.measure_system_resources()
            
            self.benchmark.measure_model_performance(
                inference_time=time.time() - frame_start,
                true_positives=1, 
                false_positives=0,
                false_negatives=0,
                calibration_error=0.1
            )
            
            self.benchmark.measure_realtime_performance(
                total_frames=processed_frames + dropped_frames,
                dropped_frames=dropped_frames,
                queue_length=0,
                buffer_size=1,
                buffer_used=1,
                start_time=frame_start,
                end_time=time.time()
            )
            
        cap.release()
        end_time = time.time()
        
        summary = self.benchmark.get_performance_summary()
        summary.update({
            'total_time': end_time - start_time,
            'total_frames': total_frames,
            'processed_frames': processed_frames,
            'dropped_frames': dropped_frames,
            'average_fps': processed_frames / (end_time - start_time)
        })
        
        return summary

    def run_stress_test(self, duration_seconds: int = 300) -> Dict[str, Any]:
        """Run a stress test using webcam feed."""
        detector = PostureDistanceDetector()
        detector.enableLogging()
        detector.enableMetrics()
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            raise ValueError("Could not open webcam")
            
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        total_frames = 0
        dropped_frames = 0
        
        while time.time() < end_time:
            frame_start = time.time()
            ret, frame = cap.read()
            
            if not ret:
                dropped_frames += 1
                continue
                
            try:
                results = detector.run_single_frame(frame)
                total_frames += 1
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                dropped_frames += 1
                continue
                
     
            self.benchmark.measure_system_resources()
            
            self.benchmark.measure_model_performance(
                inference_time=time.time() - frame_start,
                true_positives=1,  
                false_positives=0,
                false_negatives=0,
                calibration_error=0.1
            )
            
            self.benchmark.measure_realtime_performance(
                total_frames=total_frames + dropped_frames,
                dropped_frames=dropped_frames,
                queue_length=0,
                buffer_size=1,
                buffer_used=1,
                start_time=frame_start,
                end_time=time.time()
            )
            
        cap.release()
        
        # Get performance summary
        summary = self.benchmark.get_performance_summary()
        summary.update({
            'total_time': time.time() - start_time,
            'total_frames': total_frames,
            'dropped_frames': dropped_frames,
            'average_fps': total_frames / duration_seconds
        })
        
        return summary

def main():
    logging.basicConfig(level=logging.INFO,  handlers=[
            logging.StreamHandler(sys.stdout),  # Prints to console
            # logging.FileHandler('benchmark.log'),  # (Optional) Also log to a file
        ])

    logger.info(f"Current working directory: {Path.cwd()}")
    
    runner = BenchmarkRunner()
    
    try:
        logger.info("Running single image test...")
        image_results = runner.run_single_image_test("image.jpg")
        runner.benchmark.log_performance_summary()
    except Exception as e:
        logger.error(f"Image test failed: {e}")
    
    try:
        logger.info("\nRunning video test...")
        video_results = runner.run_video_test("Test.mp4")
        runner.benchmark.log_performance_summary()
    except Exception as e:
        logger.error(f"Video test failed: {e}")
    
    try:
        logger.info("\nRunning stress test (5 minutes)...")
        stress_results = runner.run_stress_test(300)
        runner.benchmark.log_performance_summary()
    except Exception as e:
        logger.error(f"Stress test failed: {e}")

if __name__ == "__main__":
    main()

