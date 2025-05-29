"""
Module for camera and facial measurements calibration.
Provides tools for calculating camera focal length and estimating actual face dimensions.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, Optional

class CameraCalibrator:
    def __init__(self):
        """Initialize the calibrator with MediaPipe Face Mesh."""
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            max_num_faces=1
        )
        
    def calibrate_focal_length_with_reference(self, 
                                            frame: np.ndarray,
                                            known_width_cm: float,
                                            known_distance_cm: float,
                                            reference_points: Tuple[Tuple[int, int], Tuple[int, int]]) -> float:
        """
        Calibrate focal length using a reference object of known width at known distance.
        
        Args:
            frame: Image frame
            known_width_cm: Actual width of reference object in centimeters
            known_distance_cm: Distance of reference object from camera in centimeters
            reference_points: Two points marking the width of reference object ((x1,y1), (x2,y2))
            
        Returns:
            float: Calculated focal length in pixels
        """
        pixel_width = np.linalg.norm(np.array(reference_points[0]) - np.array(reference_points[1]))
        focal_length = (pixel_width * known_distance_cm) / known_width_cm
        return focal_length

    def calibrate_focal_length_with_aruco(self, frame: np.ndarray) -> float:
        """
        Calibrate focal length using ArUco markers of known size.
        
        Args:
            frame: Image frame containing ArUco marker
            
        Returns:
            float: Calculated focal length in pixels
        """
        # Create ArUco dictionary and detector
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        aruco_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        
        # Detect markers
        corners, ids, _ = detector.detectMarkers(frame)
        
        if ids is not None and len(ids) > 0:
            # Assuming marker size is 5cm and distance is 50cm (adjust as needed)
            MARKER_SIZE_CM = 5.0
            KNOWN_DISTANCE_CM = 50.0
            
            marker_points = corners[0][0]
            pixel_width = np.linalg.norm(marker_points[0] - marker_points[1])
            focal_length = (pixel_width * KNOWN_DISTANCE_CM) / MARKER_SIZE_CM
            return focal_length
            
        return None

    def estimate_face_width_statistical(self, 
                                     gender: str = 'average',
                                     age_group: str = 'adult') -> float:
        """
        Get statistical average face width based on demographics.
        
        Args:
            gender: 'male', 'female', or 'average'
            age_group: 'child', 'adult', or 'elderly'
            
        Returns:
            float: Estimated face width in centimeters
        """
        # Statistical averages based on anthropometric studies
        averages = {
            'male': {'adult': 15},
            'female': {'adult': 14.6},
            'average': {'adult': 15.2}
        }
        return averages.get(gender, {}).get(age_group, 15.2)

    def estimate_face_width_multi_distance(self,
                                        frames: list,
                                        distances_cm: list,
                                        focal_length: float) -> float:
        """
        Estimate actual face width using multiple measurements at different known distances.
        
        Args:
            frames: List of frames taken at different distances
            distances_cm: List of known distances for each frame
            focal_length: Camera focal length in pixels
            
        Returns:
            float: Estimated actual face width in centimeters
        """
        width_estimates = []
        
        for frame, distance in zip(frames, distances_cm):
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                frame_height, frame_width = frame.shape[:2]
                
                # Get eye positions
                left_eye = (face_landmarks.landmark[33].x * frame_width,
                          face_landmarks.landmark[33].y * frame_width)
                right_eye = (face_landmarks.landmark[263].x * frame_width,
                           face_landmarks.landmark[263].y * frame_width)
                
                pixel_width = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
                estimated_width = (pixel_width * distance) / focal_length
                width_estimates.append(estimated_width)
        
        if width_estimates:
            # Return median to reduce impact of outliers
            return np.median(width_estimates)
        return None

    def estimate_face_width_with_reference(self,
                                         frame: np.ndarray,
                                         reference_width_cm: float,
                                         reference_points: Tuple[Tuple[int, int], Tuple[int, int]]) -> float:
        """
        Estimate face width using a reference object of known width in the same frame.
        
        Args:
            frame: Image frame containing both face and reference object
            reference_width_cm: Actual width of reference object in centimeters
            reference_points: Two points marking width of reference object ((x1,y1), (x2,y2))
            
        Returns:
            float: Estimated face width in centimeters
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            frame_height, frame_width = frame.shape[:2]
            
            # Get eye positions
            left_eye = (face_landmarks.landmark[33].x * frame_width,
                       face_landmarks.landmark[33].y * frame_width)
            right_eye = (face_landmarks.landmark[263].x * frame_width,
                        face_landmarks.landmark[263].y * frame_width)
            
            # Calculate widths in pixels
            face_width_pixels = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
            reference_width_pixels = np.linalg.norm(np.array(reference_points[0]) - np.array(reference_points[1]))
            
            # Use ratio to estimate face width
            face_width_cm = (face_width_pixels * reference_width_cm) / reference_width_pixels
            return face_width_cm
            
        return None

    def close(self):
        """Release MediaPipe resources."""
        self.face_mesh.close() 