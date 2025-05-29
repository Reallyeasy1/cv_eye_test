"""
Module for estimating the distance between a camera and a person's face using facial landmarks.
This module uses MediaPipe's Face Mesh solution to detect facial landmarks and calculate distance
based on the known average distance between human eyes.
"""

import mediapipe as mp
import numpy as np
from eye_test_cv.config.settings import (
    KNOWN_FACE_WIDTH, MIN_DISTANCE_CM, MAX_DISTANCE_CM
)

mp_face_mesh = mp.solutions.face_mesh

class DistanceEstimator:
    """
    A class to estimate the distance between a camera and a person's face.
    
    This class uses MediaPipe's Face Mesh to detect facial landmarks and calculates
    the distance based on the known average distance between human eyes and the
    focal length of the camera.

    Attributes:
        face_mesh (mp_face_mesh.FaceMesh): MediaPipe Face Mesh instance for facial landmark detection
        focal_length_px (float): The focal length of the camera in pixels
    """

    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            max_num_faces=1
        )
        self.focal_length_px = None

    def set_focal_length(self, focal_length_px):
        """
        Set the focal length of the camera in pixels.

        Args:
            focal_length_px (float): The focal length in pixels
        """
        self.focal_length_px = focal_length_px

    def estimate(self, frame_rgb, frame_width):
        """
        Estimate the distance between the camera and the detected face.

        Uses the distance between eyes as reference points for distance calculation.
        The calculation is based on the principle of similar triangles using a known
        average face width as reference.

        Args:
            frame_rgb (numpy.ndarray): RGB image frame to process
            frame_width (int): Width of the frame in pixels

        Returns:
            tuple: Contains:
                - float: Estimated distance in centimeters (0 if estimation fails)
                - str: Status message indicating the result or error
                - tuple: RGB color code for visual feedback (green for good, red for too close, etc.)
        """
        if not self.focal_length_px:
            return 0, "FOCAL LENGTH NOT SET", (255, 255, 255)

        results = self.face_mesh.process(frame_rgb)
        if not results.multi_face_landmarks:
            return 0, "NO FACE", (255, 255, 255)

        try:
            face_landmarks = results.multi_face_landmarks[0]
            left_eye = (face_landmarks.landmark[33].x * frame_width, 
                        face_landmarks.landmark[33].y * frame_width)
            right_eye = (face_landmarks.landmark[263].x * frame_width, 
                         face_landmarks.landmark[263].y * frame_width)
            
            face_width_pixels = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
            if face_width_pixels <= 0:
                return 0, "INVALID FACE", (255, 255, 255)

            distance_cm = (KNOWN_FACE_WIDTH * self.focal_length_px) / face_width_pixels

            if distance_cm < MIN_DISTANCE_CM:
                return distance_cm, "TOO CLOSE!", (0, 0, 255)
            elif distance_cm > MAX_DISTANCE_CM:
                return distance_cm, "TOO FAR!", (0, 165, 255)
            return distance_cm, "GOOD DISTANCE", (0, 255, 0)
        except Exception as e:
            return 0, "ESTIMATION ERROR", (255, 255, 255)

    def close(self):
        """
        Release the MediaPipe Face Mesh resources.
        Should be called when the estimator is no longer needed.
        """
        self.face_mesh.close()