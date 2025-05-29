import mediapipe as mp
import numpy as np
from collections import deque

class EyeTracker:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Enhanced MediaPipe indices for left eye (including more contour points)
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        # Enhanced MediaPipe indices for right eye (including more contour points)
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

        # Buffers for temporal smoothing
        self.left_ear_buffer = deque(maxlen=5)
        self.right_ear_buffer = deque(maxlen=5)
        
        # Dynamic thresholding
        self.baseline_ears = {'left': None, 'right': None}
        self.calibration_frames = 0
        self.required_calibration_frames = 30
        
        # Initial threshold (will be adjusted during calibration)
        self.EAR_THRESHOLD = 0.2
        self.calibrated = False

    def calculate_ear(self, landmarks, eye_indices):
        """Calculate the enhanced eye aspect ratio (EAR) using more points."""
        points = []
        for i in eye_indices:
            point = landmarks[i]
            points.append([point.x, point.y])
        points = np.array(points)
        
        # Calculate the mean height using multiple points
        heights = []
        for i in range(4):
            h = np.linalg.norm(points[i+1] - points[-i-2])
            heights.append(h)
        height = np.mean(heights)
        
        # Calculate the eye width
        width = np.linalg.norm(points[0] - points[8])
        
        # Calculate eye aspect ratio
        ear = height / width if width > 0 else 0
        return ear

    def update_calibration(self, left_ear, right_ear):
        """Update calibration values for dynamic thresholding."""
        if self.calibration_frames < self.required_calibration_frames:
            if self.baseline_ears['left'] is None:
                self.baseline_ears['left'] = []
                self.baseline_ears['right'] = []
            
            self.baseline_ears['left'].append(left_ear)
            self.baseline_ears['right'].append(right_ear)
            self.calibration_frames += 1
            
            if self.calibration_frames == self.required_calibration_frames:
                # Calculate baseline EAR values
                left_baseline = np.mean(self.baseline_ears['left'])
                right_baseline = np.mean(self.baseline_ears['right'])
                
                # Set threshold as percentage of baseline
                self.EAR_THRESHOLD = min(left_baseline, right_baseline) * 0.75
                self.calibrated = True
                return True
        return False

    def get_smoothed_ear(self, current_ear, buffer):
        """Apply temporal smoothing to EAR values."""
        buffer.append(current_ear)
        return np.median(buffer)

    def analyze(self, frame_rgb):
        results = self.face_mesh.process(frame_rgb)
        if not results.multi_face_landmarks:
            return "NO FACE DETECTED", None, None, False
        
        face_landmarks = results.multi_face_landmarks[0].landmark
        
        # Calculate EAR for both eyes
        left_ear = self.calculate_ear(face_landmarks, self.LEFT_EYE)
        right_ear = self.calculate_ear(face_landmarks, self.RIGHT_EYE)
        
        # Update calibration if needed
        just_calibrated = False
        if not self.calibrated:
            just_calibrated = self.update_calibration(left_ear, right_ear)
            if not self.calibrated:
                return "CALIBRATING... KEEP EYES OPEN", results.multi_face_landmarks[0], (left_ear, right_ear), False
        
        # Apply temporal smoothing
        smoothed_left_ear = self.get_smoothed_ear(left_ear, self.left_ear_buffer)
        smoothed_right_ear = self.get_smoothed_ear(right_ear, self.right_ear_buffer)
        
        # Determine eye states using smoothed values
        left_closed = smoothed_left_ear < self.EAR_THRESHOLD
        right_closed = smoothed_right_ear < self.EAR_THRESHOLD
        
        # Determine status message
        if left_closed and right_closed:
            status = "BOTH EYES CLOSED"
        elif left_closed:
            status = "LEFT EYE CLOSED"
        elif right_closed:
            status = "RIGHT EYE CLOSED"
        else:
            status = "EYES OPEN"
        
        if just_calibrated:
            status = "CALIBRATION COMPLETE - " + status
            
        return status, results.multi_face_landmarks[0], (smoothed_left_ear, smoothed_right_ear), self.calibrated

    def close(self):
        self.face_mesh.close() 