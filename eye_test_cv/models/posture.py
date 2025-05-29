import mediapipe as mp
import numpy as np
from eye_test_cv.config.settings import (
    HEAD_TILT_THRESHOLD, LEAN_FORWARD_THRESHOLD,
    SHOULDER_DIFF_THRESHOLD
)

mp_pose = mp.solutions.pose

class PostureAnalyzer:
    def __init__(self):
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.6, 
            min_tracking_confidence=0.6
        )

    def analyze(self, frame_rgb, frame_width, frame_height):
        results = self.pose.process(frame_rgb)
        if not results.pose_landmarks:
            return "NO POSE", (255, 255, 255), 0, 0, None

        landmarks = results.pose_landmarks.landmark
        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
        left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
        right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

        vertical_diff = abs((left_ear.x + right_ear.x)/2 - nose.x)
        horizontal_diff = (left_ear.y + right_ear.y)/2 - nose.y
        shoulder_diff = abs(left_shoulder.y - right_shoulder.y)

        dimension_factor = (frame_width / 640)

        if vertical_diff > HEAD_TILT_THRESHOLD * dimension_factor:
            return "HEAD TILTED", (0, 0, 255), vertical_diff, horizontal_diff, results.pose_landmarks
        elif horizontal_diff > LEAN_FORWARD_THRESHOLD * dimension_factor:
            return "LEANING FORWARD", (0, 165, 255), vertical_diff, horizontal_diff, results.pose_landmarks
        elif shoulder_diff > SHOULDER_DIFF_THRESHOLD * dimension_factor:
            return "UNEVEN SHOULDERS", (0, 100, 255), vertical_diff, horizontal_diff, results.pose_landmarks
        else:
            return "GOOD POSTURE", (0, 255, 0), vertical_diff, horizontal_diff, results.pose_landmarks

    def close(self):
        self.pose.close()