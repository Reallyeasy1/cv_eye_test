import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

class Display:
    def __init__(self, window_name='Posture & Distance Analysis'):
        self.window_name = window_name
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def update(self, frame, posture_data, distance_data, eye_status, camera_specs, 
              pose_landmarks=None, face_landmarks=None, metrics_summary=None):
        """Update display with frame and all analysis results."""
        annotated_image = frame.copy()
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]

        # Draw landmarks if available
        if pose_landmarks:
            self.draw_pose_landmarks(annotated_image, pose_landmarks, (0, 255, 0))
        if face_landmarks:
            self.draw_face_landmarks(annotated_image, face_landmarks)

        # Posture info
        if posture_data:
            posture_status, posture_color, vert_diff, horiz_diff = posture_data
            cv2.putText(annotated_image, f"Posture: {posture_status}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, posture_color, 2)
            cv2.putText(annotated_image, f"Head Tilt: {vert_diff:.3f}",
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(annotated_image, f"Forward Lean: {horiz_diff:.3f}",
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Distance info
        if distance_data:
            distance, distance_status, distance_color = distance_data
            cv2.putText(annotated_image, f"Distance: {distance:.1f} cm",
                       (frame_width - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, distance_color, 2)
            cv2.putText(annotated_image, distance_status,
                       (frame_width - 250, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, distance_color, 2)

        # Eye status
        if eye_status:
            status_color = (0, 255, 0)  # Green for active tracking
            cv2.putText(annotated_image, f"Eye Status: {eye_status}",
                       (10, frame_height - 140), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, status_color, 2)

        # Performance metrics
        if metrics_summary:
            self.draw_metrics(annotated_image, metrics_summary, frame_width, frame_height)

        # Camera specs
        cv2.putText(annotated_image, 
                   f"Focal: {camera_specs['focal_length']}mm (35mm eq.) | Sensor: {camera_specs['sensor_width']}mm",
                   (10, frame_height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow(self.window_name, annotated_image)

    def draw_metrics(self, image, metrics, frame_width, frame_height):
        """Draw performance metrics on the image."""
        # FPS
        fps_color = (0, 255, 0) if metrics['fps'] >= 25 else (0, 165, 255) if metrics['fps'] >= 15 else (0, 0, 255)
        cv2.putText(image, f"FPS: {metrics['fps']:.1f}",
                   (frame_width - 150, frame_height - 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)

        # Processing times
        y_offset = frame_height - 90
        for op, latency in metrics['latencies'].items():
            color = (200, 200, 200)
            if op == 'total':
                color = (0, 255, 0) if latency < 33 else (0, 165, 255) if latency < 66 else (0, 0, 255)
            
            cv2.putText(image, f"{op}: {latency:.1f}ms",
                       (frame_width - 200, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 20

        # Detection rates
        y_offset = frame_height - 90
        for det_type, rate in metrics['detection_rates'].items():
            color = (0, 255, 0) if rate > 90 else (0, 165, 255) if rate > 75 else (0, 0, 255)
            cv2.putText(image, f"{det_type}: {rate:.1f}%",
                       (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 20

    def draw_pose_landmarks(self, image, landmarks, color):
        """Draw pose landmarks on the image."""
        mp_drawing.draw_landmarks(
            image,
            landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=color, thickness=2)
        )

    def draw_face_landmarks(self, image, face_landmarks):
        """Draw face mesh landmarks on the image."""
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
        )

    def close(self):
        """Close all windows."""
        cv2.destroyAllWindows()