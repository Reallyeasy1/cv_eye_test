from eye_test_cv import PostureDistanceDetector

# Initialize detector
detector = PostureDistanceDetector(
    auto_calibrate=False,
    gender='male',
    face_width=None,
    camera_source=1
)

detector.enableLogging()
detector.enableMetrics()

# Run the detection system
detector.run()

