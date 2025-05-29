import cv2
import logging
from time import sleep
from eye_test_cv.config.settings import (
    MAX_CAMERA_ATTEMPTS, 
    CAMERA_BUFFER_SIZE, IMAGE_WIDTH_PX
)

logger = logging.getLogger(__name__)

class Camera:
    def __init__(self, camera_source=1):
        self.cap = None
        self.width = IMAGE_WIDTH_PX
        self.height = int(IMAGE_WIDTH_PX * 9/16)  # 16:9 aspect ratio
        self.camera_source = camera_source

    def initialize(self):
        for attempt in range(MAX_CAMERA_ATTEMPTS):
            try:
                self.cap = cv2.VideoCapture(self.camera_source)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)
                
                if self.cap.isOpened():
                    self._set_resolution()
                    logger.info(f"Camera connected at {self.width}x{self.height} using source: {self.camera_source}")
                    return True
                
                logger.warning(f"Attempt {attempt+1}: Couldn't connect to camera source: {self.camera_source}")
                self.release()
            except Exception as e:
                logger.error(f"Camera init error: {str(e)}")
            sleep(1)
        
        logger.error(f"Failed to connect to camera source {self.camera_source} after multiple attempts")
        return False

    def _set_resolution(self):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # Verify resolution
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read_frame(self):
        if not self.cap or not self.cap.isOpened():
            return False, None
        return self.cap.read()

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None