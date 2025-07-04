import cv2
import numpy as np
from utils.logger import logger

class VideoStreamProcessor:
    """
    Handles video input, frame reading, and basic preprocessing.
    """
    def __init__(self, source, width=640, height=480):
        """
        Initializes the video stream processor.

        Args:
            source (int or str): The video source (webcam index or file path).
            width (int): The desired width for frame resizing.
            height (int): The desired height for frame resizing.
        """
        self.source = source
        self.width = width
        self.height = height
        self.capture = cv2.VideoCapture(self.source)
        if self.capture.isOpened():
            logger.info(f"Successfully opened video source: {self.source}")
        else:
            logger.error(f"Failed to open video source: {self.source}")

    def read_frame(self):
        """
        Reads and preprocesses a single frame from the video source.

        Returns:
            tuple: A tuple containing:
                - bool: True if a frame was successfully read, False otherwise.
                - numpy.ndarray or None: The processed frame, or None on failure.
        """
        status, frame = self.capture.read()
        if status:
            frame = cv2.resize(frame, (self.width, self.height))
            return True, frame
        return False, None

    def release(self):
        """
        Releases the video capture object.
        """
        if self.capture.isOpened():
            self.capture.release()
            logger.info(f"Released video source: {self.source}")