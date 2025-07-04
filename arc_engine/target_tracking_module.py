import cv2
from utils.logger import get_logger

logger = get_logger(__name__)

class TargetTrackingModule:
    """
    A module for tracking a selected target across video frames.
    """
    def __init__(self, tracker_type='CSRT'):
        """
        Initializes the TargetTrackingModule.

        Args:
            tracker_type (str): The type of OpenCV tracker to use (e.g., 'CSRT', 'KCF').
        """
        self.tracker_type = tracker_type
        self.tracker = None
        self.tracked_bbox = None
        self.tracker_initialization_frame = None

        # Mapping of tracker types to their constructors
        self.tracker_constructors = {
            'CSRT': cv2.TrackerCSRT_create,
            'KCF': cv2.TrackerKCF_create,
            # Add other trackers as needed
        }

    def select_target(self, frame, detections):
        """
        Selects the best target from a list of detections and initializes the tracker.

        Args:
            frame (numpy.ndarray): The video frame where detections were made.
            detections (list): A list of human detections.

        Returns:
            bool: True if a target was selected and the tracker was initialized, False otherwise.
        """
        if not detections:
            return False

        # Select the detection with the largest bounding box area
        best_detection = max(detections, key=lambda d: (d['box'][2] - d['box'][0]) * (d['box'][3] - d['box'][1]))
        
        bbox = best_detection['box']
        
        # Convert from [x1, y1, x2, y2] to [x, y, w, h]]
        self.tracked_bbox = tuple(map(int, (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])))
        
        # Initialize the tracker
        self.tracker = self.tracker_constructors[self.tracker_type]()
        self.tracker.init(frame, self.tracked_bbox)
        self.tracker_initialization_frame = frame.copy()
        
        logger.info(f"Initialized {self.tracker_type} tracker for target at {self.tracked_bbox}")
        return True

    def update_tracker(self, frame):
        """
        Updates the tracker with the new frame.

        Args:
            frame (numpy.ndarray): The current video frame.

        Returns:
            tuple: A tuple containing the success status (bool) and the new bounding box.
        """
        if self.tracker is None:
            return False, None

        success, bbox = self.tracker.update(frame)
        
        if success:
            self.tracked_bbox = bbox
        else:
            logger.warning("Tracker update failed. Target may be lost.")
            self.reacquire_target()
            
        return success, self.tracked_bbox

    def reacquire_target(self):
        """
        Resets the tracker to handle tracking failure.
        This forces re-detection on the next frame.
        """
        logger.info("Re-acquiring target. Resetting tracker.")
        self.tracker = None
        self.tracked_bbox = None
        self.tracker_initialization_frame = None