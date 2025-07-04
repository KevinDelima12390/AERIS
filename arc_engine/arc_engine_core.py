import cv2
from utils.logger import get_logger
from communication.database_manager import DatabaseManager
from arc_engine.video_stream_processor import VideoStreamProcessor
from arc_engine.human_detection_module import HumanDetectionModule
from arc_engine.target_tracking_module import TargetTrackingModule
from arc_engine.facial_recognition_module import FacialRecognitionModule

class ARCEngineCore:
    """
    The core of the Autonomous Robot Custodian (ARC) system, responsible for
    integrating and orchestrating all perception and decision-making modules.
    """

    def __init__(self, video_source=0):
        """
        Initializes the ARCEngineCore, setting up all necessary modules.

        Args:
            video_source (int or str): The video source for the VideoStreamProcessor.
        """
        self.logger = get_logger(__name__)
        self.logger.info("Initializing ARCEngineCore...")

        # Initialize modules
        self.db_manager = DatabaseManager()
        self.video_stream_processor = VideoStreamProcessor(source=video_source)
        self.human_detection_module = HumanDetectionModule()
        self.target_tracking_module = TargetTrackingModule()
        self.facial_recognition_module = FacialRecognitionModule(self.db_manager)

        # Initialize state variables
        self.is_tracking = False
        self.identified_person = None
        
        self.logger.info("ARCEngineCore initialized successfully.")

    def process_frame(self):
        """
        Processes a single frame from the video stream.
        """
        status, frame = self.video_stream_processor.read_frame()
        if not status:
            self.logger.warning("Failed to read frame from video stream.")
            return False, None, "Failed to read frame", None

        message = "Status: Idle"
        name = None

        if not self.is_tracking:
            detections = self.human_detection_module.detect_humans(frame)
            if detections:
                if self.target_tracking_module.select_target(frame, detections):
                    self.is_tracking = True
                    message = "Status: Target Acquired"
                    self.logger.info("Target selected. Starting tracking.")
        else:
            success, bbox = self.target_tracking_module.update_tracker(frame)
            if success:
                message = "Status: Tracking Target"
                # Draw tracking box
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

                # Facial recognition
                tracked_bbox_xyxy = (int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                name, face_bbox = self.facial_recognition_module.recognize_face(frame, tracked_bbox_xyxy)
                if name:
                    self.identified_person = name
                    message = f"Recognized: {name}"
                    self.logger.info(f"Recognized: {self.identified_person}")
                    if face_bbox:
                        cv2.rectangle(frame, (face_bbox[0], face_bbox[1]), (face_bbox[2], face_bbox[3]), (0, 255, 0), 2)
                        cv2.putText(frame, name, (face_bbox[0], face_bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                self.logger.warning("Tracker update failed. Re-acquiring target.")
                self.target_tracking_module.reacquire_target()
                self.is_tracking = False
                message = "Status: Target Lost, Re-acquiring"

        return True, frame, message, name

    def shutdown(self):
        """
        Gracefully shuts down the ARC engine, releasing resources.
        """
        self.logger.info("Shutting down ARCEngineCore.")
        self.video_stream_processor.release()
        self.db_manager.close_connection()
        # cv2.destroyAllWindows() is removed as GUI handles windows
        self.logger.info("ARCEngineCore shutdown complete.")

if __name__ == '__main__':
    # This is for testing purposes.
    # It allows running this module directly to test the core engine.
    engine = ARCEngineCore(video_source=0)
    engine.run()