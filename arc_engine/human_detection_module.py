from ultralytics import YOLO
from utils.logger import logger

class HumanDetectionModule:
    """
    A module for detecting humans in video frames using a YOLOv8 model.
    """
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initializes the HumanDetectionModule.

        Args:
            model_path (str): The path to the YOLOv8 model file.
        """
        self.model = self._load_model(model_path)

    def _load_model(self, model_path):
        """
        Loads the YOLOv8 model.

        Args:
            model_path (str): The path to the model file.

        Returns:
            YOLO: The loaded YOLO model object, or None if loading fails.
        """
        try:
            model = YOLO(model_path)
            logger.info(f"Successfully loaded YOLO model from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load YOLO model from {model_path}: {e}")
            return None

    def detect_humans(self, frame):
        """
        Detects humans in a single video frame.

        Args:
            frame (numpy.ndarray): The video frame to process.

        Returns:
            list: A list of dictionaries, where each dictionary represents a
                  detected person and contains 'box', 'confidence', and 'class_id'.
        """
        if self.model is None:
            logger.warning("YOLO model is not loaded. Cannot perform detection.")
            return []

        results = self.model(frame, verbose=False)
        
        detected_humans = []
        
        # The result object contains detections for the frame
        # We iterate through each detection in the result
        for detection in results[0].boxes:
            class_id = int(detection.cls)
            
            # Class ID 0 corresponds to 'person' in the COCO dataset
            if class_id == 0:
                confidence = float(detection.conf)
                box = detection.xyxy[0].cpu().numpy().tolist() # Bounding box in xyxy format
                
                detected_humans.append({
                    'box': box,
                    'confidence': confidence,
                    'class_id': class_id
                })
                
        return detected_humans