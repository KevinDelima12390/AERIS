import face_recognition
import cv2
import numpy as np
from utils.logger import logger
from communication.database_manager import DatabaseManager

class FacialRecognitionModule:
    """
    Handles face detection and recognition within a given bounding box.
    """
    def __init__(self, db_manager: DatabaseManager, similarity_threshold=0.6):
        """
        Initializes the FacialRecognitionModule.

        Args:
            db_manager (DatabaseManager): The database manager to access known faces.
            similarity_threshold (float): The threshold for face similarity.
        """
        self.db_manager = db_manager
        self.similarity_threshold = similarity_threshold
        self.known_face_encodings = []
        self.known_face_names = []
        self._load_known_faces()
        logger.info("FacialRecognitionModule initialized.")

    def _load_known_faces(self):
        """
        Loads known faces from the database into memory.
        """
        try:
            known_faces = self.db_manager.get_all_known_faces()
            if not known_faces:
                logger.warning("No known faces found in the database.")
                return

            for face in known_faces:
                # Assuming the embedding is stored as a blob/bytes that can be converted to a numpy array
                encoding = np.frombuffer(face.embedding, dtype=np.float64)
                self.known_face_encodings.append(encoding)
                self.known_face_names.append(face.name)
            
            logger.info(f"Loaded {len(self.known_face_names)} known faces from the database.")

        except Exception as e:
            logger.error(f"Error loading known faces from database: {e}")

    def recognize_face(self, frame, tracked_bbox):
        """
        Recognizes a face within the tracked bounding box.

        Args:
            frame (numpy.ndarray): The full video frame.
            tracked_bbox (tuple): The bounding box of the tracked person (x1, y1, x2, y2).

        Returns:
            tuple: A tuple containing the name of the identified person and the face's bounding box.
                   Returns (None, None) if no face is detected.
                   Returns ("Unknown", face_bbox) if the face is not recognized.
        """
        x1, y1, x2, y2 = tracked_bbox
        cropped_frame = frame[y1:y2, x1:x2]

        if cropped_frame.size == 0:
            logger.warning("Tracked bounding box resulted in an empty frame crop.")
            return None, None

        # Resize for consistency
        resized_frame = cv2.resize(cropped_frame, (200, 200), interpolation=cv2.INTER_AREA)

        # Find all face locations and encodings in the resized frame
        face_locations = face_recognition.face_locations(resized_frame)
        face_encodings = face_recognition.face_encodings(resized_frame, face_locations)

        if not face_encodings:
            return None, None

        # For simplicity, we process only the first detected face.
        face_encoding = face_encodings[0]
        top, right, bottom, left = face_locations[0]

        # The face bounding box is relative to the cropped frame, so we adjust it to the full frame.
        face_bbox = (left + x1, top + y1, right + x1, bottom + y1)

        if not self.known_face_encodings:
            return "Unknown", face_bbox

        # Compare the detected face with known faces
        matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=self.similarity_threshold)
        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
        
        name = "Unknown"
        if True in matches:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
                logger.info(f"Face recognized: {name} with distance {face_distances[best_match_index]}")

        return name, face_bbox