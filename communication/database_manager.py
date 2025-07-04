import sqlite3
import numpy as np
import io
from utils.logger import get_logger

class DatabaseManager:
    """
    Manages the SQLite database for storing and retrieving known face embeddings.
    """

    def __init__(self, db_path='data/known_faces.db'):
        """
        Initializes the DatabaseManager, connects to the database, and creates the table if it doesn't exist.

        Args:
            db_path (str): The path to the SQLite database file.
        """
        self.db_path = db_path
        self.logger = get_logger(__name__)
        self.conn = None
        self.cursor = None
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.cursor = self.conn.cursor()
            self._create_table()
            self.logger.info(f"Successfully connected to the database at {self.db_path}")
        except sqlite3.Error as e:
            self.logger.error(f"Database error: {e}")
            raise

    def _create_table(self):
        """
        Creates the 'known_faces' table if it does not already exist.
        """
        try:
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS known_faces (
                    name TEXT NOT NULL,
                    embedding BLOB NOT NULL
                )
            ''')
            self.conn.commit()
            self.logger.info("Table 'known_faces' created or already exists.")
        except sqlite3.Error as e:
            self.logger.error(f"Error creating table: {e}")
            raise

    def add_face(self, name, embedding):
        """
        Adds a new face embedding to the database.

        Args:
            name (str): The name of the person.
            embedding (np.ndarray): The face embedding as a NumPy array.
        """
        try:
            # Serialize the NumPy array to a binary format
            out = io.BytesIO()
            np.save(out, embedding)
            out.seek(0)
            serialized_embedding = out.read()

            self.cursor.execute("INSERT INTO known_faces (name, embedding) VALUES (?, ?)", (name, serialized_embedding))
            self.conn.commit()
            self.logger.info(f"Added face for '{name}' to the database.")
        except sqlite3.Error as e:
            self.logger.error(f"Error adding face for '{name}': {e}")
            raise

    def get_all_known_faces(self):
        """
        Retrieves all face embeddings from the database.

        Returns:
            list: A list of tuples, where each tuple contains (name, embedding).
        """
        try:
            self.cursor.execute("SELECT name, embedding FROM known_faces")
            rows = self.cursor.fetchall()
            
            faces = []
            for name, serialized_embedding in rows:
                # Deserialize the binary data back to a NumPy array
                inp = io.BytesIO(serialized_embedding)
                inp.seek(0)
                embedding = np.load(inp)
                faces.append((name, embedding))
            
            self.logger.info(f"Retrieved {len(faces)} faces from the database.")
            return faces
        except sqlite3.Error as e:
            self.logger.error(f"Error retrieving faces: {e}")
            raise
        except Exception as e:
            self.logger.error(f"An unexpected error occurred while deserializing embeddings: {e}")
            raise

    def close_connection(self):
        """
        Closes the database connection.
        """
        if self.conn:
            self.conn.close()
            self.logger.info("Database connection closed.")