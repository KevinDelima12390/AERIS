import sys
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QStatusBar,
)
from PyQt5.QtCore import QThread, QObject, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from utils.logger import get_logger

logger = get_logger(__name__)
from arc_engine.arc_engine_core import ARCEngineCore


class EngineWorker(QObject):
    """
    Worker thread for running the ARCEngineCore.
    """
    frame_ready = pyqtSignal(object)
    status_updated = pyqtSignal(str)
    person_identified = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.engine = ARCEngineCore(video_source=0)
        self.running = True

    @pyqtSlot()
    def run(self):
        """
        Starts the engine's processing loop.
        """
        logger.info("EngineWorker started.")
        while self.running:
            status, frame, message, name = self.engine.process_frame()
            if not status:
                self.status_updated.emit("Failed to read frame from video stream.")
                break
            
            if frame is not None:
                self.frame_ready.emit(frame)
            
            if message:
                self.status_updated.emit(message)
            
            if name:
                self.person_identified.emit(name)
        
        self.engine.shutdown()
        logger.info("EngineWorker finished.")

    def stop(self):
        """
        Stops the processing loop.
        """
        self.running = False


class DispatcherDashboard(QMainWindow):
    """
    Main application window for the Dispatcher Dashboard.
    """

    def __init__(self):
        """
        Initializes the DispatcherDashboard window.
        """
        super().__init__()

        self.setWindowTitle("ARC System - Dispatcher Dashboard")
        self.setGeometry(100, 100, 1280, 720)  # x, y, width, height

        logger.info("Dispatcher Dashboard GUI initialized.")

        self._init_ui()
        self._init_engine_thread()

    def _init_ui(self):
        """
        Sets up the user interface widgets and layouts.
        """
        # Create a central widget and a main layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Video Display Label
        self.video_display = QLabel("Video feed will be displayed here.")
        self.video_display.setStyleSheet("background-color: black; color: white;")
        self.video_display.setFixedSize(1280, 720)  # Example size
        main_layout.addWidget(self.video_display)

        # Status Bar
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("ARC Engine status will be shown here.")

        # Quit Button
        quit_button = QPushButton("Quit", self)
        quit_button.clicked.connect(self.close)
        main_layout.addWidget(quit_button)

    def _init_engine_thread(self):
        """
        Initializes and starts the background thread for the ARC engine.
        """
        self.thread = QThread()
        self.worker = EngineWorker()
        self.worker.moveToThread(self.thread)

        # Connect signals and slots
        self.thread.started.connect(self.worker.run)
        self.worker.frame_ready.connect(self.update_frame)
        self.worker.status_updated.connect(self.update_status)
        self.worker.person_identified.connect(self.handle_person_identified)

        self.thread.start()
        logger.info("ARCEngine thread started.")

    @pyqtSlot(object)
    def update_frame(self, frame):
        """
        Updates the video display with a new frame.
        """
        # Convert OpenCV frame (BGR) to QImage (RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.video_display.setPixmap(QPixmap.fromImage(qt_image))

    @pyqtSlot(str)
    def update_status(self, message):
        """
        Updates the status bar with a new message.
        """
        self.statusBar().showMessage(message)

    @pyqtSlot(str)
    def handle_person_identified(self, name):
        """
        Handles the event when a person is identified.
        """
        # For now, just log it. This can be expanded to show alerts, etc.
        logger.info(f"Person Identified: {name}")
        self.update_status(f"Person Identified: {name}")

    def closeEvent(self, event):
        """
        Handles the main window's close event to ensure graceful shutdown.
        """
        logger.info("Closing application...")
        self.worker.stop()
        self.thread.quit()
        self.thread.wait()
        logger.info("Thread finished.")
        event.accept()


if __name__ == "__main__":
    # This part is for testing the component independently
    app = QApplication(sys.argv)
    dashboard = DispatcherDashboard()
    dashboard.show()
    sys.exit(app.exec_())