import sys
from PyQt5.QtWidgets import QApplication
from gui.dispatcher_dashboard import DispatcherDashboard
from utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


def main():
    """
    Main function to launch the ARC system's Dispatcher Dashboard GUI.
    """
    logger.info("ARC system starting up...")

    app = QApplication(sys.argv)
    dashboard = DispatcherDashboard()
    dashboard.show()

    logger.info("Dispatcher Dashboard GUI launched.")
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()