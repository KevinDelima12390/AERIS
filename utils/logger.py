import logging
import os

def get_logger(name="arc_logger"):
    """
    Configures and returns a logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create handlers if they don't exist
    if not logger.handlers:
        # Console Handler
        c_handler = logging.StreamHandler()
        
        # File Handler
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        f_handler = logging.FileHandler(os.path.join(log_dir, "arc_system.log"))

        # Create formatters and add it to handlers
        log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(log_format)
        f_handler.setFormatter(log_format)

        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)

    return logger

logger = get_logger()

# Example usage:
if __name__ == '__main__':
    logger.info("This is an info message from the centralized logger.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")