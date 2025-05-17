import logging
import torch
import os

# Define ANSI escape sequences for colors
RESET = "\033[0m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"

# Custom Formatter to add colors based on log level
class ColorFormatter(logging.Formatter):
    # Map log levels to specific colors
    LOG_COLORS = {
        logging.DEBUG: CYAN,     # Cyan for DEBUG
        logging.INFO: GREEN,     # Green for INFO
        logging.WARNING: YELLOW, # Yellow for WARNING
        logging.ERROR: RED,      # Red for ERROR
        logging.CRITICAL: MAGENTA # Magenta for CRITICAL
    }

    def format(self, record):
        # Apply color based on log level
        log_color = self.LOG_COLORS.get(record.levelno, RESET)
        formatted_log = super().format(record)
        return f'{log_color}{formatted_log}{RESET}'


class CustomLogger:
    def __init__(self, name: str, level=logging.DEBUG):
        """
        Initializes a custom logger with a specified name and logging level.

        Args:
        - name (str): Name of the logger (usually the module name).
        - level (int): Logging level (e.g., logging.DEBUG, logging.INFO).
        """
        # Create a logger with the specified name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Create a console handler for logging to the console
        ch = logging.StreamHandler()

        # Set log format and color formatting
        formatter = ColorFormatter(f'%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)

        # Add the handler to the logger
        if not self.logger.hasHandlers():  # Prevent adding handlers multiple times
            self.logger.addHandler(ch)

    def debug(self, message):
        """Logs a debug message."""
        self.logger.debug(message)

    def info(self, message):
        """Logs an info message."""
        self.logger.info(message)

    def warning(self, message):
        """Logs a warning message."""
        self.logger.warning(message)

    def error(self, message):
        """Logs an error message."""
        self.logger.error(message)

    def critical(self, message):
        """Logs a critical message."""
        self.logger.critical(message)
        

