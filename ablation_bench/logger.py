import logging

from rich.logging import RichHandler


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: The logger instance.
    """
    logger = logging.getLogger(name)
    rich_handler = RichHandler()
    rich_handler.setFormatter(logging.Formatter(fmt="%(message)s", datefmt="[%X]"))
    logger.addHandler(rich_handler)
    logger.setLevel(logging.DEBUG)
    return logger
