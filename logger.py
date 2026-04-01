import logging
from pythonjsonlogger import jsonlogger

def setup_logger():
    logger = logging.getLogger("wine_api")  # ← Changed here
    logger.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # File handler — saves logs to api.log
    file_handler = logging.FileHandler("api.log")
    file_handler.setLevel(logging.DEBUG)

    # JSON formatter for structured logs
    formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

logger = setup_logger()
