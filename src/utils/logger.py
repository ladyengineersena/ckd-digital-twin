import logging, os, sys
from logging.handlers import RotatingFileHandler

def get_logger(name=__name__, log_file="artifacts/logs/app.log"):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = RotatingFileHandler(log_file, maxBytes=10_000_00, backupCount=3)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger

