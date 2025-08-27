import logging
import datetime
import os

class Logger:
    """
    Wrapper for logging
    """
    def __init__(self, log: logging.Logger, rank: int):
        self.log = log
        self.rank = rank

    def info(self, msg: str, all = False):
        if all:
            self.log.info(msg)
        elif self.rank == 0:
            self.log.info(msg)

    def debug(self, msg: str):
        if self.rank == 0:
            self.log.debug(msg)

    def warning(self, msg: str):
        self.log.warning(f"rank {self.rank} - {msg}")

    def error(self, msg: str):
        self.log.error(f"rank {self.rank} - {msg}")

    def critical(self, msg: str):
        self.log.critical(f"rank {self.rank} - {msg}")


def setup_logger(log_dir: str, rank: int, out=True):
    logger = logging.getLogger(f"LOG")
    logger.setLevel(logging.INFO)
    os.makedirs(log_dir, exist_ok=True)
    if out:
        now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        fh = logging.FileHandler(f"{log_dir}/{now}.log")
        fh2 = logging.StreamHandler()
        fhs = [fh, fh2]
    else:
        fh = logging.FileHandler(f"{log_dir}/{now}.log")
        fhs = [fh]
    for _f in fhs:
        _f.setFormatter(logging.Formatter("|%(asctime)s|%(name)s-%(levelname)s| %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(_f)
    logger.propagate = False  # <-- prevents logs bubbling up to root
    logger.root = None
    logger.info("logger initialized")
    return Logger(logger, rank)