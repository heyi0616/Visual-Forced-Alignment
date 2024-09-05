import logging


def get_one_logger(save_path):
    training_mode = save_path.split("/")[-1]
    log_path = '{}/log.txt'.format(save_path)
    logger = logging.getLogger("mylog")
    logger.setLevel(logging.INFO)
    LOG_FORMAT = "%(asctime)s[%(levelname)s] %(message)s"
    DATE_FORMAT = "%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)
    # 输出到文件
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # 输出到console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger