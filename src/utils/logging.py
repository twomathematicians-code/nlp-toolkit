import logging, sys
def setup(name: str = "nlp-toolkit") -> logging.Logger:
    log = logging.getLogger(name)
    log.addHandler(logging.StreamHandler(sys.stdout))
    log.setLevel(logging.INFO)
    return log
