import os
import logging
import sys

def configure_logging(log_level=logging.INFO):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    logging.basicConfig(
        level=log_level,
         handlers=[
            logging.StreamHandler(sys.stdout)],
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Suppress noisy loggers
    for logger_name in ['mediapipe', 'matplotlib']:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    