import logging
from logging.handlers import RotatingFileHandler
import functools
import time
import traceback

LOG_FILE = "logs/mayank_ai_signal_bot.log"
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5

logger = logging.getLogger("MayankAISignalBot")
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    fmt='%(asctime)s | %(levelname)s | %(name)s | %(module)s | %(funcName)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

file_handler = RotatingFileHandler(LOG_FILE, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT, encoding='utf-8')
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


def retry(exceptions, tries=3, delay=1, backoff=2, logger_=logger):
    """
    Decorator for retrying a function with exponential backoff.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _tries, _delay = tries, delay
            while _tries > 1:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    msg = f"{func.__name__} failed with {type(e).__name__}: {e}, retrying in {_delay} seconds..."
                    logger_.warning(msg)
                    time.sleep(_delay)
                    _tries -= 1
                    _delay *= backoff
                except Exception as e:
                    logger_.error(f"Unexpected error in {func.__name__}: {e}\n{traceback.format_exc()}")
                    raise
            # last try
            return func(*args, **kwargs)
        return wrapper
    return decorator


def log_performance(func):
    """
    Decorator to log function entry, exit and execution time.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Entering {func.__name__}")
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.debug(f"Exiting {func.__name__} (Elapsed: {elapsed:.3f}s)")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Exception in {func.__name__} after {elapsed:.3f}s: {e}\n{traceback.format_exc()}")
            raise
    return wrapper


def log_strategy_performance(strategy_name: str, signal: str, confidence: float, result: str):
    logger.info(f"Strategy: {strategy_name} | Signal: {signal} | Confidence: {confidence:.2f} | Result: {result}")


def log_system_health(status: str, details: str = ""):
    logger.info(f"System Health | Status: {status} | Details: {details}")
