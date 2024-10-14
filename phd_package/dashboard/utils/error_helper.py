import functools
from typing import Callable, Any
import plotly.graph_objects as go
import logging


def handle_errors(default_return: Any = None):
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logging.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
                if isinstance(default_return, Callable):
                    return default_return(*args, **kwargs)
                return default_return
        return wrapper
    return decorator


def handle_data_errors(default_return: Any = None, retry_attempts: int = 1):
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(retry_attempts + 1):
                try:
                    result = func(*args, **kwargs)
                    if result is None or (isinstance(result, (list, dict)) and len(result) == 0):
                        raise ValueError(f"Empty result from {func.__name__}")
                    return result
                except Exception as e:
                    logging.error(
                        "Attempt %d/%d failed in %s: %s",
                        attempt + 1,
                        retry_attempts + 1,
                        func.__name__,
                        str(e),
                        exc_info=True
                    )
                if attempt == retry_attempts:
                    logging.error(
                        "All attempts failed in %s. Returning default value.",
                        func.__name__,
                        exc_info=True
                    )
                    return default_return() if callable(default_return) else default_return
        return wrapper
    return decorator


# Set up logging
logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s"
)
