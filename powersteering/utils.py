import sys
from loguru import logger

def setup_logging(verbose_count: int) -> None:
    """Configure logging level based on verbosity count

    Args:
        verbose_count: Number of times --verbose flag was specified
            0 = INFO only
            1 = Add WARNING
            2 = Add DEBUG
            3 = Add everything
    """
    logger.remove()  # Remove default handler

    # Map verbosity count to log level using all loguru severity levels
    levels = {
        0: "CRITICAL",  # 50 (show only critical)
        1: "ERROR",     # 40 (show error and above)
        2: "WARNING",   # 30 (show warning and above)
        3: "SUCCESS",   # 25 (show success and above)
        4: "INFO",      # 20 (show info and above)
        5: "DEBUG",     # 10 (show debug and above)
        6: "TRACE"      # 5  (show everything)
    }
    level = levels.get(min(verbose_count, max(levels.keys())), "CRITICAL")
    logger.add(
        sys.stdout,
        colorize=True,
        level=level
    )
    logger.success(f"Set log level to {level} based on verbosity count {verbose_count}")
