import os
import time
import logging

logger = logging.getLogger(__name__)

def cleanup_old_files(directories: list[str], max_age_seconds: int = 3600):
    """
    Deletes files in the specified directories that are older than max_age_seconds.
    
    Args:
        directories: List of directory paths to clean up
        max_age_seconds: Maximum age of files in seconds (default: 1 hour)
    """
    now = time.time()
    for directory in directories:
        if not os.path.exists(directory):
            continue
            
        try:
            for f in os.listdir(directory):
                filepath = os.path.join(directory, f)
                # Skip directories
                if os.path.isdir(filepath):
                    continue
                    
                if os.path.getmtime(filepath) < now - max_age_seconds:
                    try:
                        os.remove(filepath)
                        logger.info(f"Cleaned up old file: {filepath}")
                    except Exception as e:
                        logger.warning(f"Failed to remove file {filepath}: {e}")
        except Exception as e:
            logger.error(f"Error cleaning up directory {directory}: {e}")
