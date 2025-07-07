# utils/logging.py

def print_and_log(message, file=None):
    """
    Prints to console and writes to log file if provided.

    Args:
        message (str): Text to log
        file (file handle or None): Opened file object or None
    """
    print(message)
    if file is not None:
        file.write(message + "\n")
        file.flush()
