import sys
import os

class Logger(object):
    """A logger that writes to console and file, handling encoding issues gracefully."""
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        # Always open the log file with UTF-8 for full character support.
        if fpath is not None:
            self.file = open(fpath, 'w', encoding='utf-8')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        """ Overloads default write function to log to both console and file """
        # 1. Write the original, unmodified message to the log file (which is UTF-8).
        if self.file is not None:
            self.file.write(msg)

        # 2. For the console, create a safe version of the message.
        # Encode the string to the console's encoding, replacing any characters
        # that don't exist in that encoding with a placeholder (e.g., '?').
        # This prevents UnicodeEncodeError on legacy terminals like Windows' cp950.
        console_encoding = self.console.encoding or 'utf-8'
        safe_msg = msg.encode(console_encoding, errors='replace').decode(console_encoding)
        self.console.write(safe_msg)
        
        self.flush()

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            # Check if fileno is available before calling fsync, as it might not be in some environments
            try:
                os.fsync(self.file.fileno())
            except (IOError, OSError):
                pass # Ignore fsync errors (e.g., on pipes)

    def close(self):
        if self.file is not None:
            # Check if the file is already closed before trying to close it.
            if not self.file.closed:
                self.file.close()
