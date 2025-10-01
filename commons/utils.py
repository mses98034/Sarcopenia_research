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

# Score measure

# def numeric_score(y_pred, y_true):
#     """Compute True Positive, True Negative, False Positive, False Negative classifications
#     between a prediction and its ground truth
#     :param y_pred: prediction
#     :param y_true: ground truth
#     :return: True Positive, True Negative, False Positive, False Negative
#     """
#     y_pred = y_pred.astype(int)
#     y_true = y_true.astype(int)
#     FP = float(np.sum((y_pred == 1) & (y_true == 0)))
#     FN = float(np.sum((y_pred == 0) & (y_true == 1)))
#     TP = float(np.sum((y_pred == 1) & (y_true == 1)))
#     TN = float(np.sum((y_pred == 0) & (y_true == 0)))
#     return FP, FN, TP, TN

# def specificity_score(y_pred, y_true):
#     """Compute specificity (= TN / (TN+FP)) between a prediction and its ground truth
#     :param y_pred: prediction
#     :param y_true: ground truth
#     :return: specificity score value
#     """
#     FP, FN, TP, TN = numeric_score(y_pred, y_true)
#     if (FP + TN) <= 0:
#         return 0.
#     else:
#         return np.divide(TN, FP + TN)

# def sensitivity_score(y_pred, y_true):
#     """Compute sensitivity (= TP / (TP+FN)) between a prediction and its ground truth
#     :param y_pred: prediction
#     :param y_true: ground truth
#     :return: Sensitivity score value
#     """
#     FP, FN, TP, TN = numeric_score(y_pred, y_true)
#     if (TP + FN) <= 0:
#         return 0.
#     else:
#         return np.divide(TP, TP + FN)