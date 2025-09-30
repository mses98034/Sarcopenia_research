import sys
import os
import re
sys.path.extend(["../../", "../", "./"])
import numpy as np
from os import fsync
from commons.constant import *

if not os.path.isdir(TMP_DIR):
    os.makedirs(TMP_DIR)

class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, 'w', encoding='utf-8')
        # Check if the console supports UTF-8. If not, enable emoji stripping.
        self.strip_emoji = sys.stdout.encoding.lower() not in ('utf-8', 'utf8')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        processed_msg = msg
        if self.strip_emoji:
            # Regex to remove a wide range of emojis and symbols for compatibility
            emoji_pattern = re.compile(
                "["
                "\U0001F600-\U0001F64F"  # emoticons
                "\U0001F300-\U0001F5FF"  # symbols & pictographs
                "\U0001F680-\U0001F6FF"  # transport & map symbols
                "\U0001F1E0-\U0001F1FF"  # flags (iOS)
                "\u2600-\u26FF"          # miscellaneous symbols
                "\u2700-\u27BF"          # dingbats
                "\uFE0F"                # variation selector
                "]+", flags=re.UNICODE)
            processed_msg = emoji_pattern.sub(r'', msg)

        self.console.write(processed_msg)
        if self.file is not None:
            self.file.write(processed_msg)
        self.flush()

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
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