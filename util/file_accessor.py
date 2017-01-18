"""
File accessor
"""

import os
import re

from sklearn.cross_validation import ShuffleSplit

def get_filelist(directory):
    return [file for file in _find_all_files(directory)]

def get_teacher_data(directory, label):
    filelist = get_filelist(directory)

    teacher_data = []
    for file in filelist:
        teacher_data.append((file, 1 if re.search(label, file) else 0))

    return teacher_data

def _find_all_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            yield os.path.join(root, file)
