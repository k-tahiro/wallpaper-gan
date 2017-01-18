"""
File list generator
"""

import os
import re

from sklearn.cross_validation import ShuffleSplit

SUFFIX = r'_1366x768.jpg'

def get_file_list(directory):
    return [file for file in _find_all_files(directory)]

def get_teacher_data(directory, label):
    file_list = [file for file in _find_all_files(directory) if re.search(SUFFIX, file)]

    teacher_data = []
    for file in file_list:
        teacher_data.append((file, 1 if re.search(label, file) else 0))

    train_data = []
    test_data = []

    ss = ShuffleSplit(len(teacher_data), n_iter=1, test_size=0.25, random_state = 222)
    for train_index, test_index in ss:
        train_data = [teacher_data[index] for index in train_index]
        test_data = [teacher_data[index] for index in test_index]

    return train_data, test_data

def _find_all_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            yield os.path.join(root, file)
