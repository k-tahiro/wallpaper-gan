"""
Image resizer
"""

import os
import argparse

import cv2

from util.file_accessor import get_filelist

def resize(input_path, output_path, height, width):
    filelist = get_filelist(input_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for file in filelist:
        img = cv2.imread(file)
        try:
            resized_img = cv2.resize(img, (height, width))
            filename = os.path.basename(file)
            filepath = os.path.join(output_path, filename)
            cv2.imwrite(filepath, resized_img)
        except:
            continue

if __name__ == '__main__':
    # 引数指定
    parser = argparse.ArgumentParser()
    parser.add_argument('--input','--input-path' , '-i', type=str)
    parser.add_argument('--output', '--output-path', '-o', type=str)
    parser.add_argument('--height', type=int)
    parser.add_argument('--width', type=int)
    args = parser.parse_args()

    resize(args.input, args.output, args.height, args.width)
