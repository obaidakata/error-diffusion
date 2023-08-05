import os
import numpy as np
from enum import IntEnum


def image_compare(image1, image2) -> bool:
    rows1, columns1, _ = np.shape(image1)
    rows2, columns2, _ = np.shape(image2)
    if rows1 != rows2 or columns1 != columns2:
        print('size diff')
        return False

    for i in range(rows1 - 1):
        for j in range(columns1 - 1):
            if (image1[i][j] & image2[i][j]).all():
                print(i, j)
                return False
    return True

def validate_input(input):
    if len(input) != 2:
        raise Exception('Please pass a path to an image and an integer m > 0')
    image_path = input[0]
    is_file = os.path.isfile(image_path)
    if not is_file:
        raise Exception(f'Image does not exist, path provided {image_path}')
    number_of_levels = input[1]
    if number_of_levels.isdigit() and int(number_of_levels) > 0:
        number_of_levels = int(number_of_levels)
    else:
        raise Exception(f'Please provide a number greater than 0 as the second parameter, parameter provided {number_of_levels}')

    return image_path, number_of_levels

class Color(IntEnum):
    White = 0
    Black = 255
