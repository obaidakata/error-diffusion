import cv2 as cv
import numpy as np
from utils import Color


class ErrorDiffusionProcessor:

    @staticmethod
    def _get_levels(number_of_levels):
        ranges = []
        start = 0
        portion_size = int(256 / number_of_levels)
        while len(ranges) < number_of_levels:
            ranges.append((start, start + portion_size - 1))
            start += portion_size
        ranges[number_of_levels - 1] = (ranges[number_of_levels - 1][0], 255)
        return ranges

    def run_image_diffusion(self, gray_scale_image_path: str, percent):
        gray_scale_image = self._read_image_and_convert_to_grey_scale(gray_scale_image_path)
        cv.imshow(gray_scale_image_path, gray_scale_image)
        rows, columns = np.shape(gray_scale_image)
        image_diffusion = gray_scale_image.copy()
        error = np.zeros((rows, columns))
        for i in range(rows - 1):
            for j in range(columns - 1):
                image_diffusion[i, j] = image_diffusion[i, j] + error[i, j] * percent
                tmp = Color.White if image_diffusion[i, j] < 128 else Color.Black
                diff = image_diffusion[i, j] - tmp
                image_diffusion[i, j] = tmp
                error[i, j + 1] += diff * 3 / 8
                error[i + 1, j] += diff * 3 / 8
                error[i + 1, j + 1] += diff * 1 / 4
        return image_diffusion

    def run_image_diffusion_with_level(self, gray_scale_image_path: str, number_of_levels: int):
        levels = self._get_levels(number_of_levels)
        gray_scale_image = self._read_image_and_convert_to_grey_scale(gray_scale_image_path)
        cv.imshow(gray_scale_image_path, gray_scale_image)
        rows, columns = np.shape(gray_scale_image)
        image_diffusion = gray_scale_image.copy()
        error = np.zeros((rows, columns))
        for i in range(rows - 1):
            for j in range(columns - 1):
                for low, high in levels:
                    if low <= image_diffusion[i, j] <= high:
                        image_diffusion[i, j] = image_diffusion[i, j] + error[i, j]
                        threshold = int(high / pow(2, number_of_levels) + 1)
                        tmp = low if image_diffusion[i, j] < threshold else high
                        diff = image_diffusion[i, j] - tmp
                        image_diffusion[i, j] = tmp
                        error[i, j + 1] += diff * 3 / 8
                        error[i + 1, j] += diff * 3 / 8
                        error[i + 1, j + 1] += diff * 1 / 4
        return image_diffusion

    def simple_binary(self, image_path):
        simple_binary_image = self._read_image_and_convert_to_grey_scale(image_path)
        rows, columns = np.shape(simple_binary_image)
        for i in range(rows):
            for j in range(columns):
                simple_binary_image[i, j] = Color.White if simple_binary_image[i, j] < 255 / 2 else Color.Black
        return simple_binary_image

    @staticmethod
    def _read_image_and_convert_to_grey_scale(image_path):
        gray_scale_image = cv.imread(image_path)
        gray_scale_image = cv.cvtColor(gray_scale_image, cv.COLOR_RGB2GRAY)
        return gray_scale_image
