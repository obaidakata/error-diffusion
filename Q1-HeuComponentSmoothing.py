import math
import numpy as np
import cv2 as cv

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)


def normalized_pixel(pixel):
    return pixel[0] / 255, pixel[1] / 255, pixel[2] / 255

class Converter:
    round_factor = 4

    @staticmethod
    def _calc_hue(pixel):
        red, green, blue = pixel
        numerator = ((red - green) + (red - blue)) / 2
        denominator = math.sqrt(pow(red - green, 2) + (red - blue) * (green - blue))
        # if denominator == 0:
        #     raise Exception('Error')
        theta = np.rad2deg(np.arccos(numerator / denominator))
        return round(theta if blue <= green else 360 - theta, Converter.round_factor)

    @staticmethod
    def _calc_saturation(pixel):
        red, green, blue = pixel
        minimum = min(min(red, green), blue)
        return round(1 - (3 / (red + green + blue) * minimum), Converter.round_factor)

    @staticmethod
    def _calc_intensity(pixel):
        red, green, blue = pixel
        return round((red + green + blue) / 3, Converter.round_factor)

    @staticmethod
    def RGB_to_HSI(pixel):
        pixel = normalized_pixel(pixel)
        return Converter._calc_hue(pixel), Converter._calc_saturation(pixel), Converter._calc_intensity(pixel)

    @staticmethod
    def HSI_to_RGB(pixel):
        h, s, i = pixel
        r, g, b = 0, 0, 0
        my_cos = lambda x: math.cos(math.radians(x))
        if 0 <= h < 120:
            b = i * (1 - s)
            r = i * (1 + (s * my_cos(h)) / my_cos(60 - h))
            g = 3 * i - (r + b)
        elif 120 <= h < 240:
            h -= 120
            r = i * (1 - s)
            g = i * (1 + (s * my_cos(h)) / my_cos(60 - h))
            b = 3 * i - (r + g)
        else:
            h -= 240
            g = i * (1 - s)
            b = i * (1 + (s * my_cos(h)) / my_cos(60 - h))
            r = 3 * i - (g + b)

        rgb = (r, g, b)
        return tuple([math.floor(255 * x) for x in rgb])


def image_to_rgb(image, width):
    image_copy = image.copy()
    for i in range(width):
        for j in range(width):
            image_copy[i, j] = Converter.HSI_to_RGB(image[i, j])
    return image_copy


def main():
    width = 500
    img = np.zeros((width, width, 3))
    half_width = int(width / 2)
    img[0:half_width, 0:half_width] = GREEN
    img[half_width:width, half_width:width] = GREEN
    img[half_width:width, 0:half_width] = BLUE
    img[0:half_width, half_width:width] = RED

    cv.imshow('Before 500 x 500', img)
    values_HSI = set()
    for i in range(width):
        for j in range(width):
            t = Converter.RGB_to_HSI(img[i, j])
            img[i, j] = t
            values_HSI.add(t)

    kernel_size = int(width / 4)
    centerIndex = int(kernel_size / 2)
    number_of_sub_matrices = int(width - (kernel_size - 1))
    for i in range(number_of_sub_matrices):
        for j in range(number_of_sub_matrices):
            sub_img = img[i:i + kernel_size, j:j + kernel_size, :]
            average_saturation = np.average(sub_img[:, :, 0])
            sub_img[centerIndex, centerIndex, 0] = average_saturation

    cv.imshow('After 500 x 500', image_to_rgb(img, width))
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
