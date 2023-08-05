from _decimal import Decimal
import cv2 as cv

from ErrorDiffusionProcessor import ErrorDiffusionProcessor

if __name__ == '__main__':
    image_path = '8-bit-256-x-256-Grayscale-Lena-Image.png'
    processor = ErrorDiffusionProcessor()
    i = Decimal('0')
    while i <= 1:
        image_diffusion = processor.run_image_diffusion(image_path, float(i))
        new_image_path = f'images/_image_diffusion_Q2_{str(i * 100)}%.png'
        cv.imwrite(new_image_path, image_diffusion)
        print(f'New image created using the percentage {str(i * 100)} at {new_image_path}')
        i += Decimal('0.1')
