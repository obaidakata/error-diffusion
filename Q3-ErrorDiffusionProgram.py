import sys
import cv2 as cv
import utils
from ErrorDiffusionProcessor import ErrorDiffusionProcessor

if __name__ == '__main__':
    args = sys.argv[1:]
    try:
        image_path, number_of_levels = utils.validate_input(args)
        processor = ErrorDiffusionProcessor()
        image_diffusion = processor.run_image_diffusion_with_level(image_path, number_of_levels)
        print('Done running the image diffusion algorithm')
        image_diffusion_file_path = f'images/image_diffusion-m-{number_of_levels}-Q3.png'
        cv.imwrite(image_diffusion_file_path, image_diffusion)

        print(f'The new file saved at {image_diffusion_file_path}')
        cv.imshow(f'image_diffusion-m-{number_of_levels}-Q3', image_diffusion)
        cv.waitKey(0)
        cv.destroyAllWindows()
    except Exception as e:
        print(e)
