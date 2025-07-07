# data/transforms.py

import cv2
import numpy as np
from torchvision import transforms
from PIL import Image


class CLAHETransform:
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to an image.
    """

    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img):
        """
        Args:
            img (PIL.Image or np.ndarray): Input RGB image.
        Returns:
            np.ndarray: CLAHE-enhanced RGB image.
        """
        if isinstance(img, Image.Image):
            img = np.array(img)

        # Convert to LAB and apply CLAHE to the L channel
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = self.clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        rgb_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
        return Image.fromarray(rgb_clahe)


class MorphologicalDilation:
    """
    Applies morphological dilation to enhance cellular boundaries.
    """

    def __init__(self, kernel_size=3, iterations=1):
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.iterations = iterations

    def __call__(self, img):
        """
        Args:
            img (PIL.Image or np.ndarray): Input RGB image.
        Returns:
            PIL.Image: Dilated image.
        """
        if isinstance(img, Image.Image):
            img = np.array(img)

        # Convert to grayscale for dilation
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        dilated = cv2.dilate(gray, self.kernel, iterations=self.iterations)

        # Stack grayscale back into RGB (dummy channels)
        dilated_rgb = cv2.merge([dilated] * 3)
        return Image.fromarray(dilated_rgb)


def get_preprocessing_pipeline(resize=(32, 32), apply_clahe=True, apply_dilation=True):
    """
    Builds the preprocessing pipeline for training and inference.

    Returns:
        torchvision.transforms.Compose: Transform pipeline
    """
    transform_list = []

    if apply_clahe:
        transform_list.append(CLAHETransform())

    if apply_dilation:
        transform_list.append(MorphologicalDilation())

    transform_list += [
        transforms.Resize(resize),
        transforms.ToTensor(),  # Converts to [0, 1]
    ]

    return transforms.Compose(transform_list)
