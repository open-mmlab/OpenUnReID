# Credit to https://github.com/facebookresearch/moco/blob/master/moco/loader.py

from PIL import ImageFilter
import random

__all__ = ['GaussianBlur']

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, img):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img

    def __repr__(self):
        return "GaussianBlur Transformer"
