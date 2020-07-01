# Written by Yixiao Ge

from PIL import Image
import numpy as np
import random
import copy

__all__ = ['MutualTransform']

class MutualTransform:
    """Apply the transformer more times on a same raw image."""

    def __init__(
            self,
            transformer,
            times = 2
        ):
        self.transformer = transformer
        self.times = times

    def __call__(self, img):
        imgs = []
        for _ in range(self.times):
            img_copy = copy.deepcopy(img)
            img_copy = self.transformer(img_copy)
            imgs.append(img_copy)

        return imgs

    def __repr__(self):
        return "Mutual Transformer"
