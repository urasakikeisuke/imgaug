"""color.py"""

from typing import Union

import cv2
import numpy

from .randoms import RandomGenerator
from .misc import get_input_shape, check_input, transpose_array
from .constants import *

def _blend(img1: numpy.ndarray, img2: numpy.ndarray, ratio: float) -> numpy.ndarray:
    ratio = float(ratio)
    bound = 1.0 if numpy.nanmax(img1) < 1.0 else 255.0
    dst: numpy.ndarray = ratio * img1 + (1.0 - ratio) * img2

    return dst.clip(0, bound).astype(img1.dtype)


def _rgb_to_grayscale(img: numpy.ndarray) -> numpy.ndarray:
    if check_input(img, desire=("HWC", 3)):
        img = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(img.dtype)
        return img[..., None]
    elif check_input(img, desire=("CHW", 3)):
        img = (0.299 * img[0, :, :] + 0.587 * img[1, :, :] + 0.114 * img[2, :, :]).astype(img.dtype)
        return img[None, ...]
    elif check_input(img, desire=("CHW", 1)) or check_input(img, desire=("HWC", 1)) or check_input(img, desire=("HW", 1)):
        return img
    else:
        raise ValueError(f"Expected input shape is HWC or CHW or HW and its number of channels is 1 or 3, but got {img.shape}")


class Brightness():
    def __init__(
        self,
        factor: Union[int, float, RandomGenerator],
        only_first_image: bool = True,
    ) -> None:
        self.factor = factor
        self.only_first_image = only_first_image

    def exec(self, *input: numpy.ndarray) -> numpy.ndarray:        
        factor = self.factor() if isinstance(self.factor, RandomGenerator) else self.factor

        ret = []
        for i, input_ in enumerate(input):
            if i == 0:
                ret.append(_blend(input_, numpy.zeros_like(input_), factor))
            else:
                if self.only_first_image:
                    ret.append(input_)
                else:
                    ret.append(_blend(input_, numpy.zeros_like(input_), factor))

        return ret


class Saturation():
    def __init__(
        self,
        factor: Union[int, float, RandomGenerator],
        only_first_image: bool = True,
    ) -> None:
        self.factor = factor
        self.only_first_image = only_first_image

    def exec(self, *input: numpy.ndarray) -> numpy.ndarray:
        factor = self.factor() if isinstance(self.factor, RandomGenerator) else self.factor

        ret = []
        for i, input_ in enumerate(input):
            if i == 0:
                ret.append(_blend(input_, _rgb_to_grayscale(input_), factor))
            else:
                if self.only_first_image:
                    ret.append(input_)
                else:
                    ret.append(_blend(input_, _rgb_to_grayscale(input_), factor))

        return ret


class Contrast():
    def __init__(
        self,
        factor: Union[int, float, RandomGenerator],
        only_first_image: bool = True,
    ) -> None:
        self.factor = factor
        self.only_first_image = only_first_image

    def exec(self, *input: numpy.ndarray) -> numpy.ndarray:
        factor = self.factor() if isinstance(self.factor, RandomGenerator) else self.factor

        ret = []
        for i, input_ in enumerate(input):
            mean = numpy.mean(_rgb_to_grayscale(input_).astype(input_.dtype), axis=(-2, -1), keepdims=True)
    
            if i == 0:
                ret.append(_blend(input_, mean, factor))
            else:
                if self.only_first_image:
                    ret.append(input_)
                else:
                    ret.append(_blend(input_, mean, factor))

        return ret


class AutoContrast():
    def __init__(
        self,
        clip_hist_percent: Union[float, RandomGenerator] = 1.0,
        only_first_image: bool = True,
    ) -> None:
        self.clip_hist_percent = clip_hist_percent
        self.only_first_image = only_first_image

    def _adjust(self, img: numpy.ndarray) -> numpy.ndarray:
        input_shape = get_input_shape(img)

        if input_shape == HW_C1 or input_shape == HWC_C1 or input_shape == CHW_C1:
            return img
        img = transpose_array(img, HWC_C3)
        gray: numpy.ndarray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        hist: numpy.ndarray = cv2.calcHist([gray], [0] , None, [256], [0,256])
        hist_size = len(hist)

        accumulator = [float(hist[0])]
        for index in range(1, hist_size):
            accumulator.append(accumulator[index -1] + float(hist[index]))

        maximum = accumulator[-1]
        clip_hist_percent = self.clip_hist_percent() if isinstance(self.clip_hist_percent, RandomGenerator) else self.clip_hist_percent
        clip_hist_percent *= (maximum / 100.0)
        clip_hist_percent /= 2.0

        minimum_gray = 0
        while accumulator[minimum_gray] < clip_hist_percent:
            minimum_gray += 1

        maximum_gray = hist_size -1
        while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
            maximum_gray -= 1
        
        alpha = 255 / (maximum_gray - minimum_gray + 1e-10)
        beta = -minimum_gray * alpha

        auto_result = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        auto_result = transpose_array(auto_result, input_shape)

        return auto_result


    def exec(self, *input: numpy.ndarray) -> numpy.ndarray:
        ret = []
        for i, input_ in enumerate(input):
            if i == 0:
                ret.append(self._adjust(input_))
            else:
                if self.only_first_image:
                    ret.append(input_)
                else:
                    ret.append(self._adjust(input_))

        return ret
