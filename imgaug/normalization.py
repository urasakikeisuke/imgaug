"""normalization.py"""

from typing import List, Tuple, Union

import numpy

from .misc import check_input, get_input_shape, transpose_array
from .constants import *

class Normalization():
    def __init__(
        self,
        mean: Union[List[float], Tuple[float, ...]],
        std: Union[List[float], Tuple[float, ...]],
        only_first_image: bool = True,
    ) -> None:
        self.mean = mean
        self.std = std
        self.only_first_image = only_first_image

    def _normalize(self, input: numpy.ndarray) -> numpy.ndarray:
        input_shape = get_input_shape(input)

        if input_shape == CHW_C3:
            input = transpose_array(input, HWC_C3)

        for ch in range(input.shape[-1]):
            input[ch] = (input[ch] - self.mean[ch]) / self.std[ch]

        if input_shape == CHW_C3:
            input = transpose_array(input, input_shape)
        
        return input

    def exec(self, *input: numpy.ndarray) -> numpy.ndarray:
        ret = []
        for i, input_ in enumerate(input):  
            if i == 0:
                ret.append(self._normalize(input_))
            else:
                if self.only_first_image:
                    ret.append(input_)
                else:
                    ret.append(self._normalize(input_))

        return ret
