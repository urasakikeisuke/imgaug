"""affine.py"""

from typing import Optional, Tuple, Union
import numpy
import cv2

from .randoms import RandomGenerator
from .misc import check_input, get_input_shape, transpose_array
from .constants import *

def _get_affine_matrix() -> numpy.ndarray:
    return numpy.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0]], numpy.float32)


class AffineBaseClass(object):
    def __init__(
        self,
        shift: Optional[Union[int, float, RandomGenerator]] = None, 
        border_mode: int = None, 
    ) -> None:
        self.border_mode = border_mode

    def _get_transform(self) -> numpy.ndarray:
        raise NotImplementedError

    def exec(self, *input: numpy.ndarray) -> numpy.ndarray:
        self._get_transform()

        if self.transform is None:
            raise NotImplementedError()

        ret = []
        for i, input_ in enumerate(input):          
            input_shape = get_input_shape(input_)

            if input_shape == CHW_C1 or input_shape == CHW_C3:
                input_ = transpose_array(input_, HWC_C3)

            warped: numpy.ndarray = cv2.warpAffine(input_, self.transform, (input_.shape[1], input_.shape[0]), flags=cv2.INTER_NEAREST, borderMode=self.border_mode)

            if input_shape == CHW_C1 or input_shape == CHW_C3:
                warped = transpose_array(warped, input_shape)
            
            ret.append(warped)

        return ret


class Identity(AffineBaseClass):
    def __init__(
        self,
        shift: Optional[Union[int, float, RandomGenerator]] = None,
        border_mode: int = cv2.BORDER_WRAP,
    ) -> None:
        super().__init__(shift, border_mode)
        
        self.shift: Union[int, float, RandomGenerator]
        if shift is not None:
            if isinstance(shift, (int, float)):
                self.shift = shift
            elif isinstance(shift, RandomGenerator):
                self.shift = shift
            else:
                raise TypeError(f'Expected type is `int` or `float` but got {type(shift).__name__}')

    def _get_transform(self) -> numpy.ndarray:
        sam: numpy.ndarray = _get_affine_matrix()
        dam: numpy.ndarray = sam.copy()
        self.transform: numpy.ndarray = cv2.getAffineTransform(sam, dam)

    def exec(self, *input: numpy.ndarray) -> numpy.ndarray:
        return super().exec(*input)


class ShiftHorizontally(Identity):
    def __init__(
        self,
        shift: Union[int, float, RandomGenerator],
        border_mode: int = cv2.BORDER_WRAP,
    ) -> None:
        super().__init__(shift, border_mode)

    def _get_transform(self) -> numpy.ndarray:
        sam: numpy.ndarray = _get_affine_matrix()
        dam: numpy.ndarray = sam.copy()

        shift = self.shift() if isinstance(self.shift, RandomGenerator) else self.shift
        dam[:, 0] += shift

        self.transform: numpy.ndarray = cv2.getAffineTransform(sam, dam)
    
    def exec(self, *input: numpy.ndarray) -> numpy.ndarray:
        return super().exec(*input)


class ShiftVertically(Identity):
    def __init__(
        self,
        shift: Union[int, float, RandomGenerator],
        border_mode: int = cv2.BORDER_REFLECT_101,
    ) -> None:
        super().__init__(shift, border_mode)

    def _get_transform(self) -> numpy.ndarray:
        sam: numpy.ndarray = _get_affine_matrix()
        dam: numpy.ndarray = sam.copy()

        shift = self.shift() if isinstance(self.shift, RandomGenerator) else self.shift
        dam[:, 0] += shift

        self.transform: numpy.ndarray = cv2.getAffineTransform(sam, dam)
    
    def exec(self, *input: numpy.ndarray) -> numpy.ndarray:
        return super().exec(*input)


class ShearHorizontally(Identity):
    def __init__(
        self,
        shift: Union[int, float, RandomGenerator],
        img_size: Tuple[int, int],
        border_mode: int = cv2.BORDER_WRAP,
    ) -> None:
        super().__init__(shift, border_mode)

        self.img_size = img_size

    def _get_transform(self) -> numpy.ndarray:
        sam: numpy.ndarray = _get_affine_matrix()
        dam: numpy.ndarray = sam.copy()
        shift = self.shift() if isinstance(self.shift, RandomGenerator) else self.shift
        dam[:, 0] += (shift / self.img_size[0] * (self.img_size[0] - sam[:, 1])).astype(numpy.float32)
    
        self.transform: numpy.ndarray = cv2.getAffineTransform(sam, dam)
    
    def exec(self, *input: numpy.ndarray) -> numpy.ndarray:
        return super().exec(*input)


class ShearVertically(Identity):
    def __init__(
        self,
        shift: Union[int, float, RandomGenerator],
        img_size: Tuple[int, int],
        border_mode: int = cv2.BORDER_REFLECT_101,
    ) -> None:
        super().__init__(shift, border_mode)

        self.img_size = img_size

    def _get_transform(self) -> numpy.ndarray:
        sam: numpy.ndarray = _get_affine_matrix()
        dam: numpy.ndarray = sam.copy()
        shift = self.shift() if isinstance(self.shift, RandomGenerator) else self.shift
        dam[:, 1] += (shift / self.img_size[1] * (self.img_size[1] - sam[:, 0])).astype(numpy.float32)
    
        self.transform: numpy.ndarray = cv2.getAffineTransform(sam, dam)
    
    def exec(self, *input: numpy.ndarray) -> numpy.ndarray:
        return super().exec(*input)


class FlipHorizontally(Identity):
    def __init__(
        self,
        img_size: Tuple[int, int],
    ) -> None:
        super().__init__()

        self.img_size = img_size

    def _get_transform(self) -> numpy.ndarray:
        sam: numpy.ndarray = _get_affine_matrix()
        dam: numpy.ndarray = sam.copy()
        dam[:,0] = self.img_size[1] - sam[:,0]
    
        self.transform: numpy.ndarray = cv2.getAffineTransform(sam, dam)
    
    def exec(self, *input: numpy.ndarray) -> numpy.ndarray:
        return super().exec(*input)
