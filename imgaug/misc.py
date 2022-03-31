"""misc.py"""

from typing import List, Optional, Tuple
import numpy

from .constants import *


def get_input_shape(input: numpy.ndarray) -> Optional[Tuple[str, int]]:
    dst: Tuple[str, int] = None

    if input.ndim == 3:
        if input.shape[0] == 1:
            dst = CHW_C1
        elif input.shape[2] == 1:
            dst = HWC_C1
        elif input.shape[0] == 3:
            dst = CHW_C3
        elif input.shape[2] == 3:
            dst = HWC_C3
    elif input.ndim == 2:
        dst = HW_C1

    return dst


def check_input(input: numpy.ndarray, desire: Tuple[str, int], terminate: bool = False) -> bool:
    src: Tuple[str, int] = desire

    dst: Optional[Tuple[str, int]] = get_input_shape(input)
    if dst is None:
        if terminate:
            raise ValueError(f"Expected input shape is {desire[0]} and its number of channels is {desire[1]}, but got {input.shape}")

    return True if src == dst else False


def transpose_array(input: numpy.ndarray, desire: Tuple[str, int]) -> numpy.ndarray:
    src = get_input_shape(input)

    if desire[1] not in [1, 3]:
        raise ValueError(f"Expected number of channels is 1 or 3, but got {desire[1]}")

    if src is None:
        raise ValueError(f"Input shape is not supported")

    output = input.copy()

    map = {
        "HWC": {
            "HWC": [0 ,1, 2],
            "CHW": [2 ,0, 1],
        },
        "CHW": {
            "HWC": [1 ,2, 0],
            "CHW": [0 ,1, 2],
        }
    }

    if src[0] == "HW":
        if desire[0] == "HW":
            pass
        elif desire[0] == "HWC":
            output = output[..., None]

            if desire[1] == 3:
                output = numpy.dstack([output, output, output])
        elif desire[0] == "CHW":
            output = output[None, ...]

            if desire[1] == 3:
                output = numpy.stack([output, output, output], 0)
        else:
            raise ValueError(f"Expected order is HW or HWC or CHW, but got {desire[0]}")
    else:
        if desire[0] == "HW":
            if src[0] == "HWC":
                output = output[0]
            elif src[0] == "CHW":
                output = output[0, ...]
            else:
                raise RuntimeError
        else:
            try:
                output = output.transpose(map[src[0]][desire[0]])
            except KeyError:
                raise ValueError(f"Expected order is HW or HWC or CHW, but got {desire[0]}") from None

            if desire[1] == 1:
                if get_input_shape(output)[0] == "HWC":
                    output = output[0][..., None]
                elif get_input_shape(output)[0] == "CHW":
                    output = output[0, ...][None, ...]

    return output


if __name__ == "__main__":
    raise NotImplementedError
