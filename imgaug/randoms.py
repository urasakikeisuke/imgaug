"""random.py"""

import random
from typing import Any, List, Optional, Tuple, Union

import numpy


def seed(seed: Optional[int]):
    """Set a seed value manually

    Args:
        seed (Union[int, float]): A seed value
    """    
    if seed is not None:
        if not isinstance(seed, int):
            raise TypeError(f"Expect `seed` type is `int`, but got {type(seed).__name__}")

        random.seed(seed)
        numpy.random.seed(seed)


class RandomGenerator(object):
    def __init__(
        self,
        range_: Tuple[Union[int, float], Union[int, float]],
    ) -> None:
        super().__init__()

        self.range_ = None

        if isinstance(range_, tuple):
            if len(range_) == 2:
                self.range_: Tuple[Union[int, float], Union[int, float]] = range_
            else:
                raise ValueError(f'Expected `range_` length is 2 but got {len(range_)}')
        else:
            raise TypeError(f'Expected `range_` type is `tuple` but got {type(range_).__name__}')

    def __call__(self):
        if isinstance(self.range_[0], int) and isinstance(self.range_[1], int):
            return random.randrange(self.range_[0], self.range_[1])
        else:
            return random.uniform(self.range_[0], self.range_[1])


class SomeOf(object):
    def __init__(
        self,
        p: float,
        children: Optional[List[Any]],
        random_order: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(p, float):
            self.p = p
        else:
            raise TypeError(f"Expected type is `float`, but got `{type(p).__name__}`")

        if random_order:
            random.shuffle(children)
        
        self.children: List[Any] = children

    def __call__(self, *input: numpy.ndarray) -> numpy.ndarray:
        if self.children is not None:
            for augmentor in self.children:
                if self.p > random.random():
                    input = augmentor.exec(*input)

        return input


if __name__ == '__main__':
    raise NotImplementedError