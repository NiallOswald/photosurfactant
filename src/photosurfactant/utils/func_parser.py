from typing import Callable
from photosurfactant.parameters import Parameters

from photosurfactant.intensity_functions import *  # noqa: F403
import numpy as np  # noqa: F403


def parse_func(func_str: str) -> Callable[[float, Parameters], float]:
    return eval("lambda x, params: " + func_str)
