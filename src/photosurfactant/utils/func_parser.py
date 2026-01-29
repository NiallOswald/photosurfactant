from typing import Callable

import numpy as np  # noqa: F401

from photosurfactant.intensity_functions import *  # noqa: F403
from photosurfactant.parameters import Parameters


def parse_func(func_str: str) -> Callable[[float, Parameters], float]:
    return eval("lambda x, params: " + func_str)
