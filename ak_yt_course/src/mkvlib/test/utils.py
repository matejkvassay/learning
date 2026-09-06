from torch import allclose, Tensor
from typing import Callable
import logging

logger = logging.getLogger(__name__)

TOL = 0.1


def assert_tensor_equal(func: Callable, expected_output: Tensor, *inputs: Tensor):
    result = func(*inputs)
    if isinstance(result, tuple):
        print(f'{func.__name__} call result detected as tuple": {result}')
        print(f'{func.__name__} ancillary outputs: {result[1:]}')
        result = result[0]

    try:
        assert allclose(result, expected_output, atol=TOL)
        assert result.shape == expected_output.shape
        assert result.dtype == expected_output.dtype
    except AssertionError as ex:
        print(f'Assert equal test failed for: {func.__name__}.')
        print(f'Inputs: {inputs}')
        print(f'Result: {result}')
        print(f'Expected result: {expected_output}')
        raise ex
