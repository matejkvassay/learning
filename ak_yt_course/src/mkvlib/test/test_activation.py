import pytest
import torch
from mkvlib.activation import sigmoid
from mkvlib.test.utils import assert_tensor_equal


@pytest.mark.parametrize("target_shape", [(1, -1), (2, 3), (3, 2)])
def test_sigmoid(target_shape):
    data_in = [4.1, 0.333, 1., 0., 22.4, -50.1]
    data_out = [0.9836975, 0.5824891, 0.7310585, 0.5, 0.9999, 0.]
    tensor_in = torch.tensor(data_in).view(target_shape)
    sigmoids_out = torch.tensor(data_out).view(target_shape)
    assert_tensor_equal(sigmoid, sigmoids_out, tensor_in)
