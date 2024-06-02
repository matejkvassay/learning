import torch

from mkvlib.normalize import normalize_zscore
from mkvlib.test.utils import assert_tensor_equal


def test_normalize_zscore():
    data_in = torch.tensor([
        [2, 5, 8],
        [2, 7, 4]
    ], dtype=torch.float64)

    normalized = torch.tensor([
        [0, -0.707106, 0.707106],
        [0, 0.707106, -0.707106]
    ], dtype=torch.float64)

    assert_tensor_equal(normalize_zscore, normalized, data_in)
