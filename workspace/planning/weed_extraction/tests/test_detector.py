"""The weed detector's geometry: a box in the bed camera picture becomes a place on the bed."""

import numpy as np
import pytest

pytest.importorskip("torchvision")
from weed_extraction.detector import WeedDetector, place  # noqa: E402

# The bed camera as the scene mounts it with the base at x = 0: 1.61 m up, 0.14 m behind
# the base, looking forward and 55 degrees down.
CAM_POS = np.array([-0.14, 0.0, 1.61])
CAM_MAT = np.array([[0.0, 0.819, -0.574], [-1.0, 0.0, 0.0], [0.0, 0.574, 0.819]])


def box_around(u, v):
    return np.array([u - 5, v - 5, u + 5, v + 5])


def test_the_picture_centre_lies_on_the_camera_axis():
    half = WeedDetector.SIZE / 2
    x, y = place(CAM_POS, CAM_MAT, 45.0, box_around(half, half), height=0.1)
    forward = -CAM_MAT[:, 2]
    assert y == pytest.approx(0.0, abs=1e-9)
    assert x == pytest.approx(CAM_POS[0] + forward[0] * (0.1 - CAM_POS[2]) / forward[2])


def test_picture_left_is_the_left_bed_and_picture_top_is_farther_ahead():
    half = WeedDetector.SIZE / 2
    centre = place(CAM_POS, CAM_MAT, 45.0, box_around(half, half), height=0.1)
    left = place(CAM_POS, CAM_MAT, 45.0, box_around(half - 100, half), height=0.1)
    top = place(CAM_POS, CAM_MAT, 45.0, box_around(half, half - 100), height=0.1)
    assert left[1] > centre[1]  # left is +y
    assert top[0] > centre[0]
