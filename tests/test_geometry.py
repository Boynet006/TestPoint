import math

from testpoint.vision.geometry import angle_deg, distance, elevation_deg


def test_right_angle():
    assert math.isclose(angle_deg((0, 1), (0, 0), (1, 0)), 90.0, abs_tol=1e-6)


def test_straight_line():
    assert math.isclose(angle_deg((-1, 0), (0, 0), (1, 0)), 180.0, abs_tol=1e-6)


def test_degenerate_points_return_zero():
    assert angle_deg((0, 0), (0, 0), (1, 0)) == 0.0


def test_elevation_up_is_positive_and_side_independent():
    # image coords: y grows downward, so "up" is smaller y
    assert math.isclose(elevation_deg((0, 0), (10, -10)), 45.0, abs_tol=1e-6)
    assert math.isclose(elevation_deg((0, 0), (-10, -10)), 45.0, abs_tol=1e-6)
    assert math.isclose(elevation_deg((0, 0), (10, 0)), 0.0, abs_tol=1e-6)
    assert elevation_deg((0, 0), (10, 10)) < 0


def test_distance():
    assert distance((0, 0), (3, 4)) == 5.0
