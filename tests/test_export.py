import numpy as np

from subway.export import simplify


def test_simplify_drops_collinear_points_and_keeps_corners():
    # 0..100 m along x (10 m steps), then a right-angle turn up to y = 50
    xy = np.array([[x, 0.0] for x in range(0, 101, 10)] + [[100.0, y] for y in range(10, 51, 10)])
    xy[3, 1] = 1.0                      # 1 m wiggle: below a 2 m tolerance
    keep = simplify(xy, 2.0)
    assert keep.tolist() == [0, 10, len(xy) - 1]


def test_simplify_keeps_deviations_above_tolerance():
    xy = np.array([[0.0, 0.0], [50.0, 5.0], [100.0, 0.0]])
    assert simplify(xy, 2.0).tolist() == [0, 1, 2]
    assert simplify(xy, 6.0).tolist() == [0, 2]
