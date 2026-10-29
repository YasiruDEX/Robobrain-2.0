import numpy as np
from metrics import traj_metrics, box_iou, resample, discrete_frechet, hausdorff
from parse import parse_points, parse_box


def test_identical_traj_zero():
    t = [(0.1, 0.1), (0.5, 0.5), (0.9, 0.2)]
    m = traj_metrics(t, t)
    assert all(abs(v) < 1e-9 for v in m.values()), m


def test_shift():
    g = [(0.1, 0.1), (0.5, 0.1)]
    p = [(0.1, 0.2), (0.5, 0.2)]
    m = traj_metrics(p, g)
    for k in ("dfd", "hd", "rmse", "ade", "fde"):
        assert abs(m[k] - 0.1) < 1e-9, (k, m[k])


def test_resample_density_invariant():
    a = resample([(0, 0), (1, 0)], 5)
    b = resample([(0, 0), (0.3, 0), (1, 0)], 5)
    assert np.allclose(a, b)


def test_frechet_ge_hausdorff():
    rng = np.random.default_rng(0)
    for _ in range(50):
        p, q = rng.random((6, 2)), rng.random((8, 2))
        assert discrete_frechet(p, q) >= hausdorff(p, q) - 1e-12


def test_iou():
    assert box_iou((0, 0, 1, 1), (0, 0, 1, 1)) == 1.0
    assert abs(box_iou((0, 0, 2, 1), (1, 0, 3, 1)) - 1 / 3) < 1e-9
    assert box_iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0.0


def test_parse():
    s = "[(100, 50), (200, 100)]"
    assert parse_points(s, (400, 200)) == [(0.25, 0.25), (0.5, 0.5)]
    assert parse_points("[[100, 50], [200, 100]]", (400, 200)) == [(0.25, 0.25), (0.5, 0.5)]
    assert parse_box("box: [300, 150, 100, 50]", (400, 200)) == (0.25, 0.25, 0.75, 0.75)
    assert parse_box("no box", (1, 1)) is None
    assert parse_points("[(0.25, 0.5), (0.1, 0.9)]", (400, 200)) == [(0.25, 0.5), (0.1, 0.9)]
    assert parse_points("[(1, 0)]", (400, 200)) == [(1 / 400, 0.0)]  # integer pixels stay pixels


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
