import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from mltut import plot_derivative, seed_everything  # noqa: E402


def test_plot_derivative_matches_analytic_slope():
    f = lambda x: 3 * x**2 - 4 * x + 5  # f'(x) = 6x - 4
    fig, ax = plt.subplots()
    fx, slope = plot_derivative(f, 3.0, ax=ax)
    plt.close(fig)
    assert fx == pytest.approx(20.0)
    assert slope == pytest.approx(14.0, abs=1e-4)


def test_seed_everything_is_reproducible():
    seed_everything(0)
    a = np.random.rand(3)
    seed_everything(0)
    b = np.random.rand(3)
    np.testing.assert_array_equal(a, b)
