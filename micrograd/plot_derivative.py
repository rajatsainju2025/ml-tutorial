import numpy as np
import matplotlib.pyplot as plt


def plot_derivative(f, x, h=1e-6, xs=None, ax=None, span=1.0):
    """Plot f, mark the point (x, f(x)), and draw the tangent line (derivative direction).

    Returns (f(x), derivative) so the values can also be used or printed.
    """
    if xs is None:
        xs = np.arange(-5, 5, 0.25)
    if ax is None:
        ax = plt.gca()

    fx = f(x)
    slope = (f(x + h) - fx) / h  # numerical derivative

    # function curve and the point
    ax.plot(xs, f(xs), label="f(x)")
    ax.scatter([x], [fx], color="red", zorder=3)

    # tangent line segment centred on the point: y = fx + slope * (t - x)
    t = np.array([x - span, x + span])
    ax.plot(t, fx + slope * (t - x), "--", color="red", label="tangent")

    # arrow showing the direction of increase (along the slope)
    ax.annotate("", xy=(x + span * 0.6, fx + slope * span * 0.6), xytext=(x, fx),
                arrowprops=dict(arrowstyle="->", color="green", lw=2))

    # text with f(x) and derivative
    ax.text(0.03, 0.95, f"x = {x}\nf(x) = {fx:.4f}\nf'(x) ≈ {slope:.4f}",
            transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    return fx, slope


if __name__ == "__main__":
    f = lambda x: 3 * x**2 - 4 * x + 5
    plot_derivative(f, 3.0)
    plt.show()
