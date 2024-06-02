import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline


def demo_b_spline_curve():
    # Define control points that form a curve
    control_points = np.array([
        [0, 0],
        [0.1, 0.5],
        [0.4, 0.9],
        [0.7, 0.6],
        [1, 0.2]
    ])

    # Knot vector for clamped B-spline
    n = len(control_points)
    k = 3  # cubic B-spline
    t = np.concatenate(([0] * k, np.linspace(0, 1, n - k + 1), [1] * k))

    # Parameter values
    u = np.linspace(0, 1, 100)

    # Create B-spline
    spline = BSpline(t, control_points, k)

    # Evaluate B-spline
    spline_points = spline(u)

    # Plot control points and B-spline
    with plt.style.context('ggplot'):
        plt.figure(figsize=(4, 3))
        plt.plot(control_points[:, 0], control_points[:, 1], 'ro-', label='Control Points', alpha=0.5)
        plt.plot(spline_points[:, 0], spline_points[:, 1], 'b-', label='B-spline Curve', alpha=0.5)
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('B-spline Curve')
        plt.grid(True)
        plt.savefig('demo_splines_1.png')
        plt.show()


# Run the demo function
demo_b_spline_curve()
