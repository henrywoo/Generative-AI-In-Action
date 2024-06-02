import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, make_interp_spline, interp1d


def demo_splines():
    # Define the original function and sampling points
    def f(x):
        return np.piecewise(x, [x < 0.5, x >= 0.5], [0, 1])

    # Sample points
    x = np.linspace(0, 1, 10)
    y = f(x)

    # Create a dense set of points for plotting the true function
    x_dense = np.linspace(0, 1, 1000)
    y_dense = f(x_dense)

    # Cubic spline interpolation
    cubic_spline_interp = CubicSpline(x, y)
    y_cubic_spline = cubic_spline_interp(x_dense)

    # 5th-order spline interpolation
    poly_5th_order = np.poly1d(np.polyfit(x, y, 5))
    y_poly_5th_order = poly_5th_order(x_dense)

    # 7th-order spline interpolation
    poly_7th_order = np.poly1d(np.polyfit(x, y, 7))
    y_poly_7th_order = poly_7th_order(x_dense)

    # Linear spline interpolation
    linear_spline_interp = interp1d(x, y, kind='linear')
    y_linear_spline = linear_spline_interp(x_dense)

    # Plotting the results
    with plt.style.context('ggplot'):
        fig, axs = plt.subplots(2, 2, figsize=(9, 7))
        axs[0, 0].plot(x_dense, y_dense, 'r', label='Original function')
        axs[0, 0].plot(x_dense, y_cubic_spline, 'b--', label='Cubic spline')
        axs[0, 0].scatter(x, y, color='black')
        axs[0, 0].set_title('Cubic interpolation')

        axs[0, 1].plot(x_dense, y_dense, 'r', label='Original function')
        axs[0, 1].plot(x_dense, y_poly_5th_order, 'g--', label='5th-order spline')
        axs[0, 1].scatter(x, y, color='black')
        axs[0, 1].set_title('5th-order interpolation')

        axs[1, 0].plot(x_dense, y_dense, 'r', label='Original function')
        axs[1, 0].plot(x_dense, y_poly_7th_order, 'm--', label='7th-order spline')
        axs[1, 0].scatter(x, y, color='black')
        axs[1, 0].set_title('7th-order interpolation')

        axs[1, 1].plot(x_dense, y_dense, 'r', label='Original function')
        axs[1, 1].plot(x_dense, y_linear_spline, 'c--', label='Linear spline')
        axs[1, 1].scatter(x, y, color='black')
        axs[1, 1].set_title('Linear spline interpolation')

        for ax in axs.flat:
            ax.legend()
            ax.set_xlabel('x')
            ax.set_ylabel('f(x)')

        plt.tight_layout()
        plt.savefig('demo_splines_2.png')
        plt.show()


# Run the demo function
demo_splines()
