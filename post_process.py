import numpy as np
import matplotlib.pyplot as plt
from library import gausslobatto, gausslegendre, eval_pk


def ξ_to_x(ξ, a, b):
    """Map the points ξ in [-1, 1] (reference element) to x in [a, b]."""
    return 0.5 * (b - a) * (ξ + 1.0) + a


def error_Lp(ψ_weights, xs, Np, exact_ψ_func, p=2):
    """Compute the L2 error of the DG solution against an exact solution."""
    Ne = len(xs) - 1
    # interpolation nodes for Legendre basis funcs v_m
    ξ_b, _ = gausslobatto(Np)
    ξ_q, w_q = gausslegendre(3*Np)  # quadrature nodes, ξ_q in [-1, 1]

    error = 0.0
    for je in range(Ne):
        a, b = xs[je], xs[je+1]
        x_q = ξ_to_x(ξ_q, a, b)   # mapped quad nodes, x_q in [a, b]

        ψ_vals = np.zeros_like(ξ_q)
        for n in range(Np):
            ψ_vals += ψ_weights[je, n] * eval_pk(ξ_q, n, ξ_b)

        exact_vals = exact_ψ_func(x_q)
        diff = ψ_vals - exact_vals
        if p == 'inf':
            error = max(error, np.max(np.abs(diff)))
        else:
            error += np.sum(w_q * (ψ_vals - exact_vals)**p) * (b-a)/2.0
    return error if p == 'inf' else error**(1/p)


def plot_solution(ψ_weights, xs, Np, μ=None, num_plot_pts=200, exact_ψ_func=None, save_plot=False, show_plot=True):
    """
    Reconstructs and plots the DG solution ψ(x) over the mesh xs.

    Plot optional:  saved to "test_figures/transport_solution.png" if save_plot=True.
    """

    Ne = len(xs) - 1
    # interpolation nodes for Legendre basis ("b") funcs v_m
    ξ_b, _ = gausslobatto(Np)
    # plot points (in reference space [-1,1]) for each element
    ξ_p = np.linspace(-1, 1, num_plot_pts)

    if ψ_weights.ndim == 1:
        Ne
        ψ_weights = ψ_weights.reshape((Ne, -1))

    plt.figure()
    for je in range(Ne):
        a, b = xs[je], xs[je+1]
        x_p = ξ_to_x(ξ_p, a, b)

        # Reconstruct polynomial on this element
        ψ_weights_loc = ψ_weights[je, :]
        ψ_vals = np.zeros_like(ξ_p)
        for n in range(Np):
            ψ_vals += ψ_weights_loc[n] * eval_pk(ξ_p, n, ξ_b)
        plt.plot(x_p, ψ_vals, '-')

    if exact_ψ_func is not None:
        exactx = np.linspace(xs[0], xs[-1], num_plot_pts)
        exactψ = exact_ψ_func(exactx)
        plt.plot(exactx, exactψ, 'k--', label='Exact solution')
        plt.legend()

    plt.xlabel('x')
    plt.ylabel(r'$\psi(x)$')
    if μ is not None:
        plt.title(f'DG solution, μ={μ}, Np={Np}, Ne={Ne}')
    plt.grid(True)

    if save_plot:
        import os
        current_directory = os.getcwd()
        if not os.path.exists("test_figures"):
            os.makedirs("test_figures")
        file_name = os.path.join(
            current_directory, "test_figures/transport_plot.png")
        plt.savefig(file_name, dpi=500)
        print(f"Plot saved as {file_name}")
    elif show_plot:
        plt.show()
