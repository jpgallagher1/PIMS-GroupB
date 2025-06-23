import numpy as np
import matplotlib.pyplot as plt
from may_work_implementation import transport_direct_solve, gausslobatto
import os

if __name__ == "__main__":
    # Parameters
    current_directory = os.getcwd()
    mu = 1.0 
    sigma_t = lambda x: x**2 # Try 1, 1.0, lambda x: x**2
    Np = 10 # default 10 for plots
    Ne = 20 # default 20 for plots
    xs = np.linspace(0, 1, Ne+1)

    # Define source and inflow
    qs = 1.0
    inflow = lambda x: 1.0

    # Solve
    psi = transport_direct_solve(mu, sigma_t, qs, inflow, Np, xs)

    # Plot
    mus, _ = gausslobatto(Np)
    X_plot = []
    for i in range(Ne):
        a, b = xs[i], xs[i+1]
        X_plot.extend((b*(mus+1) + a*(1-mus))/2)
    plt.plot(X_plot, psi, marker='o')
    plt.xlabel("x")
    plt.ylabel("psi(x)")
    plt.title("DG solution to 1D transport equation")
    plt.grid(True)
    if not os.path.exists("test_figures"):
        os.makedirs("test_figures")
    file_name = os.path.join(current_directory, "test_figures/transport_solution_sigma_t=x**2.png")
    plt.savefig(file_name)
    print(f"Plot saved as {file_name}")
    plt.close()

    sigma_t = 1
    psi = transport_direct_solve(mu, sigma_t, qs, inflow, Np, xs)

    # Plot
    mus, _ = gausslobatto(Np)
    X_plot = []

    for i in range(Ne):
        a, b = xs[i], xs[i+1]
        X_plot.extend((b*(mus+1) + a*(1-mus))/2)
    plt.plot(X_plot, psi, marker='o')
    plt.xlabel("x")
    plt.ylabel("psi(x)")
    plt.title("DG solution to 1D transport equation")
    plt.grid(True)
    file_name = os.path.join(current_directory, "test_figures/transport_solution_sigma_t=1.png")
    plt.savefig(file_name)
    print(f"Plot saved as {file_name}")
    plt.close()