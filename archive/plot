if __name__ == "__main__":
    # Parameters
    mu = 1.0
    sigma_t = 1.0
    Np = 10
    Ne = 20
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
    plt.show()
