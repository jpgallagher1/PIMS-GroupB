import numpy as np
import matplotlib.pyplot as plt
from library import *

# Parameters for the transport problem
μ       = 0.5
σ_t     = lambda x: x**2 + 1
σ_a     = lambda x: x**2 + 1

# Mesh parameters
xs      = np.linspace(0, 1, 21)
Np      = 5
for_TSA = False

# Set \epsilon hyperparameter
epsilon = 1e-3

# Get alpha from omega_k and mu_k
N_mu = 3*Np
mu_k, w_k = gausslegendre(N_mu)
alpha = 0.5 * np.dot(w_k, np.abs(mu_k))


def compute_D(epsilon, alpha, Np, xs, σ_t, σ_a, for_TSA=for_TSA):
    F_plus, F_minus = assemble_face_matrices(Np, xs, for_TSA=for_TSA)
    F = F_plus + F_minus
    F_parallel = F_plus - F_minus
    M_t = assemble_mass_matrix(σ_t, Np, xs)/epsilon
    M_a = assemble_mass_matrix(σ_a, Np, xs)*epsilon
    M_t_inv = np.linalg.inv(M_t)
    G = assemble_deriv_matrix(Np, xs)
    G_minus_half_of_F = G - 0.5*F
    return (alpha / epsilon) * F_parallel - ((1.0 / 3.0) * (M_t_inv @ G_minus_half_of_F @ M_t_inv @ G_minus_half_of_F) - (M_t_inv @ M_a))


def plot_D_spectrum(D_dict, epsilon, save_plot=False, save_name="D_spectrum_plot"):
    """ Plot the spectral components of the transport approximations. """


    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Spectral Components of Transport Approximations", fontsize=14)

    # First plot: real part of eigenvalues of (I - T)^(-1) ≈ ε² D
    ax = axes[0][0]
    for label in D_dict:
        D = D_dict[label]
        matrix = epsilon**2 * D
        eigvals = np.linalg.eigvals(matrix)
        real_vals = np.real(eigvals)
        ax.plot(range(len(real_vals)), real_vals, label=label)
    ax.set_title('Re eigs of (I - T)$^{-1}$')
    ax.set_xlabel('eigenvalue index')
    ax.set_ylabel('value')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True)

    # Second plot: imaginary part of eigenvalues of (I - T)^(-1)
    ax = axes[0][1]
    for label in D_dict:
        D = D_dict[label]
        matrix = epsilon**2 * D
        eigvals = np.linalg.eigvals(matrix)
        imag_vals = np.imag(eigvals)
        ax.plot(range(len(imag_vals)), imag_vals, label=label)
    ax.set_title('Im eigs of (I - T)$^{-1}$')
    ax.set_xlabel('eigenvalue index')
    ax.set_ylabel('value')
    ax.legend()
    ax.grid(True)

    # Third plot: real part of eigenvalues of I - T ≈ (ε² D)^(-1)
    ax = axes[1][0]
    for label in D_dict:
        D = D_dict[label]
        matrix = np.linalg.inv(epsilon**2 * D)
        eigvals = np.linalg.eigvals(matrix)
        real_vals = np.real(eigvals)
        ax.plot(range(len(real_vals)), real_vals, label=label)
    ax.set_title('Re eigs of I - T')
    ax.set_xlabel('eigenvalue index')
    ax.set_ylabel('value')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True)

    # Fourth plot: imaginary part of eigenvalues of I - T
    ax = axes[1][1]
    for label in D_dict:
        D = D_dict[label]
        matrix = np.linalg.inv(epsilon**2 * D)
        eigvals = np.linalg.eigvals(matrix)
        imag_vals = np.imag(eigvals)
        ax.plot(range(len(imag_vals)), imag_vals, label=label)
    ax.set_title('Im eigs of I - T')
    ax.set_xlabel('eigenvalue index')
    ax.set_ylabel('value')
    ax.legend()
    ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_plot:
        import os
        current_directory = os.getcwd()
        if not os.path.exists("test_figures"):
            os.makedirs("test_figures")
        file_name = os.path.join(current_directory, f"test_figures/{save_name}.png")
        plt.savefig(file_name)
        print(f"Plot saved as {file_name}")
        plt.close()
    else:
        plt.show()


def plot_elements_of_matrix_mul(D_dict, epsilon, save_plot=False, save_name="elements_of_matrix_mul"):
    num_matrices = len(D_dict)
    fig, axes = plt.subplots(1, num_matrices, figsize=(6 * num_matrices, 5))

    if num_matrices == 1:
        axes = [axes]

    for ax, (D_str, D) in zip(axes, D_dict.items()):
        I_minus_T_inv_perturbation = (epsilon**2) * D
        n = D.shape[0]
        I = np.eye(n)
        mat_mul = (I - (epsilon**-2) * np.linalg.inv(D)) @ I_minus_T_inv_perturbation

        im = ax.imshow(mat_mul)
        fig.colorbar(im, ax=ax)
        ax.set_xlabel('n')
        ax.set_ylabel('m')
        ax.set_title(r'$(I - \epsilon^{-2}D^{-1})(I-T)^{-1}$ with ' + D_str)

    plt.tight_layout()
    if save_plot:
        import os
        current_directory = os.getcwd()
        if not os.path.exists("test_figures"):
            os.makedirs("test_figures")
        file_name = os.path.join(current_directory, f"test_figures/{save_name}.png")
        plt.savefig(file_name)
        print(f"Plot saved as {file_name}")
        plt.close()
    else:
        plt.show()


def plot_singular_values_comparison(D_dict, epsilon, save_plot=False, save_name="singular_values_comparison"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Singular Value Spectra of $(I - T)^{-1}$ and $I - T$", fontsize=14)

    # Left subplot: singular values of (I - T)^(-1) ≈ ε² D
    ax = axes[0]
    for label in D_dict:
        D = D_dict[label]
        approx_inv = epsilon**2 * D
        svals = np.linalg.svd(approx_inv, compute_uv=False)
        print(f'largest singular value of inv(I-T) = {svals[0]}')
        ax.plot(range(len(svals)), svals, label=label)
    ax.set_title('Singular values of $(I - T)^{-1}$')
    ax.set_xlabel('index')
    ax.set_ylabel('value')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True)

    # Right subplot: singular values of I - T ≈ inv(ε² D)
    ax = axes[1]
    for label in D_dict:
        D = D_dict[label]
        approx_direct = np.linalg.inv(epsilon**2 * D)
        svals = np.linalg.svd(approx_direct, compute_uv=False)
        ax.plot(range(len(svals)), svals, label=label)
    ax.set_title('Singular values of $I - T$')
    ax.set_xlabel('index')
    ax.set_ylabel('value')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    if save_plot:
        import os
        current_directory = os.getcwd()
        if not os.path.exists("test_figures"):
            os.makedirs("test_figures")
        file_name = os.path.join(current_directory, f"test_figures/{save_name}.png")
        plt.savefig(file_name)
        print(f"Plot saved as {file_name}")
        plt.close()
    else:
        plt.show()

print(f'epsilon = {epsilon}')


D_dict = {
    'D': compute_D(epsilon, alpha, Np, xs, σ_t, σ_a, for_TSA=for_TSA)
    }

plot_D_spectrum(D_dict, epsilon, save_plot=True)
plot_singular_values_comparison(D_dict, epsilon, save_plot=True)
plot_elements_of_matrix_mul(D_dict, epsilon, save_plot=True)