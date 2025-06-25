import numpy as np
import matplotlib.pyplot as plt
from library import *

# Parameters for the transport problem
μ       = 0.5
σ_t     = lambda x: x**2 + 1
σ_a     = lambda x: x**2 + 1

# Mesh parameters
xs      = np.linspace(0, 1, 21) # Mesh points / element boundaries
Np      = 5                     # Legendre polynomials per element (basis size)
for_TSA = False

# Set \epsilon hyperparameter
epsilon = 1e-3

# Get F, F_{||} from F^+ and F^- via assemble_face_matrices
F_plus, F_minus = assemble_face_matrices(Np, xs, for_TSA=for_TSA)
F = F_plus + F_minus
F_parallel = F_plus - F_minus

# Get G from assemble_deriv_matrix
G = assemble_deriv_matrix(Np, xs)

# Get M_t/a via sigma_t/a
M_t = assemble_mass_matrix(σ_t, Np, xs)
M_a = assemble_mass_matrix(σ_a, Np, xs)

# Get alpha from omega_k and mu_k
N_mu = 3*Np
mu_k, w_k = gausslegendre(N_mu)
alpha = 0.5 * np.dot(w_k, np.abs(mu_k))

# Precompute M_t_inv and (G - 1/2 F)
M_t_inv = np.linalg.inv(M_t)
G_minus_half_of_F = G - 0.5*F

# Compute D from equations 18 and 35
D_18 = (1.0/3.0)*G_minus_half_of_F.T @ M_t_inv @ G_minus_half_of_F + alpha*F_parallel + M_a
D_35 = (alpha/epsilon)*F_parallel - ((1.0/3.0)*(M_t_inv @ G_minus_half_of_F @ M_t_inv @ G_minus_half_of_F) - (M_t_inv @ M_a))


def plot_D_spectrum(D_dict, epsilon, save_plot=False, save_name="D_spectrum_plot"):
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
        plt.close(fig)
    else:
        plt.show()


def plot_elements_of_matrix_mul(D_dict, epsilon, save_plot=False, save_name="elements_of_matrix_mul_plot"):
    for D_str in D_dict.keys():
        if D_str == 'D_35':
            D = D_dict[D_str]
        else:
            D = D_dict[D_str]
        I_minus_T_inv_perturbation = (epsilon**2)*D

        n = D.shape[0]
        I = np.eye(n)
        mat_mul = (I - (epsilon**-2)*np.linalg.inv(D)) @ I_minus_T_inv_perturbation
        plt.imshow(mat_mul)
        plt.colorbar()
        plt.xlabel('n')
        plt.ylabel('m')
        plt.title(r'$(I - \epsilon^{-2}D^{-1})(I-T)^{-1}$ with '+D_str)
        if save_plot:
            import os
            current_directory = os.getcwd()
            if not os.path.exists("test_figures"):
                os.makedirs("test_figures")
            file_name = os.path.join(current_directory, f"test_figures/{save_name}_{D_str}.png")
            plt.savefig(file_name)
            print(f"Plot saved as {file_name}")
            plt.close()
        else:
            plt.show()

D_dict = {
    'D_18': D_18,
    'D_35': D_35}

plot_D_spectrum(D_dict, epsilon, save_plot=True, save_name="D_spectrum_plot")
plot_elements_of_matrix_mul(D_dict, epsilon, save_plot=True)