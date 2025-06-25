from library import *
from typing import Callable


def compute_source_term(source: Callable[[float], float],
                        Np: int,
                        xs: np.ndarray) -> np.ndarray:
    """
    Compute the source term for the transport equation.
        source : Source term func (a func of x)
        Np     : Number of Legendre basis funcs per element
        xs     : DG mesh points defining the domain [x0, ..., xn]
    Returns:
        qs : Source term vector
    """
    Ne = len(xs) - 1
    s, ws = gausslobatto(Np)  # Interpolation points and weights
    s2, ws2 = gausslegendre(3*Np)  # For integration
    qs = np.zeros(Ne * Np)
    for je in range(Ne):
        a = xs[je]
        b = xs[je + 1]
        for m in range(Np):
            val = 0.0
            for k in range(len(s2)):
                x = s2[k]
                # Map from reference [-1,1] to [a,b]
                y = b * (x + 1) / 2 + a * (1 - x) / 2
                val += ws2[k] * eval_pk(x, m, s) * source(y)
            qs[je * Np + m] = (b - a) / 2.0 * val
    return qs


def transport_direct_solve(mu: float,
                           sigma_t: Callable[[float], float],
                           qs: np.ndarray,
                           inflow: Callable[[float], float],
                           Np: int,
                           xs: np.ndarray,
                           F_plus: np.ndarray = None, F_minus: np.ndarray = None,
                           G: np.ndarray = None, M: np.ndarray = None
                           ) -> np.ndarray:
    """
    Solve the transport eq using a DG + collocation (discrete ordinates) method.
        mu      : Transport coefficient (+ for rightward transport, - for leftward)
        sigma_t : Total scattering opacity func (a func of x)
        qs      : Source term vector
        inflow  : Inflow term func (a func of x)
        Np      : Number of Legendre basis funcs per element
        xs      : DG mesh points defining the domain [x0, ..., xn]
        F_plus  : Face matrix for rightward transport, precomputed to reduce function overhead
        F_minus : Face matrix for leftward transport, precomputed to reduce function overhead
        G       : Derivative matrix, precomputed to reduce function overhead
        M       : Mass matrix, precomputed to reduce function overhead
    Returns:
        ψ_weights : Solution vector containing the weights of the polynomial basis funcs
    """
    if mu > 0:
        psi = transport_direct_solve_plus(mu, sigma_t, qs, inflow, Np, xs,
                                          F_plus, F_minus, G, M)
    else:
        psi = transport_direct_solve_minus(mu, sigma_t, qs, inflow, Np, xs,
                                          F_plus, F_minus, G, M)
    return psi


def transport_direct_solve_diffusive(sigma_t: Callable[[float], float],
                                     sigma_a: Callable[[float], float],
                                     e: float,
                                     source: Callable[[float, float], float],
                                     inflow: Callable[[float, float], float],
                                     Np: int,
                                     Nμ: int,
                                     xs: np.ndarray,
                                     max_iter: int = 1000,
                                     tol: float = None
                                     ) -> np.ndarray:
    """
    Solve the transport eq using a fixed point iteration method with 
        sigma_t    : Total scattering opacity func (a func of x)
        sigma_a    : Absorption scattering opacity func (a func of x)
        e          : Scattering parameter
        source     : Source term func (a func of x and μ)
        inflow     : Inflow term func (a func of x and μ)
        Np         : Number of Legendre basis funcs per element
        Nμ         : Number of polynomial degrees in μ direction (number of Gauss-Legendre points)
        xs         : DG mesh points defining the domain [x0, ..., xn]
        max_iter   : Maximum number of iterations for fixed point iteration
        tol        : Tolerance for fixed point iteration
    Returns:
        psi_weights: Solution vector containing the weights of the polynomial basis funcs
        mus        : Array of μ values used in the solution
    """
    Ne = len(xs) - 1

    # Assemble matrices
    M_s = assemble_mass_matrix(lambda x:
                               sigma_t(x)/e - e*sigma_a(x),
                               Np, xs)  # shape (Ne*Np, Ne*Np)
    M_t = assemble_mass_matrix(lambda x:
                               sigma_t(x)/e,
                               Np, xs)
    F_plus, F_minus = assemble_face_matrices(Np, xs)
    G = assemble_deriv_matrix(Np, xs)

    mus, ws = gausslobatto(Nμ)

    # For each μ, for each element, we store the weight vector
    psi_weights_all = np.zeros((Nμ, Ne*Np))

    for t in tqdm.tqdm(range(max_iter)):
        psi_weights_all_old = psi_weights_all.copy()
        # Compute integral from -1 to 1 of psiμ by quadrature
        phi = (ws.reshape((-1, 1)) * psi_weights_all).sum(axis=0)  # shape (Ne*Np)
        Ms_phi = 1/2 * M_s @ phi

        for i_mu, mu in enumerate(mus):
            # Compute RHS
            qs = e * compute_source_term(lambda x: source(x, mu), Np, xs)
            psi_weights_all[i_mu] = transport_direct_solve(
                mu, lambda x: sigma_t(x)/e, Ms_phi + qs, lambda x: inflow(x, mu), Np, xs,
                F_plus, F_minus, G, M_t
                )

        diff = np.linalg.norm(psi_weights_all - psi_weights_all_old)
        if tol is not None and diff < tol:
            print(f"Converged in {t} iterations")
            break
    return psi_weights_all.reshape((Nμ, Ne, Np)), mus, t