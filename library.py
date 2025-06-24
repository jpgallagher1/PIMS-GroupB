import numpy as np
import matplotlib.pyplot as plt
import os
# from   numpy.polynomial.legendre import Legendre

def ξ_to_x(ξ, a, b):
    """Map the points ξ in [-1, 1] (reference element) to x in [a, b]."""
    return 0.5 * (b - a) * (ξ + 1.0) + a

def gausslegendre(N):
    ξ, w = np.polynomial.legendre.leggauss(N)
    return ξ, w

def gausslobatto(N):
    ξ       = np.zeros(N)
    ξ[0]    = -1.0
    ξ[-1]   = 1.0

    Pn_1    = np.polynomial.legendre.Legendre.basis(N-1)
    dPn_1   = Pn_1.deriv()
    ξ[1:-1] = np.sort(dPn_1.roots())

    # Compute weights
    w = np.zeros(N)
    for i in range(N):
        ξi       = ξ[i]
        Pn_i_val = Pn_1(ξi)
        w[i]     = 2.0 / (N * (N-1) * (Pn_i_val**2))
    return ξ, w

# print(gausslegendre(3))
# print(gausslobatto(3))

def eval_pk(ξ, i, ξ_b):
    """Compute the i-th Lagrange basis polynomial at x"""
    terms = [(ξ - ξ_b[j])/(ξ_b[i] - ξ_b[j]) for j in range(len(ξ_b)) if j != i]
    return np.prod(terms, axis=0)

def eval_pk_deriv(ξ, i, ξ_b):
    """Compute the derivative of the i-th Lagrange basis polynomial at x"""
    Np     = len(ξ_b)
    result = 0.0
    for j in range(Np):
        if j == i: continue
        term = 1.0 / (ξ_b[i] - ξ_b[j])
        for k in range(Np):
            if k == i or k == j: continue
            term *= (ξ - ξ_b[k]) / (ξ_b[i] - ξ_b[k])
        result += term
    return result

# print(eval_pk(np.array([-1, 0, 1]), 0, np.array([-1, 0, 1])))
# print(eval_pk_deriv(np.array([-1, 0, 1]), 0, np.array([-1, 0, 1])))

def compute_mass_matrix(σ_t, a, b, Np):
    """
    Full (consistent) mass matrix on [a,b]:
      Me[m,n] = (b-a)/2 * sum_k{ w_k * σ_t(x_k) * l_m(ξ_k) * l_n(ξ_k) }
    """
    ξ_b, _   = gausslobatto(Np)    # interpolation nodes for Legendre basis funcs v_m
    ξ_q, w_q = gausslegendre(3*Np) # quadrature nodes,  ξ_q in [-1, 1]
    x_q      = ξ_to_x(ξ_q, a, b)   # mapped quad nodes, x_q in [a, b]
    Me = np.zeros((Np, Np)) # Mass matrix local to element e
    for m in range(Np):
        for n in range(Np):
            val = 0.0
            for k in range(3*Np):
                lm   = eval_pk(ξ_q[k], m, ξ_b)
                ln   = eval_pk(ξ_q[k], n, ξ_b)
                σ_t  = σ_t(x_q[k]) if callable(σ_t) else σ_t
                val += w_q[k] * σ_t * lm * ln
            Me[m, n] = (b-a)/2 * val
    return Me

# print(compute_mass_matrix(lambda x: 1.0, 0, 1, 3))

def compute_deriv_matrix(a, b, Np):
    """
    Full (consistent) derivative matrix on [a,b]:
        Ge[m,n] = sum_k{ w_k * d/dξ l_m(ξ_k) * l_n(ξ_k) }
    """
    ξ_b, _   = gausslobatto(Np)    # interpolation nodes for Legendre basis ("b") funcs v_m
    ξ_q, w_q = gausslegendre(3*Np) # quadrature nodes, ξ_q in [-1, 1]

    Ge = np.zeros((Np, Np))
    for m in range(Np):
        for n in range(Np):
            val = 0.0
            for k in range(3*Np):
                dlmdx = eval_pk_deriv(ξ_q[k], m, ξ_b)
                ln    = eval_pk(ξ_q[k], n, ξ_b)
                val  += w_q[k] * dlmdx * ln
            Ge[m, n] = val
    return Ge

# print(compute_deriv_matrix(0,1,3))

def assemble_mass_matrix(σ_t, Np, xs):
    """Assemble matrix M"""
    Ne = len(xs) - 1
    M  = np.zeros((Np*Ne, Np*Ne))
    for je in range(Ne):
        M_local = compute_mass_matrix(σ_t, xs[je], xs[je+1], Np)
        for n in range(Np):
            for m in range(Np):
                M[je*Np + m, je*Np + n] = M_local[m, n]
    return M

def assemble_deriv_matrix(Np, xs):
    """Assemble matrix G"""
    Ne = len(xs) - 1
    G  = np.zeros((Np*Ne, Np*Ne))
    for je in range(Ne):
        G_local = compute_deriv_matrix(xs[je], xs[je+1], Np)
        for n in range(Np):
            for m in range(Np):
                G[je*Np + m, je*Np + n] = G_local[m, n]
    return G

# print(assemble_mass_matrix(lambda x: 1.0, 2, np.array([0.0, 0.5, 1.0])))
# print(assemble_deriv_matrix(2, np.array([0.0, 0.5, 1.0])))

def assemble_face_matrices(Np, xs, for_TSA=False):
    """Assemble matrices M^+ and M^- for face fluxes"""
    Ne = len(xs) - 1
    ξ_b, _  = gausslobatto(Np)
    M_plus  = np.zeros((Np*Ne, Np*Ne))
    M_minus = np.zeros((Np*Ne, Np*Ne))
    pk0 = np.zeros(Np)
    pk1 = np.zeros(Np)
    
    for n in range(Np):
        pk0[n] = eval_pk(-1.0, n, ξ_b)
        pk1[n] = eval_pk(1.0, n, ξ_b)
    
    # Interior faces
    for je in range(1, Ne-1):
        for n in range(Np):
            for m in range(Np):
                M_plus[je*Np + m, je*Np + n] = pk1[m] * pk1[n]
                M_plus[je*Np + m, (je-1)*Np + n] = -pk0[m] * pk1[n]
                M_minus[je*Np + m, (je+1)*Np + n] = pk1[m] * pk0[n]
                M_minus[je*Np + m, je*Np + n] = -pk0[m] * pk0[n]
    
    # Left boundary
    je = 0
    for n in range(Np):
        for m in range(Np):
            M_plus[je*Np + m, je*Np + n] = pk1[m] * pk1[n]
            if for_TSA:
                M_plus[je*Np + m, je*Np + n] -= pk0[m] * pk0[n]
            M_minus[je*Np + m, (je+1)*Np + n] = pk1[m] * pk0[n]
            M_minus[je*Np + m, je*Np + n] = -pk0[m] * pk0[n]
    
    # Right boundary
    je = Ne-1
    for n in range(Np):
        for m in range(Np):
            M_plus[je*Np + m, je*Np + n] = pk1[m] * pk1[n]
            M_plus[je*Np + m, (je-1)*Np + n] = -pk0[m] * pk1[n]
            M_minus[je*Np + m, je*Np + n] = -pk0[m] * pk0[n]
            if for_TSA:
                M_minus[je*Np + m, je*Np + n] += pk1[m] * pk1[n]
    
    return M_plus, M_minus

# print(assemble_face_matrices(2, np.array([0.0, 0.5, 1.0])))

def compute_inflow_term_plus(inflow, Np, xs):
    """Compute inflow term for the left boundary (x=0)"""
    Ne = len(xs) - 1
    ξ_b, _ = gausslobatto(Np)
    qs = np.zeros(Ne*Np)
    je = 0
    for m in range(Np):
        qs[je*Np + m] = inflow(xs[0]) * eval_pk(-1.0, m, ξ_b)
    return qs

def compute_inflow_term_minus(inflow, Np, xs):
    """Compute inflow term for the right boundary (x=1)"""
    Ne = len(xs) - 1
    ξ_b, _ = gausslobatto(Np)
    qs = np.zeros(Ne*Np)
    je = Ne-1
    for m in range(Np):
        qs[je*Np + m] = inflow(xs[-1]) * eval_pk(1.0, m, ξ_b)
    return qs

# print(compute_inflow_term_plus(lambda x: 1.0, 2, np.array([0.0, 0.5, 1.0])))
# print(compute_inflow_term_minus(lambda x: 1.0, 2, np.array([0.0, 0.5, 1.0])))

def compute_source_term(source, Np, xs):
    """
    Compute source term for the given source func on the mesh defined by xs.
    The source function should accept a single argument (x) and return a scalar value.
    """
    Ne = len(xs) - 1
    ξ_b, _   = gausslobatto(Np)    # interpolation nodes for Legendre basis ("b") funcs v_m
    ξ_q, w_q = gausslegendre(3*Np) # quadrature nodes, ξ_q in [-1, 1]
    qs = np.zeros(Ne * Np)
    for je in range(Ne):
        a = xs[je]
        b = xs[je + 1]
        for m in range(Np):
            val = 0.0
            for k in range(len(ξ_q)):
                x = ξ_q[k]
                # Map from reference [-1,1] to [a,b]
                y = b * (x + 1) / 2 + a * (1 - x) / 2
                val += w_q[k] * eval_pk(x, m, ξ_b) * source(y)
            qs[je * Np + m] = (b - a) / 2.0 * val
    return qs

# print(compute_source_term(lambda x: 1.0, 2, np.array([0.0, 0.5, 1.0])))

def transport_direct_solve(μ:float, σ_t, source, inflow, Np, xs):
    """
    Solve the transport eq using a DG + collocation (discrete ordinates) method.
        μ      : Transport coefficient (+ for rightward transport, - for leftward)
        σ_t    : Total scattering opacity func (can be a const or a func of x)
        source : Source term func (can be a const or a func of x)
        inflow : Inflow term func (can be a const or a func of x)
        Np     : Number of Legendre basis funcs per element
        xs     : DG mesh points defining the domain [x0, ..., xn]
    Returns:
        ψ_weights : Solution vector containing the weights of the polynomial basis funcs
    """
    Ne     = len(xs) - 1

    # Assemble matrices
    M   = assemble_mass_matrix(σ_t, Np, xs)
    G   = assemble_deriv_matrix(Np, xs)
    F_plus, F_minus = assemble_face_matrices(Np, xs)
    print(np.shape(M), np.shape(G), np.shape(F_plus), np.shape(F_minus))

    # Compute A and inflow depending on the sign of μ
    if μ>=0:
        A = -μ * G + μ * F_plus + M
        qs_inflow = compute_inflow_term_plus(inflow, Np, xs)
    else:
        A = -μ * G + μ * F_minus + M
        qs_inflow = compute_inflow_term_minus(inflow, Np, xs)
    
    # Compute source term
    qs = compute_source_term(source, Np, xs) + np.abs(μ)*qs_inflow
    ψ_weights = np.linalg.solve(A, qs)
    
    return ψ_weights.reshape((Ne,Np))

def transport_direct_solve_diffusive(σ_t, σ_a, ε, source, inflow, Np, Nμ, Nt, xs):
    """
    Solve the transport eq using a fixed point iteration method.
        σ_t    : Total scattering opacity func (can be a const or a func of x)
        σ_a    : Absorption scattering opacity func (can be a const or a func of x)
        ε      : Scattering parameter
        # ψ_0    : Initial guess for the solution (can be a const or a func of x,k)
        source : Source term func (can be a const or a func of x and μ)
        inflow : Inflow term func (can be a const or a func of x)
        Np     : Number of Legendre basis funcs per element
        Nμ     : Number of polynomial degrees in μ direction (number of Gauss-Legendre points)
        Nt     : Number of time steps for fix-point iteration
        xs     : DG mesh points defining the domain [x0, ..., xn]
    Returns:
        ψ_weights : Solution vector containing the weights of the polynomial basis funcs
        μs        : Array of μ values used in the solution
    """
    Ne     = len(xs) - 1

    # Define scattering opacity σ_s
    if not callable(σ_t): σ_t = lambda x: σ_t
    if not callable(σ_a): σ_a = lambda x: σ_a
    σ_s = lambda x: σ_t(x)/ε - ε*σ_a(x)

    # Assemble matrices
    M_t = assemble_mass_matrix(lambda x: σ_t(x)/ε, Np, xs)
    M_s = assemble_mass_matrix(σ_s, Np, xs) # shape (Ne*Np, Ne*Np)
    G   = assemble_deriv_matrix(Np, xs)
    F_plus, F_minus = assemble_face_matrices(Np, xs)
    
    ξ_μ, w_μ = gausslegendre(Nμ)
    μs       = ξ_μ # because μ and ξ both in [-1,1]
    
    ψ_weights_all = np.zeros((Nμ, Ne*Np)) # For each μ, for each element, we store the weight vector

    for t in range(Nt):
        # Compute integral from -1 to 1 of ψμ by quadrature
        φ = (w_μ.reshape((-1, 1)) * ψ_weights_all).sum(axis=0) # shape (Ne*Np)
        Msφ = 1/2 * M_s @ φ

        for i_μ, μ in enumerate(μs):
            # Compute A and inflow depending on the sign of μ
            if μ>=0:
                A = -μ * G + μ * F_plus + M_t
                qs_inflow = compute_inflow_term_plus(lambda x: inflow(x,μ), Np, xs)
            else:
                A = -μ * G + μ * F_minus + M_t
                qs_inflow = compute_inflow_term_minus(lambda x: inflow(x,μ), Np, xs)
        
            # Compute RHS
            qs = ε * (compute_source_term(lambda x: source(x, μ), Np, xs) + np.abs(μ)*qs_inflow)
            b  = Msφ + qs
            ψ_weights_all[i_μ] = np.linalg.solve(A, b)
    
    return ψ_weights_all.reshape((Nμ,Ne,Np)), μs

# print(transport_direct_solve( 1.0, lambda x: 1.0, lambda x: 1.0, lambda x: 1.0, 2, np.array([0.0, 0.5, 1.0])))
# print(transport_direct_solve(-1.0, lambda x: 1.0, lambda x: 1.0, lambda x: 1.0, 2, np.array([0.0, 0.5, 1.0])))

def error_Lp(ψ_weights, xs, Np, exact_ψ_func, p=2):
    """Compute the L2 error of the DG solution against an exact solution."""
    Ne = len(xs) - 1
    ξ_b, _   = gausslobatto(Np)    # interpolation nodes for Legendre basis funcs v_m
    ξ_q, w_q = gausslegendre(3*Np) # quadrature nodes, ξ_q in [-1, 1]
    
    error = 0.0
    for je in range(Ne):
        a, b = xs[je], xs[je+1]
        x_q  = ξ_to_x(ξ_q, a, b)   # mapped quad nodes, x_q in [a, b]
        
        ψ_vals = np.zeros_like(ξ_q)
        for n in range(Np):
            ψ_vals += ψ_weights[je,n] * eval_pk(ξ_q, n, ξ_b)
        
        exact_vals = exact_ψ_func(x_q)
        diff       = ψ_vals - exact_vals
        if p == 'inf':
            error  = max(error, np.max(np.abs(diff)))
        else:
            error += np.sum(w_q * (ψ_vals - exact_vals)**p) * (b-a)/2.0
    return error if p=='inf' else error**(1/p)

def plot_solution(ψ_weights, xs, Np, μ=None, num_plot_pts=200, exact_ψ_func=None, save_plot=False):
    """
    Reconstructs and plots the DG solution ψ(x) over the mesh xs.

    Plot optional:  saved to "test_figures/transport_solution.png" if save_plot=True.
    """
    Ne     = len(xs) - 1
    ξ_b, _ = gausslobatto(Np)                 # interpolation nodes for Legendre basis ("b") funcs v_m
    ξ_p    = np.linspace(-1, 1, num_plot_pts) # plot points (in reference space [-1,1]) for each element

    plt.figure()
    for je in range(Ne):
        a, b = xs[je], xs[je+1]
        x_p  = ξ_to_x(ξ_p, a, b)
        
        # Reconstruct polynomial on this element
        ψ_weights_loc = ψ_weights[je,:]
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
        file_name = os.path.join(current_directory, "test_figures/transport_plot.png")
        plt.savefig(file_name)
        print(f"Plot saved as {file_name}")
    else:
        plt.show()

