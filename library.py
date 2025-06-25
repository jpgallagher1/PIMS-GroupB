import numpy as np
import matplotlib.pyplot as plt
from   mpl_toolkits.mplot3d import Axes3D
import tqdm
import os

def ξ_to_x(ξ, a, b):
    """Map the points ξ in [-1, 1] (reference element) to x in [a, b]."""
    return 0.5 * (b - a) * (ξ + 1.0) + a

def preprocess_exact_sol(exact_sol): 
    assert callable(exact_sol), "exact_sol must be a callable function or None"
    try:
        exact_sol(0,0)
        return exact_sol
    except:
        exact_sol_xμ = lambda x, μ: exact_sol(x)  # Ensure exact_sol is callable with (x, μ)
        return  exact_sol_xμ

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

def compute_mass_matrix(σ_t, a, b, Np):
    """
    Full (consistent) mass matrix on [a,b]:
      Me[m,n] = (b-a)/2 * sum_k{ w_k * σ_t(x_k) * l_m(ξ_k) * l_n(ξ_k) }
    """
    ξ_b, _   = gausslobatto(Np)    # interpolation nodes for Legendre basis funcs v_m
    ξ_q, w_q = gausslobatto(3*Np) # quadrature nodes,  ξ_q in [-1, 1]
    x_q      = ξ_to_x(ξ_q, a, b)   # mapped quad nodes, x_q in [a, b]
    Me = np.zeros((Np, Np)) # Mass matrix local to element e
    for m in range(Np):
        for n in range(Np):
            val = 0.0
            for k in range(3*Np):
                lm   = eval_pk(ξ_q[k], m, ξ_b)
                ln   = eval_pk(ξ_q[k], n, ξ_b)
                val += w_q[k] * (σ_t(x_q[k]) if callable(σ_t) else σ_t) * lm * ln
            Me[m, n] = (b-a)/2 * val
    return Me

def compute_deriv_matrix(a, b, Np):
    """
    Full (consistent) derivative matrix on [a,b]:
        Ge[m,n] = sum_k{ w_k * d/dξ l_m(ξ_k) * l_n(ξ_k) }
    """
    ξ_b, _   = gausslobatto(Np)    # interpolation nodes for Legendre basis ("b") funcs v_m
    ξ_q, w_q = gausslobatto(3*Np) # quadrature nodes, ξ_q in [-1, 1]

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

def error_Lp(ψ_weights_all, xs, Np, exact_sol, μ_single=None, p=2):
    """
    Compute the L2 error of the DG solution against an exact solution.
    Interally uses ψ_weights_all, the ψ_weights for each μ,
    and uses exact_sol a function of x and μ.
    """
    Ne = len(xs) - 1
    ξ_b, _   = gausslobatto(Np)    # interpolation nodes for Legendre basis funcs v_m
    ξ_q, w_q = gausslegendre(3*Np) # quadrature nodes, ξ_q in [-1, 1]

    # Preprocess inputs
    if exact_sol is not None: exact_sol = preprocess_exact_sol(exact_sol)
    if len(ψ_weights_all.shape) == 1:
        ψ_weights_all = ψ_weights_all.reshape((1,-1))
        assert μ_single is not None, "For solution at single μ, μ_single must be provided."
        μs = [μ_single]
    else: μs,_ = gausslegendre(ψ_weights_all.shape[0])

    error = 0.0
    for iμ,μ in enumerate(μs):
        for je in range(Ne):
            a, b = xs[je], xs[je+1]
            x_q  = ξ_to_x(ξ_q, a, b)   # mapped quad nodes, x_q in [a, b]
            
            ψ_vals = np.zeros_like(ξ_q)
            for n in range(Np):
                ψ_vals += ψ_weights_all[iμ,je*Np+n] * eval_pk(ξ_q, n, ξ_b)
            
            exact_vals = exact_sol(x_q,μ)
            diff       = ψ_vals - exact_vals
            if p == 'inf':
                error  = max(error, np.max(np.abs(diff)))
            else:
                error += np.sum(w_q * (ψ_vals - exact_vals)**p) * (b-a)/2.0
    return error if p=='inf' else error**(1/p)

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
    Ne = len(xs) - 1

    # Assemble matrices
    M  = assemble_mass_matrix(σ_t, Np, xs)
    G  = assemble_deriv_matrix(Np, xs)
    F_plus, F_minus = assemble_face_matrices(Np, xs)

    # Compute A and inflow depending on the sign of μ
    if μ>0:
        A = -μ * G + μ * F_plus + M
        qs_inflow = compute_inflow_term_plus(inflow, Np, xs)
    else:
        A = -μ * G + μ * F_minus + M
        qs_inflow = compute_inflow_term_minus(inflow, Np, xs)
    
    # Compute source term
    qs = compute_source_term(source, Np, xs) + np.abs(μ)*qs_inflow
    ψ_weights = np.linalg.solve(A, qs)
    
    return ψ_weights

def transport_direct_solve_diffusive(σ_t, σ_a, ε, source, inflow, Np, Nμ, Nt, xs,
                                     max_iter=1000, exact_sol=None, tol=None, tolnorm=2):
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

        max_iter : Maximum number of iterations for convergence
        exact_sol: Optional exact solution function for validation (can be None)
        tol      : Tolerance for convergence. If exact_sol not specified, it uses self-convergence
        tolnorm  : Norm to use for convergence check (1,2,... or 'inf')
    Returns:
        ψ_weights_all : Solution vector containing the weights of the polynomial basis funcs
        μs            : Array of μ values used in the solution
    """
    # Preprocess inputs
    Ne = len(xs) - 1
    if tolnorm == 'inf': tolnorm = np.inf
    if exact_sol is not None: exact_sol = preprocess_exact_sol(exact_sol)
    
    # Define scattering opacity σ_s
    if not callable(σ_t): σ_t = lambda x: σ_t
    if not callable(σ_a): σ_a = lambda x: σ_a
    σ_s = lambda x: σ_t(x)/ε - ε*σ_a(x)

    # Assemble matrices
    M_t = assemble_mass_matrix(lambda x: σ_t(x)/ε, Np, xs)
    M_s = assemble_mass_matrix(σ_s, Np, xs) # shape (Ne*Np, Ne*Np)
    G   = assemble_deriv_matrix(Np, xs)
    F_plus, F_minus = assemble_face_matrices(Np, xs)
    
    μs, w_μ = gausslegendre(Nμ)
    
    ψ_weights_all = np.zeros((Nμ, Ne*Np)) # For each μ, for each element, we store the weight vector

    for t in tqdm.tqdm(range(Nt)):
        ψ_weights_all_old = ψ_weights_all.copy() # Store old weights for convergence check

        # Compute integral from -1 to 1 of ψ by quadrature
        φ = (w_μ.reshape((-1, 1)) * ψ_weights_all).sum(axis=0) # shape (Ne*Np)
        Msφ = 1/2 * M_s @ φ

        for i_μ, μ in enumerate(μs):
            # Compute A and inflow depending on the sign of μ
            if μ>0:
                A = -μ * G + μ * F_plus + M_t
                qs_inflow = compute_inflow_term_plus(lambda x: inflow(x,μ), Np, xs)
            else:
                A = -μ * G + μ * F_minus + M_t
                qs_inflow = compute_inflow_term_minus(lambda x: inflow(x,μ), Np, xs)
            # Compute RHS
            qs = ε*(compute_source_term(lambda x: source(x, μ), Np, xs)) + np.abs(μ)*qs_inflow
            b  = Msφ + qs
            # Solve
            ψ_weights_all[i_μ] = np.linalg.solve(A, b)
        
        if tol is not None:
            if exact_sol is not None: # If exact solution is provided, use it for convergence check
                error = error_Lp(ψ_weights_all, xs, Np, exact_sol, p=tolnorm)
            else: # Self-convergence check
                error = np.linalg.norm(ψ_weights_all - ψ_weights_all_old, ord=tolnorm)
            if error < tol:
                print(f"Converged after {t+1} iterations with tolerance {tol}.")
                break
    
    return ψ_weights_all, μs

def plot_solution(ψ_weights_all, xs, Np, plot_3D=False, num_plot_pts=200, μ_single=None, exact_sol=None, save_plot=False):
    """
    Reconstructs and plots the DG solution ψ(x) over the mesh xs.
    Plot optional:  saved to "test_figures/transport_solution.png" if save_plot=True.
    """
    # Preprocess inputs
    if exact_sol is not None: exact_sol = preprocess_exact_sol(exact_sol)
    if len(ψ_weights_all.shape) == 1:
        ψ_weights_all = ψ_weights_all.reshape((1,-1))
        assert μ_single is not None, "For solution at single μ, μ_single must be provided."

    Ne     = len(xs) - 1
    ξ_b, _ = gausslobatto(Np)                 # interpolation nodes for Legendre basis ("b") funcs v_m
    ξ_p    = np.linspace(-1, 1, num_plot_pts) # plot points (in reference space [-1,1]) for each element
    if μ_single is not None: μs = [μ_single]
    else: μs,_ = gausslegendre(ψ_weights_all.shape[0])

    # Create figure
    fig = plt.figure(figsize=(6, 5))      
    if plot_3D:
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('μ')
        ax.set_zlabel(r'$\psi(x,\mu)$')
        ax.grid(False)
        ax.set_xlim(xs[0], xs[-1])
        ax.set_ylim(-1, 1)
        ax.set_zlim(0, 3.65)
        ax.set_xticks([0, 1])
        ax.set_yticks([-1, 0, 1])
    else:
        ax = fig.add_subplot(111)
        ax.set_xlabel('x')
        ax.set_ylabel(r'$\psi(x,\mu)$')

    for iμ,μ in enumerate(μs):
        maxψ,minψ = 0, 0
        for je in range(Ne):
            a, b = xs[je], xs[je+1]
            numx = ξ_to_x(ξ_p, a, b)
            ψ_weights_loc = ψ_weights_all[iμ,je*Np:(je+1)*Np]
            numψ = np.zeros_like(ξ_p)
            for n in range(Np):
                numψ += ψ_weights_loc[n] * eval_pk(ξ_p, n, ξ_b)
            maxψ,minψ = max(maxψ, np.max(numψ)), min(minψ, np.min(numψ))
        if plot_3D:
            shadowx = np.linspace(xs[0], xs[-1], num_plot_pts)
            shadowψ = exact_sol(shadowx,μ)
            ax.plot(shadowx, np.full_like(shadowx, μ), np.zeros_like(shadowx), '-', color='darkgray', lw=1.2)
            ax.plot(np.full_like(shadowx, 0), np.full_like(shadowx, μ), np.linspace(minψ,maxψ,len(shadowx)), '-', color='darkgray', lw=1.2)

    clrs = ['cornflowerblue', 'darkorange', 'forestgreen', 'crimson', 'purple']
    for iμ,μ in reversed(list(enumerate(μs))):
        maxψ,minψ = 0, 0
        for je in range(Ne):
            a, b = xs[je], xs[je+1]
            numx  = ξ_to_x(ξ_p, a, b)
            
            # Reconstruct polynomial on this element
            ψ_weights_loc = ψ_weights_all[iμ,je*Np:(je+1)*Np]
            numψ = np.zeros_like(ξ_p)
            for n in range(Np):
                numψ += ψ_weights_loc[n] * eval_pk(ξ_p, n, ξ_b)
            maxψ,minψ = max(maxψ, np.max(numψ)), min(minψ, np.min(numψ))
            if plot_3D:
                ax.plot(numx, np.full_like(numx, μ), numψ, '-', color=clrs[iμ%len(clrs)], lw=2.5)
            else:
                ax.plot(numx, numψ, '-', color=clrs[iμ%len(clrs)], lw=2.5)

        if exact_sol is not None:
            exactx = np.linspace(xs[0], xs[-1], num_plot_pts)
            exactψ = exact_sol(exactx,μ)
            if plot_3D:
                ax.plot(exactx, np.full_like(exactx, μ), exactψ, '--', color='black', lw=1.25)
                # ax.plot(exactx, np.full_like(exactx, μ), np.zeros_like(exactx), '-', color='darkgray', lw=1.2)
            else:
                ax.plot(exactx, exactψ, '--', color='black', lw=1.25)
            plt.legend()

    if μs is not None:
        if len(μs) == 1: plt.title(f'DG solution, μ={μs}, Np={Np}, Ne={Ne}')
        else: plt.title(f'DG solution, Np={Np}, Ne={Ne}')
    
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