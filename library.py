import numpy as np
from numpy.polynomial.legendre import Legendre
import matplotlib.pyplot as plt

def compute_mass_matrix(sigma_t, x_L, x_R, Np):
      Nq = 3 * Np
      mus, ws = gausslobatto(Np)
      ir_mus, ir_ws = gausslobatto(Nq)
      M_local = np.zeros((Np, Np))
      for m in range(Np):
          for n in range(Np):
            for pt in range(Nq):
                x_pt = (x_R - x_L) * ir_mus[pt] / 2 + (x_R + x_L) / 2
                M_local[m, n] += sigma_t(x_pt) * ir_ws[pt] * eval_pk(ir_mus[pt], m, mus) * eval_pk(ir_mus[pt], n, mus)

      jacobi = (x_R - x_L) / 2.0
      M_local *= jacobi
      return M_local

def compute_deriv_matrix(x_L, x_R, Np, Nq = None):
      Nq = 3 * Np
      mus, ws = gausslobatto(Np)
      ir_mus, ir_ws = gausslobatto(Nq)
      M_local = np.zeros((Np, Np))
      for m in range(Np):
          for n in range(Np):
            for pt in range(Nq):              
                M_local[m, n] += ir_ws[pt] * eval_pk_deriv(ir_mus[pt], m, mus) * eval_pk(ir_mus[pt], n, mus)
      return M_local


def gausslegendre(N):
    points, weights = np.polynomial.legendre.leggauss(N)
    return points, weights

def gausslobatto(N):
    x = np.zeros(N)
    x[0] = -1.0
    x[-1] = 1.0

    Pn_1 = Legendre.basis(N-1)
    dPn_1 = Pn_1.deriv()
    x[1:-1] = np.sort(dPn_1.roots())

    # Compute weights
    w = np.zeros(N)
    for i in range(N):
        xi = x[i]
        Pn_1_val = Pn_1(xi)
        w[i] = 2.0 / (N * (N-1) * (Pn_1_val**2))
    return x, w

def eval_pk(x, i, nodes):
    # Compute the i-th Lagrange basis polynomial at x
    xi = nodes[i]
    terms = [(x - nodes[j])/(xi - nodes[j]) for j in range(len(nodes)) if j != i]
    return np.prod(terms, axis=0)

def eval_pk_deriv(x, i, nodes):
    n = len(nodes)
    xi = nodes[i]
    result = 0.0
    for j in range(n):
        if j == i:
            continue
        xj = nodes[j]
        term = 1.0 / (xi - xj)
        for k in range(n):
            if k == i or k == j:
                continue
            term *= (x - nodes[k]) / (xi - nodes[k])
        result += term
    return result
  
def assemble_mass_matrix(sigma_t, Np, xs):
    Ne = len(xs) - 1
    M = np.zeros((Np*Ne, Np*Ne))
    for je in range(Ne):
        M_local = compute_mass_matrix(sigma_t, xs[je], xs[je+1], Np)
        for n in range(Np):
            for m in range(Np):
                M[je*Np + m, je*Np + n] = M_local[m, n]
    return M

def assemble_deriv_matrix(Np, xs):
    Ne = len(xs) - 1
    M = np.zeros((Np*Ne, Np*Ne))
    for je in range(Ne):
        M_local = compute_deriv_matrix(xs[je], xs[je+1], Np)
        for n in range(Np):
            for m in range(Np):
                M[je*Np + m, je*Np + n] = M_local[m, n]
    return M

def assemble_face_matrices(Np, xs, for_TSA=False):
    Ne = len(xs) - 1
    mus, ws = gausslobatto(Np)
    M_plus = np.zeros((Np*Ne, Np*Ne))
    M_minus = np.zeros((Np*Ne, Np*Ne))
    pk0 = np.zeros(Np)
    pk1 = np.zeros(Np)
    
    for n in range(Np):
        pk0[n] = eval_pk(-1.0, n, mus)
        pk1[n] = eval_pk(1.0, n, mus)
    
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
    Ne = len(xs) - 1
    mus, ws = gausslobatto(Np)
    qs = np.zeros(Ne*Np)
    je = 0
    for m in range(Np):
        qs[je*Np + m] = inflow(xs[0]) * eval_pk(-1.0, m, mus)
    return qs

def compute_inflow_term_minus(inflow, Np, xs):
    Ne = len(xs) - 1
    mus, ws = gausslobatto(Np)
    qs = np.zeros(Ne*Np)
    je = Ne-1
    for m in range(Np):
        qs[je*Np + m] = inflow(xs[-1]) * eval_pk(1.0, m, mus)
    return qs

def compute_source_term(source, Np, xs):
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

def transport_direct_solve_plus(mu, sigma_t, qs, inflow, Np, xs):
    Ne = len(xs) - 1
    mus, ws = gausslobatto(Np)
    qs_inflow = compute_inflow_term_plus(inflow, Np, xs)
    F_plus, F_minus = assemble_face_matrices(Np, xs)
    G = assemble_deriv_matrix(Np, xs)
    M = assemble_mass_matrix(sigma_t, Np, xs)
    A = -mu * G + mu * F_plus + M
    qs += mu * qs_inflow
    psi = np.linalg.solve(A, qs)
    return psi

def transport_direct_solve_minus(mu, sigma_t, qs, inflow, Np, xs):
    Ne = len(xs) - 1
    mus, ws = gausslobatto(Np)
    qs_inflow = compute_inflow_term_minus(inflow, Np, xs)
    F_plus, F_minus = assemble_face_matrices(Np, xs)
    G = assemble_deriv_matrix(Np, xs)
    M = assemble_mass_matrix(sigma_t, Np, xs)
    A = -mu * G + mu * F_minus + M
    qs -= mu * qs_inflow
    psi = np.linalg.solve(A, qs)
    return psi

def transport_direct_solve(mu, sigma_t, qs, inflow, Np, xs):
    if mu > 0:
        psi = transport_direct_solve_plus(mu, sigma_t, qs, inflow, Np, xs)
    else:
        psi = transport_direct_solve_minus(abs(mu), sigma_t, qs, inflow, Np, xs)
    return psi

