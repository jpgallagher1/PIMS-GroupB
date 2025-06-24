def compute_mass_matrix(sigma_t, x_L, x_R, Np, Nq=None):
    if Nq is None:
        Nq = Np
    mus, ws = gausslobatto(Np)
    ir_mus, ir_ws = gausslobatto(Nq)
    M_local = np.zeros((Np, Np))
    for m in range(Np):
        for n in range(Np):
            for pt in range(Nq):
                # Map reference point to physical point
                x_pt = 0.5 * (x_R - x_L) * ir_mus[pt] + 0.5 * (x_R + x_L)
                M_local[m, n] += sigma_t(x_pt) * ir_ws[pt] * eval_pk(ir_mus[pt], m, mus) * eval_pk(ir_mus[pt], n, mus)
    
    # Account for Jacobian of the transformation
    M_local *= 0.5 * (x_R - x_L)
    return M_local
