# PIMS-GroupB
PIMS Hackathon Group B

June 23-25, 2025

## Main Files

```
library.py         Main library with functions for simulation
main.ipynb         Main executable calling functions from library.py
test_cases.py      Test file for checking if updated functions are working 


/archive/          Old Code 
/test_figures/     Plots from test_cases
```

## Equations for Analyzing Spectral Properties of $(I-T)^{-1}$

### Spectrum of $(I-T)^{-1}$

$(I-T)^{-1} = \epsilon \alpha F_{||} - \epsilon^2 [ \frac{1}{3} M_{t}^{-1} (G - \frac{1}{2}F) M_{t}^{-1} (G - \frac{1}{2}F) - M_{t}^{-1}M_{a} ] + \mathcal{O}(\epsilon^3)$

$D = \alpha \epsilon^{-1} F_{||} - [ \frac{1}{3} M_{t}^{-1} (G - \frac{1}{2}F) M_{t}^{-1} (G - \frac{1}{2}F) - M_{t}^{-1}M_{a} ]$

$F^+,F^- \leftarrow$ assemble_face_matrices

$F = F^+ + F^-$

$F_{||} = F^+ - F^-$

$G \leftarrow$ assemble_deriv_matrix

$M_{t/a,mn} = \int_{x_e}^{x_{e+1}} \sigma_{t/a} v_{e,n} v_{e,m} dx$

$\alpha = \frac{1}{2}\sum_{k} \omega_k |\mu_k|$

$\epsilon \leftarrow$ from user as a hyperparameter

## Hackathon plan

Physics-informed preconditioners for a simplified
model of photon transport

### Day 1
1. Read Section 3.1, which covers
– A highly simplified model photon transport equation.
– Discretization of this equation in angle and space.
2. Implement DG discretization of transport operator
– Use the pseudocode in Section 5.1 as a reference Python implemen-
tation
3. Implement unaccelerated fixed point iteration from Section 4.2
### Day 2
1. Verify solution accuracy of DG solver using Method of Manufactured So-
lutions (MMS)
– Use different polynomial orders
– Test accuracy vs spatial resolution and local polynomial order
2. Read Section 4 on the diffusion limit
3. Explore how diffusion scaling ε effects fixed-point iteration count
– How does number of iterations scale as a function of ε?
4. Read Section 4.4 on an “asymptotically consistent” diffusion precondi-
tioner
– look at spectrum of transport operator $(I−T)^{−1}$, where $T$ is defined
in equations (31), (32), and (33)
### Day 3
1. Implement consistent diffusion preconditioner
– How does number of iterations scale as a function of $\epsilon$ for the pre-
conditioned solver?
– How does number of iterations scale as a function of $\epsilon$ for the un-
preconditioned solver?
2. Stretch goal:
– Go through theory in Section 4.6
– Numerically look at the spectrum of the un-preconditioned transport
operator $I−T$
3
– Verify the the preconditioned operator satisfies
$$I−\epsilon^{−2}D^{−1} (I−T)^{−1}
= I+ O(\epsilon),$$
where the diffusion matrix D defined in equation (35).
– Show that the fixed point iteration scheme can be cast in terms of $(I - \epsilon^{-2}D^{-1})(I - T)^{-1}$.
