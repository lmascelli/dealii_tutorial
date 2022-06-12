/*
 * THE SYSTEM
 *
 * du/dt - v = 0
 * dv/dt - lamda(u) = f
 * u(x, t) = g on dSigma
 * u(x, 0) = u0(x) on Sigma
 * v(x, 0) = u1(x) on Sigma
 *
 * TIME DISCRETIZATION
 *
 * k = t;n - t;n-1
 *
 * (u;n - u;n-1)/k - [sigma * v;n + (1-sigma) * v;n-1] = 0
 * (v;n - v;n-1)/k - [sigma * lambda(u);n + (1-sigma) * lambda(u);n-1] = sigma *
 *                                                    f;n + (1 - sigma) * f;n-1
 *
 * with sigma = 0.5 --> Crank-Nicolson method
 *
 * v;n = v;n-1 + k(sigma * (f;n + lambda(u);n) + (1-sigma) * (lambda(u);n-1 +
 *                                                            f;n-1))
 *
 * (u;n - u;n-1)/k = 
 *      sigma * v;n-1 + 
 *      sigma * k * (sigma* (f;n + lambda(u);n)) +
 *      sigma * k * (1 - sigma) * lambda(u); n-1 +
 *      (1-sigma) * v;n-1
 *
 * 
 */
