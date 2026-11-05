from typing import Optional

import numpy as np
from scipy.optimize import root
from scipy.optimize import least_squares

def compute_bruggeman_n_eff(n_incl: np.ndarray,
                            n_host: np.ndarray,
                            m_incl: float,
                            m_host: float,
                            rho_incl: float,
                            rho_host: float,
                            vol_frac: Optional[float] = None) -> np.ndarray:
    """
    Compute effective complex refractive index via Bruggeman:
      - Convert mass fractions -> volume fractions using densities.
      - Build epsilons (n^2) per component and solve Bruggeman for epsilon_eff.
      - Return n_eff = sqrt(epsilon_eff).

    n_incl, n_host: 1D arrays over frequency (must have same length)
    m_incl, m_host, rho_incl, rho_host: scalars (mass fractions and densities)
    vol_frac: optional explicit volume fraction for inclusion (overrides mass->vol conversion)
    """
    # ensure arrays
    n_incl = np.asarray(n_incl, dtype=np.complex128)
    n_host = np.asarray(n_host, dtype=np.complex128)
    if n_incl.shape != n_host.shape:
        raise ValueError("n_incl and n_host must have the same shape")

    # mass -> volume fraction (scalar) unless vol_frac provided
    v_incl = m_incl / rho_incl / (m_incl / rho_incl + m_host / rho_host)
    if vol_frac is not None:
        v_incl = vol_frac
    v_host = 1.0 - v_incl

    # epsilons per component (shape (2, n_pts))
    eps_incl = n_incl ** 2
    eps_host = n_host ** 2
    epsilons = np.vstack([eps_incl, eps_host])  # shape (2, n_pts)

    # volume fraction array (shape (2,) will be broadcast to (2, n_pts) inside bruggeman_eps_eff)
    v_arr = np.array([v_incl, v_host])

    # solve Bruggeman for effective epsilon
    epsilon_eff = bruggeman_eps_eff(epsilons, v_arr)

    # return effective complex refractive index
    return np.sqrt(epsilon_eff)

def emt(n_m, n_eff, m_i, m_h, rho_i, rho_h, vol_frac=None, method="bruggeman"):
    epsilon_h = n_m ** 2
    epsilon_eff = n_eff ** 2

    v_i = m_i / rho_i / (m_i / rho_i + m_h / rho_h)
    v_h = m_h / rho_h / (m_i / rho_i + m_h / rho_h)

    if vol_frac:
        print(f"v_i = {v_i}, vol_frac = {vol_frac}")
        v_i = vol_frac
        v_h = 1 - v_i

    if method == "bruggeman":
        denom = epsilon_h + epsilon_eff * (3 * v_i - 1)
        epsilon_incl = (epsilon_h ** 2 * v_i + 2 * epsilon_eff * epsilon_h * (2 * v_i - 1) - 2 * epsilon_eff ** 2 * (
                v_i - 1)) / denom
    elif method == "maxwell-garnett":
        denom = 2 * epsilon_h * v_i + epsilon_h + epsilon_eff * (v_i - 1)
        epsilon_incl = epsilon_h * (2 * epsilon_h * (v_i - 1) + epsilon_eff * (v_i + 2)) / denom
    elif method == "LLL":
        epsilon_incl = (1 / v_i * (epsilon_eff ** (1 / 3) - v_h * epsilon_h ** (1 / 3))) ** 3
    else:
        raise ValueError(f"Unknown EMT method: {method}")

    return np.sqrt(epsilon_incl)


def bruggeman_eps_eff(epsilons: np.ndarray,
                            v: Optional[np.ndarray] = None,
                            tol: float = 1e-10,
                            maxiter: int = 200):
    """
    Solve Bruggeman equation for multiple components:
      F(x) = sum_i v_i * (eps_i - x) / (eps_i + 2 x) = 0
    epsilons: shape (n_comp,) or (n_comp, n_pts)
    v: volume fractions, shape (n_comp,) or (n_comp, n_pts). If None equal fractions are used.
    Returns: epsilon_eff as scalar if single point, or array shape (n_pts,)
    """
    eps = np.asarray(epsilons)
    if eps.ndim == 1:
        eps = eps[:, None]               # (n_comp, 1)
    n_comp, n_pts = eps.shape

    if v is None:
        v_arr = np.full((n_comp, n_pts), 1.0 / n_comp, dtype=np.complex128)
    else:
        v_arr = np.asarray(v)
        if v_arr.ndim == 1:
            v_arr = v_arr[:, None]
        if v_arr.shape[0] != n_comp:
            raise ValueError("First dimension of `v` must match number of components in `epsilons`")
        # broadcast if needed
        if v_arr.shape[1] != n_pts:
            v_arr = np.broadcast_to(v_arr, (n_comp, n_pts))

    # normalize volume fractions per column
    v_sum = np.sum(v_arr, axis=0)
    v_sum = np.where(v_sum == 0, 1.0, v_sum)
    v_arr = v_arr / v_sum

    results = np.zeros(n_pts, dtype=np.complex128)

    for j in range(n_pts):
        eps_col = eps[:, j].astype(np.complex128)
        v_col = v_arr[:, j].astype(np.complex128)
        # initial guess: volume-weighted mean
        x = np.sum(v_col * eps_col)
        # ensure nonzero starting value
        if x == 0:
            x = np.mean(eps_col) + 1e-6

        for _ in range(maxiter):
            denom = eps_col + 2.0 * x
            # avoid division by zero by small shift
            denom = np.where(denom == 0, denom + 1e-20, denom)
            F = np.sum(v_col * (eps_col - x) / denom)
            # derivative: dF/dx = -3 * sum_i v_i * eps_i / (eps_i + 2x)^2
            dF = -3.0 * np.sum(v_col * eps_col / (denom * denom))
            if dF == 0:
                # small perturbation to escape flat derivative
                x = x + 1e-8
                dF = -3.0 * np.sum(v_col * eps_col / ( (eps_col + 2.0*x)**2 ))
            dx = F / dF
            # damping for stability
            if np.abs(dx) > 1.0 * np.abs(x):
                dx = dx * (1.0 * np.abs(x) / np.abs(dx))
            x_new = x - dx
            if np.allclose(x_new, x, atol=tol, rtol=0):
                x = x_new
                break
            x = x_new
        else:
            raise ValueError(f"Bruggeman solver failed to converge for column {j}")

        results[j] = x

    return results[0] if n_pts == 1 else results


def solve_v_i(epsilons: np.ndarray, epsilon_eff: complex, tol: float = 1e-10):
    """
    Solve for volume fractions v_i by minimizing |bruggeman_eps_eff(v) - epsilon_eff|^2

    This ensures consistency with the forward bruggeman_eps_eff calculation.
    """
    eps = np.asarray(epsilons, dtype=np.complex128)
    epsilon_eff = np.complex128(epsilon_eff)
    n_comp = len(eps)

    if n_comp == 1:
        return np.array([1.0])

    def objective(v_free):
        """
        v_free: volume fractions for first n-1 components
        Last component is v_n = 1 - sum(v_free)
        """
        if np.any(v_free < 0) or np.any(v_free > 1) or np.sum(v_free) > 1:
            return 1e10

        # Construct full volume fraction array
        v_full = np.zeros(n_comp)
        v_full[:n_comp - 1] = v_free
        v_full[-1] = 1.0 - np.sum(v_free)

        if v_full[-1] < 0 or v_full[-1] > 1:
            return 1e10

        # Calculate epsilon_eff using the forward model
        try:
            eps_calc = bruggeman_eps_eff(eps, v_full)
        except ValueError:
            return 1e10

        # Minimize squared difference
        diff = eps_calc - epsilon_eff
        return np.abs(diff) ** 2

    from scipy.optimize import minimize

    # Initial guess: equal fractions
    v0_free = np.full(n_comp - 1, 1.0 / n_comp)

    # Bounds: each component between 0 and 1
    bounds = [(0, 1)] * (n_comp - 1)

    result = minimize(objective, v0_free, method='L-BFGS-B', bounds=bounds,
                      options={'ftol': tol})

    # Construct final result
    v = np.zeros(n_comp)
    v[:n_comp - 1] = result.x
    v[-1] = 1.0 - np.sum(result.x)

    # Verify solution
    eps_calc = bruggeman_eps_eff(eps, v)
    error = np.abs(eps_calc - epsilon_eff)

    if error > 0.1:  # 10% tolerance
        import warnings
        warnings.warn(f"Large error in solution: |eps_calc - eps_target| = {error:.6e}")

    return v


def solve_v_i_two_component(eps1: complex, eps2: complex, epsilon_eff_target: complex,
                            tol: float = 1e-10) -> np.ndarray:
    """
    Solve for volume fractions for a 2-component Bruggeman mixture.

    Finds v1 that matches Im(sqrt(epsilon_eff_target)).
    """
    from scipy.optimize import minimize_scalar

    eps1 = np.complex128(eps1)
    eps2 = np.complex128(eps2)
    epsilon_eff_target = np.complex128(epsilon_eff_target)

    # Target is imaginary part of sqrt(epsilon)
    target_imag = np.sqrt(epsilon_eff_target).imag

    def objective(v1):
        """Return squared difference in Im(sqrt(epsilon)) only"""
        v = np.array([v1, 1.0 - v1])
        epsilons = np.array([eps1, eps2])

        try:
            epsilon_eff_calc = bruggeman_eps_eff(epsilons, v)
            # Match imaginary part of sqrt(epsilon)
            calc_imag = np.sqrt(epsilon_eff_calc).imag
            error_imag = calc_imag - target_imag
            return error_imag ** 2
        except:
            return 1e10

    # Use bounded minimization
    result = minimize_scalar(objective, bounds=(0, 1), method='bounded')

    v = np.array([result.x, 1.0 - result.x])

    # Verify solution
    epsilon_eff_calc = bruggeman_eps_eff(np.array([eps1, eps2]), v)
    calc_imag = np.sqrt(epsilon_eff_calc).imag
    error_imag = np.abs(calc_imag - target_imag)

    if error_imag > 0.01:
        import warnings
        warnings.warn(f"Im(sqrt(eps)) error: {error_imag:.6e}")

    print(f"Im(sqrt(eps)) match: target={target_imag:.6f}, calc={calc_imag:.6f}")
    print(f"Full epsilon: target={epsilon_eff_target}, calc={epsilon_eff_calc}")

    return v


if __name__ == "__main__":

    ## this test here throws an error currently.

    epsilons = np.array([1.0, 2.0, 3.0])
    epsilon_h = 0.1
    epsilon_eff = 1.5

    v_i = solve_v_i(epsilons, epsilon_h, epsilon_eff)
    print(f"Solved volume fraction v_i: {v_i}")
    print(f"Sum of volume fractions: {np.sum(v_i)}")
