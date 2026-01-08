import numpy as np
from scipy.optimize import brentq
from scipy import special
import profiles as dp

def mass_profile(r, density_func, **kwargs):
    """
    Compute the enclosed mass within radius r for a given density profile.
    
    This function uses Gauss-Legendre quadrature with log-space transformation
    to efficiently calculate the enclosed mass by integrating 4πr²ρ(r) from a 
    small inner radius to r.
    
    Parameters
    ----------
    r : float
        Radius within which to compute the enclosed mass, in Mpc/h
    density_func : callable
        Function that returns the density at a given radius
    **kwargs : dict
        Additional arguments to pass to the density function
    
    Returns
    -------
    float
        Enclosed mass in Msun/h
    
    Notes
    -----
    For NFW profiles with r_tr >> r_s, this function automatically uses the
    analytical solution for better accuracy and performance.
    """
    """if density_func == rho_nfw and 'r_tr2' in kwargs and kwargs['r_tr'] > 5*kwargs['r_s']:
        r_s = kwargs['r_s']
        rho0 = kwargs['rho0']
        x = r / r_s
        return 4 * np.pi * rho0 * r_s**3 * (np.log(1 + x) - x/(1 + x))
    
    if density_func == mass_nfw_analytical_inf and 'r_tr' in kwargs and kwargs['r_tr'] > 5*kwargs['r_s']:
        r_s = kwargs['r_s']
        rho0 = kwargs['rho0']
        r_tr = kwargs['r_tr']
        return mass_nfw_analytical_inf(r_tr, r_s, rho0)
    """
    r_min = 1e-8  # Lower minimum radius to capture more central mass
    
    n_points = 48  # Higher number of points for better accuracy
    x, w = special.roots_legendre(n_points)
    
    s_min = np.log(r_min)
    s_max = np.log(r)
    s = 0.5 * (s_max - s_min) * x + 0.5 * (s_max + s_min)
    radius = np.exp(s)
    
    integrand = 4 * np.pi * radius**3 * np.array([density_func(r_i, **kwargs) for r_i in radius])
    
    M = 0.5 * (s_max - s_min) * np.sum(w * integrand)
    
    return M

def create_radius_array(r_min, r_max, n_points):
        """
        Create a radius array for calculations.
        
        This method generates a non-uniform array of radii that has higher
        resolution in the inner regions of the halo.
        
        Parameters
        ----------
        r_min : float
            Minimum radius in Mpc/h.
        r_max : float
            Maximum radius in Mpc/h.
        n_points : int
            Total number of points in the array.
            
        Notes
        -----
        The array uses logarithmic spacing for the inner 70% of points and
        linear spacing for the outer 30%, providing better resolution where
        profiles change more rapidly.
        Sets the class attribute r_vals.
        """
        # Use a combination of log and linear spacing to get more points in the center
        n_log = int(n_points * 0.7)
        n_lin = n_points - n_log

        # Log-spaced points for the inner region
        r_log = np.logspace(np.log10(r_min), np.log10(r_max * 0.1), n_log, endpoint=False)
        # Linearly spaced points for the outer region
        r_lin = np.linspace(r_max * 0.1, r_max, n_lin)

        # Concatenate and ensure uniqueness and sorting
        return np.unique(np.concatenate([r_log, r_lin]))
    
def bracket_rho0(M_target, r_s, r_tr, r200, r_max=None):
    """
    Solve for the NFW density normalization (rho0) that produces a specified enclosed mass.
    
    Uses the Brent method to find the value of rho0 such that the enclosed mass
    at radius r_max equals M_target.
    
    Parameters
    ----------
    M_target : float
        Target enclosed mass at r_max in Msun/h
    r_s : float
        Scale radius of the NFW profile in Mpc/h
    r_tr : float
        Truncation radius in Mpc/h (can be np.inf for untruncated profiles)
    r200 : float
        Halo radius r_200 in Mpc/h
    r_max : float, optional
        Radius at which the enclosed mass should match M_target, defaults to r200
        
    Returns
    -------
    float
        The normalized density parameter rho0 for the NFW profile
        
    Raises
    ------
    ValueError
        If the root finding algorithm cannot bracket the solution
    """
    if r_max is None:
        r_max = r200
    def f(rho0):
        return dp.mass_profile(r_max, dp.rho_nfw, r_s=r_s, rho0=rho0, r_tr=r_tr) - M_target
    low, high = 1e-12, 1e-8
    factor = 10
    for _ in range(50):
        high *= factor
        if f(low) * f(high) < 0:
            break
    if f(low) * f(high) > 0:
        raise ValueError("Could not bracket root for rho0.")
    rho0_solved = brentq(f, low, high)
    return rho0_solved
    
def cumul_mass(r_array, rho_array):
    """
    Calculate the cumulative mass profile using logarithmic integration.
    
    For spherical symmetry: M(r) = ∫[0 to r] 4π r'² ρ(r') dr'
    Change variables: x = ln(r'), dx = dr'/r', so dr' = r' dx = e^x dx
    Then: M(r) = ∫[ln(r_min) to ln(r)] 4π e^(3x) ρ(e^x) dx
    
    Parameters
    ----------
    r_array : array_like
        Array of radii in Mpc/h
    rho_array : array_like
        Array of density values in Msun/h/Mpc³, corresponding to r_array
        
    Returns
    -------
    array_like
        Array of cumulative masses in Msun/h, corresponding to each radius in r_array
    
    Notes
    -----
    Uses logarithmic integration throughout for numerical stability and
    to avoid scipy.quad integration warnings.
    """
    r_array = np.asarray(r_array, dtype=float)
    rho_array = np.asarray(rho_array, dtype=float)
    mass = np.zeros_like(r_array)
    
    r_min = 1e-10
    
    # Filter valid data points
    valid_mask = (rho_array > 0) & np.isfinite(rho_array) & (r_array > 0) & np.isfinite(r_array)
    if not np.any(valid_mask):
        return mass
    
    r_valid = r_array[valid_mask]
    rho_valid = rho_array[valid_mask]
    
    # Sort by radius for proper interpolation
    sort_idx = np.argsort(r_valid)
    r_sorted = r_valid[sort_idx]
    rho_sorted = rho_valid[sort_idx]
    
    for i, r_max in enumerate(r_array):
        if r_max < r_min:
            mass[i] = 0.0
            continue
        
        # Adaptive number of integration points based on range
        log_range = np.log(r_max / r_min)
        n_points = max(50, min(200, int(log_range * 40)))
        
        # Create logarithmic integration grid
        log_r_min = np.log(r_min)
        log_r_max = np.log(r_max)
        log_r_grid = np.linspace(log_r_min, log_r_max, n_points)
        r_grid = np.exp(log_r_grid)
        
        # Safe interpolation with extrapolation handling
        rho_interp = np.zeros_like(r_grid)
        
        for j, r_val in enumerate(r_grid):
            if r_val < r_sorted[0]:
                # Power-law extrapolation at small radii: ρ ∝ r^(-1)
                rho_interp[j] = rho_sorted[0] * (r_val / r_sorted[0])**(-1)
            elif r_val > r_sorted[-1]:
                # Power-law extrapolation at large radii: ρ ∝ r^(-3)
                rho_interp[j] = rho_sorted[-1] * (r_val / r_sorted[-1])**(-3)
            else:
                rho_interp[j] = np.interp(r_val, r_sorted, rho_sorted)
        
        # Integrand in log space: 4π r³ ρ(r)
        # This accounts for the Jacobian dr = r d(ln r)
        integrand = 4 * np.pi * r_grid**3 * rho_interp
        
        # Filter out any problematic values
        valid_integrand = np.isfinite(integrand) & (integrand >= 0)
        
        if np.any(valid_integrand):
            # Trapezoidal integration in log space
            mass[i] = np.trapz(integrand[valid_integrand], 
                             log_r_grid[valid_integrand])
        else:
            mass[i] = 0.0
    
    return mass
    
def normalize_component(density_func, args, M_target, r200_loc, r_inf_factor=200.0):
    """
    Calculate normalization factor so that the component encloses M_target within ~infinity (default 100*r200).

    This treats R_inf = r_inf_factor * r200_loc as a practical infinity. Set r_inf_factor large enough
    (e.g. 50-200) to capture the asymptotic mass for truncated / fast–decaying profiles.

    Parameters
    ----------
    density_func : callable
        Density profile function ρ(r, *args) WITHOUT overall normalization.
    args : tuple
        Arguments passed to density_func (excluding normalization).
    M_target : float
        Desired total mass when integrated to R_inf (e.g. f_comp * M_tot_asymp).
    r200_loc : float
        Halo r200 in Mpc/h.
    r_inf_factor : float, optional
        Multiple of r200 used as practical infinity (default 100).

    Returns
    -------
    float
        Normalization A such that 4π ∫₀^{R_inf} A·ρ(r) r² dr ≈ M_target.

    Notes
    -----
    Use this when fractions are defined w.r.t. the asymptotic (truncated) halo mass rather than mass inside r200.
    If you instead want mass inside r200 to match, reduce r_inf_factor to 1 or use a separate function.
    """
    def unnorm_func(r):
        return density_func(r, *args)
    R_inf = r_inf_factor * r200_loc
    unnorm_mass = dp.mass_profile(R_inf, unnorm_func)
    if unnorm_mass <= 0 or not np.isfinite(unnorm_mass):
        raise ValueError("Invalid unnormalized mass in normalize_component (infinite mode).")
    return M_target / unnorm_mass
    
def cumul_mass_single(r, rho_array, r_array):
    """
    Calculate cumulative mass using logarithmic integration.
    
    For spherical symmetry: M(r) = ∫[0 to r] 4π r'² ρ(r') dr'
    Change variables: x = ln(r'), dx = dr'/r', so dr' = r' dx = e^x dx
    Then: M(r) = ∫[ln(r_min) to ln(r)] 4π e^(3x) ρ(e^x) dx
    """
    r_min = 1e-8
    if r < r_min:
        return 0.0
    
    # Use logarithmic spacing
    log_r_min = np.log(r_min)
    log_r_max = np.log(r)
    log_r_points = np.linspace(log_r_min, log_r_max, 1000)
    r_points = np.exp(log_r_points)
    
    # Interpolate density at these points
    rho_points = np.interp(r_points, r_array, rho_array)
    
    # Integrand in log space: 4π r³ ρ(r) (since dr = r d(ln r) in log space)
    integrand = 4 * np.pi * r_points**3 * rho_points
    
    # Integrate using trapezoidal rule in log space
    mass = np.trapz(integrand, log_r_points)
    
    return mass    
    
    
    