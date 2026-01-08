import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy import special
import utils as ut

# --------------------------------------------------------------------
# Density Profiles
# --------------------------------------------------------------------

def rho_background(v, halo_masses):
    """
    Calculate the background density profile.
    
    This function computes the cosmic background density using the critical 
    density and matter density parameter. The formula is given by Eq. 2.9:
    ρ_background(v) = ρ_c * Omega_m - (1/v) * Σ_i ρ_i
    where ρ_i is the density of each halo.
    
    Parameters
    ----------
    v : float
        Volume of the simulation in (Mpc/h)^3
    halo_masses : array_like
        List of halo masses in Msun/h
        
    Returns
    -------
    float
        Background density in Msun/h/(Mpc/h)^3
    
    Notes
    -----
    Currently uses fixed values for critical density and Omega_m.
    """
    rho_c = 2.775e11 #h^2 M_sun / Mpc^3
    Omega_m = 0.3071
    return rho_c * Omega_m 

def rho_nfw(r, r_s, rho0, r_tr, r_0=1e-10):
    """
    Calculate the truncated NFW density profile.
    
    Implements the truncated NFW profile from Eq. 2.8:
    ρ_nfw(x, τ) = ρ0 / [x (1+x)^2 (1 + (x/τ)^2)^2]
    where x = r/r_s and τ = r_tr/r_s.
    
    Parameters
    ----------
    r : float or array_like
        Radius in Mpc/h
    r_s : float
        Scale radius in Mpc/h
    rho0 : float
        Characteristic density in Msun/h/(Mpc/h)^3
    r_tr : float
        Truncation radius in Mpc/h
    r_0 : float, optional
        Minimum radius to avoid singularity, default 1e-10 Mpc/h
    
    Returns
    -------
    float or array_like
        NFW density at the specified radius/radii in Msun/h/(Mpc/h)^3
    
    Notes
    -----
    Best results are achieved with τ = r_tr/r_s = 8c (Schneider & Teyssier 2016).
    """
    r = np.maximum(r, r_0)  # Ensure r is at least r_0 to avoid division by zero
    x = r / r_s
    tau = r_tr / r_s 
    return rho0 / ( x * (1 + x)**2 * (1 + (x/tau)**2)**2 )

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

def y_bgas(r, r_s, r200, y0, c, rho0, r_tr):
    """
    Calculate the baryonic gas density profile.
    
    This function implements a modified profile for baryonic gas that consists
    of an inner profile (power law) and an outer NFW profile, with a smooth
    transition between them at r200/√5.
    
    Parameters
    ----------
    r : float or array_like
        Radius in Mpc/h
    r_s : float
        Scale radius in Mpc/h
    r200 : float
        Virial radius in Mpc/h
    y0 : float
        Normalization parameter
    c : float
        Concentration parameter (r200/r_s)
    rho0 : float
        Characteristic density in Msun/h/(Mpc/h)^3
    r_tr : float
        Truncation radius in Mpc/h
    
    Returns
    -------
    float or array_like
        Baryonic gas density at the specified radius/radii in Msun/h/(Mpc/h)^3
    
    Notes
    -----
    The inner profile is given by Eq. 2.10: y0 * (ln(1+x)/x)^Γ_eff
    The outer profile follows the NFW form.
    The transition radius is r200/√5.
    """
    sqrt5 = np.sqrt(5)
    r_transition = r200 / sqrt5
    x = r / r_s

    Gamma_eff = (1 + 3*c/sqrt5) * np.log(1 + c/sqrt5) / ((1 + c/sqrt5)*np.log(1 + c/sqrt5) - c/sqrt5)
    
    inner_val = y0 * (np.log(1 + x) / x)**Gamma_eff
    
    outer_val = rho_nfw(r, r_s, y0, r_tr)
    
    x_trans = r_transition / r_s
    inner_at_trans = y0 * (np.log(1 + x_trans) / x_trans)**Gamma_eff
    outer_at_trans = rho_nfw(r_transition, r_s, y0, r_tr)
    scale_factor = outer_at_trans / inner_at_trans
    inner_scaled = inner_val * scale_factor
    #outer_scaled = outer_val / scale_factor
    
    return np.where(r <= r_transition, inner_scaled, outer_val)

def y_egas(r, M_tot, r_ej):
    """
    Calculate the ejected gas density profile.
    
    This function implements the ejected gas profile as a 3D Gaussian 
    distribution as described in Eq. 2.13:
    y_egas(r) = M_tot / ((2π r_ej²)^(3/2)) * exp[-r²/(2r_ej²)]
    
    Parameters
    ----------
    r : float or array_like
        Radius in Mpc/h
    M_tot : float
        Total mass of the ejected gas component in Msun/h
    r_ej : float
        Characteristic ejection radius in Mpc/h
    
    Returns
    -------
    float or array_like
        Ejected gas density at the specified radius/radii in Msun/h/(Mpc/h)^3
    
    Notes
    -----
    This component represents gas that has been ejected from the halo due to
    feedback processes such as AGN activity and supernovae.
    """
    norm = M_tot / ((2 * np.pi * r_ej**2)**(1.5))
    return norm * np.exp(- r**2 / (2 * r_ej**2))

def y_cgal(r, M_tot, R_h):
    """
    Calculate the central galaxy (stellar) density profile.

    Implements: y_cgal(r) = M_tot / (4 π^{3/2} R_h) * r^{-2} * exp[-(r/(2 R_h))^2]
    Integrated over r∈[0,∞) gives total mass M_tot.

    Parameters
    ----------
    r : float or array_like
        Radius in Mpc/h
    M_tot : float
        Total stellar mass in Msun/h
    R_h : float
        Characteristic (half-light / Hernquist-like scale) radius in Mpc/h
    """
    # Ensure array operations are element-wise; previous implementation used built-in max(r, 1e-10)
    # which returns a scalar when r is an ndarray (the global maximum), flattening the profile.
    r_arr = np.asarray(r, dtype=float)
    r_safe = np.maximum(r_arr, 1e-10)

    norm = M_tot / (4.0 * np.pi**1.5 * R_h)
    exp_term = np.exp(- (r_safe / (2.0 * R_h))**2)
    rho = norm * r_safe**(-2.0) * exp_term

    # Preserve scalar input type
    if np.isscalar(r):
        return float(rho)
    return rho

def y_rdm_ac(r, r_s, rho0, r_tr, norm=1.0,
             a=0.68,                # contraction strength
             f_cdm=0.839,           # CDM fraction of total mass
             baryon_components=None, # list of (r_array, y_vals) tuples
             verbose=False):
    """
    Calculate the adiabatically contracted dark matter profile with debugging.
    """
    def xi_of_r(rf):
        
        #if verbose:
        #    print(f"\n=== DEBUG xi_of_r for rf={rf:.6e} ===")
        
        # Calculate baryon mass at rf
        M_b_rf = 0.0
        if baryon_components:
            #if verbose:
            #    print(f"Processing {len(baryon_components)} baryon components:")
            
            for i, (r_array, rho_array) in enumerate(baryon_components):
                try:
                    # Debug the input arrays
                    #if verbose:
                    #    print(f"  Component {i}:")
                    #    print(f"    r_array range: {r_array.min():.3e} to #{r_array.max():.3e}")
                    #    print(f"    rho_array range: {rho_array.min():.3e} to {rho_array.max():.3e}")
                    #    print(f"    Valid rho values: {np.sum(np.isfinite#(rho_array) & (rho_array > 0))}/{len(rho_array)}")
                    
                    baryon_mass = ut.cumul_mass_single(rf, rho_array, r_array)
                    
                    #if verbose:
                    #    print(f"    Baryon mass at rf={rf:.3e}: {baryon_mass:.3e}")
                    
                    if not np.isfinite(baryon_mass) or baryon_mass < 0:
                        #if verbose:
                        #    print(f"    WARNING: Invalid baryon mass {baryon_mass}, setting to 0")
                        baryon_mass = 0.0
                    
                    M_b_rf += baryon_mass
                    
                except Exception as e:
                    #if verbose:
                    #    print(f"    ERROR calculating baryon mass for component {i}: {e}")
                    continue
        
        #if verbose:
        #    print(f"Total baryon mass M_b_rf: {M_b_rf:.6e}")
        
        def G(ri):
            #if verbose and ri in [1e-8, 1e8]:  # Only debug at bounds
            #    print(f"  G(ri={ri:.6e}):")
            
            try:
                # Calculate initial mass using mass_profile
                M_i_ri = mass_profile(ri, rho_nfw, r_s=r_s, rho0=rho0, r_tr=r_tr)
                
                #if verbose and ri in [1e-8, 1e8]:
                #    print(f"    M_i_ri: {M_i_ri:.6e}")
                
                # Check for invalid mass
                if not np.isfinite(M_i_ri) or M_i_ri <= 0:
                    #if verbose and ri in [1e-8, 1e8]:
                    #    print(f"    ERROR: Invalid M_i_ri = {M_i_ri}")
                    return np.nan
                
                # Calculate f_cdm - fraction of CDM in initial mass
                #total_initial_mass = M_i_ri + M_b_rf
                #if total_initial_mass <= 0:
                    #if verbose and ri in [1e-8, 1e8]:
                    #    print(f"    ERROR: Total initial mass <= 0: {total_initial_mass}")
                #    return np.nan
                
                #f_cdm_actual = M_i_ri / total_initial_mass
                f_cdm_actual = f_cdm
                # Calculate final mass
                M_f_rf = f_cdm_actual * M_i_ri + M_b_rf
                
                #if verbose and ri in [1e-8, 1e8]:
                    #print(f"    f_cdm_actual: {f_cdm_actual:.6f}")
                    #print(f"    M_f_rf: {M_f_rf:.6e}")
                
                # Check for invalid final mass
                if not np.isfinite(M_f_rf) or M_f_rf <= 0:
                    #if verbose and ri in [1e-8, 1e8]:
                    #    print(f"    ERROR: Invalid M_f_rf = {M_f_rf}")
                    return np.nan
                
                # Calculate the ratio
                ratio = M_i_ri / M_f_rf
                
                #if verbose and ri in [1e-8, 1e8]:
                    #print(f"    ratio M_i/M_f: {ratio:.6f}")
                
                # Calculate G
                result = rf/ri - 1.0 - a*(ratio - 1.0)
                
                #if verbose and ri in [1e-8, 1e8]:
                #    print(f"    rf/ri: {rf/ri:.6f}")
                #    print(f"    a*(ratio-1): {a*(ratio-1.0):.6f}")
                #    print(f"    G result: {result:.6e}")
                
                return result
                
            except Exception as e:
                #if verbose and ri in [1e-8, 1e8]:
                #    print(f"    EXCEPTION in G: {e}")
                return np.nan

        # Set bounds more conservatively
        ri_min = max(rf*1e-3, 1e-10)  # Less aggressive lower bound
        ri_max = min(rf*1e3, r_tr*2)   # Less aggressive upper bound
        
        #if verbose:
        #    print(f"Bounds: ri_min={ri_min:.6e}, ri_max={ri_max:.6e}")
        
        try:
            g_min = G(ri_min)
            g_max = G(ri_max)
            
            #if verbose:
            #    print(f"G(ri_min) = {g_min:.6e}")
            #    print(f"G(ri_max) = {g_max:.6e}")

            # Check for NaN values
            if np.isnan(g_min) or np.isnan(g_max):
                #if verbose:
                #    print(f"ERROR: G(ri) returned NaN at bounds. Returning xi=1.0")
                return 1.0

            # Check if root is bracketed
            if g_min * g_max >= 0:
                #if verbose:
                #    print(f"WARNING: Root not bracketed.")
                #    print(f"  This suggests either:")
                #    print(f"  - Very weak contraction (xi ≈ 1)")
                #    print(f"  - Mass calculation issues")
                #    print(f"  - Need wider bounds")
                
                # Try to find a reasonable solution
                if abs(g_min) < 1e-6:
                    xi = rf / ri_min
                    #if verbose:
                    #    print(f"  Using ri_min solution: xi = {xi:.6f}")
                    return xi
                elif abs(g_max) < 1e-6:
                    xi = rf / ri_max
                    #if verbose:
                    #    print(f"  Using ri_max solution: xi = {xi:.6f}")
                    return xi
                else:
                    #if verbose:
                    #    print(f"  No good solution found, returning xi=1.0")
                    return 1.0

            # Solve for ri
            ri = brentq(G, ri_min, ri_max, xtol=1e-8, rtol=1e-8, maxiter=100)
            xi = rf / ri
            
            #if verbose:
            #    print(f"Solution: ri = {ri:.6e}, xi = {xi:.6f}")

            return xi

        except Exception as e:
            if verbose:
                print(f"ERROR: Exception in xi_of_r: {e}")
                import traceback
                traceback.print_exc()
            return 1.0

    # Main function logic
    if np.isscalar(r):
        xi = xi_of_r(r)
        ri = r/xi
        return xi**(-3) * rho_nfw(ri, r_s, rho0, r_tr)

    out = np.zeros_like(r)
    xi_vals = np.zeros_like(r)
    M_c = np.zeros_like(r) 
    for i, rf in enumerate(r):
        xi = xi_of_r(rf)
        #xi = min(xi, 10.0)  # Cap maximum contraction
        #xi = max(xi, 0.1)   # Cap minimum contraction
        xi_vals[i] = xi
        #xi = 0.85
        ri = rf/xi
        #out[i] = norm * xi**(-3) * rho_nfw(ri, r_s, rho0, r_tr)
        M_c[i] = mass_profile(ri/xi, rho_nfw, r_s=r_s, rho0=rho0, r_tr=r_tr)*f_cdm
    dr = np.gradient(r)
    rho_c = np.gradient(M_c, r) / (4 * np.pi * r**2)
    if verbose:  
        print(f"\nFinal xi_vals: {xi_vals[:10]}")
        idx_almost_one = np.argmax(np.isclose(xi_vals, 1.0, atol=1e-3))
        print(f"First xi ~ 1 at index: {idx_almost_one} and r = {r[idx_almost_one]}")
        
        # Additional diagnostics
        print(f"Xi statistics:")
        print(f"  Min xi: {xi_vals.min():.6f}")
        print(f"  Max xi: {xi_vals.max():.6f}")
        print(f"  Mean xi: {xi_vals.mean():.6f}")
        print(f"  Xi = 1 count: {np.sum(np.isclose(xi_vals, 1.0, atol=1e-3))/ len(xi_vals)}")
    
    return rho_c



def y_rdm_ac2(r, r_s, rho0, r_tr, M_i, M_f, verbose, norm=1.0,
              a=0.68,               # contraction strength
              f_cdm=0.839):         # CDM fraction of total mass
    """
    Calculate the adiabatically contracted dark matter profile using pre-computed mass profiles.
    
    This alternative implementation of adiabatic contraction uses pre-computed
    mass profiles M_i and M_f rather than calculating them on-the-fly.
    
    Parameters
    ----------
    r : array_like
        Radius array in Mpc/h
    r_s : float
        Scale radius in Mpc/h
    rho0 : float
        Characteristic density in Msun/h/(Mpc/h)^3
    r_tr : float
        Truncation radius in Mpc/h
    M_i : array_like
        Initial mass profile (before contraction) in Msun/h
    M_f : array_like
        Final mass profile (after contraction) in Msun/h
    verbose : bool
        Whether to print diagnostic information
    norm : float, optional
        Normalization factor, default 1.0
    a : float, optional
        Contraction strength parameter, default 0.68
    f_cdm : float, optional
        CDM fraction of total mass, default 0.839
    
    Returns
    -------
    array_like
        Contracted dark matter density at the specified radii in Msun/h/(Mpc/h)^3
    
    Notes
    -----
    This method interpolates the mass profiles to determine the contraction
    parameter ξ at each radius. It then applies the same contraction formula as
    y_rdm_ac but avoids recomputing the mass profiles at each step.
    """
    def xi_of_r(rf):
        def G(ri):
            M_i_interp = np.interp(ri, r, M_i)
            M_f_interp = np.interp(rf, r, M_f)
            return rf/ri - 1.0 - a*(M_i_interp/M_f_interp - 1.0)

        ri_min, ri_max = rf*1e-3, rf * 100
        
        g_min, g_max = G(ri_min), G(ri_max)
        if g_min * g_max >= 0:
            print(f"WARNING: Root not bracketed for rf={rf:.3e}. Bounds=[{ri_min:.3e}, {ri_max:.3e}].")
            return 1.0
        
        ri = brentq(G, ri_min, ri_max, xtol=1e-6, disp=True)
        return rf/ri

    out = np.zeros_like(r)
    xi_vals = np.zeros_like(r)
    for i, rf in enumerate(r):
        xi = xi_of_r(rf)
        xi_vals[i] = xi
        ri = rf/xi
        out[i] = norm * xi**(-3) * rho_nfw(ri, r_s, rho0, r_tr)
    if verbose:
        print(f"xi_vals2: {xi_vals[:10]}")
        idx_almost_one = np.argmax(np.isclose(xi_vals, 1.0, atol=1e-3))
        print(f"First xi ~ 1 at index: {idx_almost_one} and r = {r[idx_almost_one]}")
    return out

def y_rdm_simple_xi(r, r_s, rho0, r_tr, xi=0.85, norm=1.0,
                   a=0.68,          # contraction strength
                   f_cdm=0.839,     # CDM fraction of total mass
                   baryon_components=None): # list of (r_array, y_vals) tuples
    """
    Calculate the dark matter profile with a fixed contraction parameter.
    
    This simplified version of the adiabatic contraction model uses a fixed
    contraction parameter ξ for all radii instead of solving for it separately
    at each radius.
    
    Parameters
    ----------
    r : float or array_like
        Radius in Mpc/h
    r_s : float
        Scale radius in Mpc/h
    rho0 : float
        Characteristic density in Msun/h/(Mpc/h)^3
    r_tr : float
        Truncation radius in Mpc/h
    xi : float, optional
        Fixed contraction parameter, default 0.85
    norm : float, optional
        Normalization factor, default 1.0
    a : float, optional
        Contraction strength parameter (not used in this function), default 0.68
    f_cdm : float, optional
        CDM fraction of total mass (not used in this function), default 0.839
    baryon_components : list, optional
        Baryon components (not used in this function)
    
    Returns
    -------
    float or array_like
        Contracted dark matter density at the specified radius/radii in Msun/h/(Mpc/h)^3
    
    Notes
    -----
    This simplified model provides a way to apply contraction without the
    computational expense of solving for ξ at each radius. This method uses
    radius-dependent contraction for r < 0.04, transitioning to no contraction (ξ=1)
    for r ≥ 0.04.
    """
    limit = 0.04
    inner_xi = 0.65
    if r < limit:
        xi_interp = 1 - ((1 - inner_xi)) * ((1 - r / limit))
        xi = np.clip(xi_interp, inner_xi, 1.0)
    else:
        xi = 1
    ri = r / xi
    return norm * xi**(-3) * rho_nfw(ri, r_s, rho0, r_tr)

def y_rdm_fixed_xi(r, r_s, rho0, r_tr, xi=0.85, norm=1.0,
                   a=0.68,          # contraction strength
                   f_cdm=0.839,     # CDM fraction of total mass
                   baryon_components=None): # list of (r_array, y_vals) tuples
    """
    Calculate the dark matter profile with a fixed contraction parameter.
    
    This simplified version of the adiabatic contraction model uses a fixed
    contraction parameter ξ for all radii instead of solving for it separately
    at each radius.
    
    Parameters
    ----------
    r : float or array_like
        Radius in Mpc/h
    r_s : float
        Scale radius in Mpc/h
    rho0 : float
        Characteristic density in Msun/h/(Mpc/h)^3
    r_tr : float
        Truncation radius in Mpc/h
    xi : float, optional
        Fixed contraction parameter, default 0.85
    norm : float, optional
        Normalization factor, default 1.0
    a : float, optional
        Contraction strength parameter (not used in this function), default 0.68
    f_cdm : float, optional
        CDM fraction of total mass (not used in this function), default 0.839
    baryon_components : list, optional
        Baryon components (not used in this function)
    
    Returns
    -------
    float or array_like
        Contracted dark matter density at the specified radius/radii in Msun/h/(Mpc/h)^3
    
    Notes
    -----
    This simplified model provides a way to apply contraction without the
    computational expense of solving for ξ at each radius. This method uses
    radius-dependent contraction for r < 0.04, transitioning to no contraction (ξ=1)
    for r ≥ 0.04.
    """
    limit = 0.04
    inner_xi = 0.65
    if r < limit:
        xi_interp = 1 - ((1 - inner_xi)) * ((1 - r / limit))
        xi = np.clip(xi_interp, inner_xi, 1.0)
    else:
        xi = 1
    xi = 1
    ri = r / xi
    return norm * xi**(-3) * rho_nfw(ri, r_s, rho0, r_tr)

def y_rdm_ac3(r, r_s, rho0, r_tr, norm=1.0,
             a=0.68,
             f_cdm=0.839,
             baryon_components=None,
             verbose=False):
    """
    Calculate the adiabatically contracted dark matter profile.
    """
    
    # Baryon mass profile diagnostic
    if verbose and baryon_components is not None:
        print("\n=== Baryon Mass Profile Check ===")
        test_radii = np.logspace(-2, 2, 10)
        for rf_test in test_radii:
            M_b_test = 0.0
            rho_b_test = 0.0
            for r_array, rho_array in baryon_components:
                M_b_test += ut.cumul_mass_single(rf_test, rho_array, r_array)
                rho_b_test += np.interp(rf_test, r_array, rho_array)
            M_i_test = mass_profile(rf_test, rho_nfw, r_s=r_s, rho0=rho0, r_tr=r_tr)
            rho_dm_test = rho_nfw(rf_test, r_s, rho0, r_tr)
            print(f"r={rf_test:8.3f}: M_b={M_b_test:.3e}, M_i={M_i_test:.3e}, "
                  f"M_b/M_i={M_b_test/M_i_test:.4f}, ρ_b={rho_b_test:.3e}, ρ_dm={rho_dm_test:.3e}")
    
    def xi_of_r(rf):
        """
        Solve for contraction parameter ξ at final radius rf.
        """
        
        # Calculate baryon mass and density at rf
        M_b_rf = 0.0
        rho_b_rf = 0.0
        
        if baryon_components:
            for i, (r_array, rho_array) in enumerate(baryon_components):
                try:
                    baryon_mass = ut.cumul_mass_single(rf, rho_array, r_array)
                    
                    if not np.isfinite(baryon_mass) or baryon_mass < 0:
                        baryon_mass = 0.0
                    
                    M_b_rf += baryon_mass
                    
                    # Get local density at rf
                    baryon_density = np.interp(rf, r_array, rho_array)
                    if np.isfinite(baryon_density) and baryon_density > 0:
                        rho_b_rf += baryon_density
                    
                except Exception as e:
                    if verbose:
                        print(f"    WARNING: Error calculating baryon mass for component {i}: {e}")
                    continue
        
        # If no baryons, no contraction
        if M_b_rf <= 0:
            return 1.0
        
        # Get DM properties at rf
        M_i_at_rf = mass_profile(rf, rho_nfw, r_s=r_s, rho0=rho0, r_tr=r_tr)
        rho_dm_rf = rho_nfw(rf, r_s, rho0, r_tr)
        
        if M_i_at_rf <= 0 or not np.isfinite(M_i_at_rf):
            return 1.0
        
        # IMPROVED THRESHOLD LOGIC:
        # Check if baryons are negligible using MULTIPLE criteria
        
        # 1. Absolute density threshold: if baryon density < 1e4 Msun/h/Mpc³, skip
        #    (this is ~10^-9 of typical inner halo densities)
        absolute_density_threshold = 1e4  # Msun/h/Mpc³
        
        # 2. Relative density threshold: if ρ_b/ρ_dm < 0.001
        relative_density_threshold = 0.001
        
        # 3. Mass ratio threshold: if M_b/M_i < 0.01
        mass_ratio_threshold = 0.01
        
        # Check all conditions
        absolute_density_negligible = rho_b_rf < absolute_density_threshold
        relative_density_negligible = (rho_b_rf / rho_dm_rf) < relative_density_threshold if rho_dm_rf > 0 else True
        mass_negligible = (M_b_rf / M_i_at_rf) < mass_ratio_threshold
        
        # If absolute density is negligible OR (both relative density AND mass are negligible)
        if absolute_density_negligible or (relative_density_negligible and mass_negligible):
            if verbose and rf > 1.0:
                print(f"  rf={rf:.3e}: ρ_b={rho_b_rf:.3e}, ρ_b/ρ_dm={rho_b_rf/rho_dm_rf:.6f}, "
                      f"M_b/M_i={M_b_rf/M_i_at_rf:.4f} -> skipping contraction")
            return 1.0
        
        def G(ri):
            """
            Root function: G(ri) = rf/ri - [M_i(ri) / M_f(rf)]^a
            """
            try:
                M_i_ri = mass_profile(ri, rho_nfw, r_s=r_s, rho0=rho0, r_tr=r_tr)
                
                if not np.isfinite(M_i_ri) or M_i_ri <= 0:
                    return np.nan
                
                # Mass conservation: M_f = M_i + M_b
                M_f_rf = M_i_ri + M_b_rf
                
                if M_f_rf <= 0:
                    return np.nan
                
                mass_ratio = M_i_ri / M_f_rf
                result = rf / ri - mass_ratio**a
                
                return result
                
            except Exception as e:
                if verbose:
                    print(f"    EXCEPTION in G(ri={ri:.3e}): {e}")
                return np.nan
        
        # Bounds: ri ∈ [rf, 10*rf] for ξ ∈ [1.0, 0.1]
        ri_min = rf * 1.0
        ri_max = rf * 10.0
        
        try:
            g_min = G(ri_min)
            g_max = G(ri_max)
            
            if np.isnan(g_min) or np.isnan(g_max):
                if verbose:
                    print(f"  rf={rf:.3e}: G(ri) returned NaN. Returning ξ=1.0")
                return 1.0
            
            # Check if root is bracketed
            if g_min * g_max >= 0:
                if abs(g_min) < 1e-6:
                    return 1.0
                elif abs(g_max) < 1e-6:
                    return rf / ri_max
                
                if g_min < 0 and g_max < 0:
                    ri_max *= 10
                    g_max = G(ri_max)
                    if np.isnan(g_max) or g_min * g_max >= 0:
                        return 1.0
                else:
                    return 1.0
            
            # Solve for ri
            ri = brentq(G, ri_min, ri_max, xtol=1e-8, rtol=1e-8, maxiter=100)
            xi = rf / ri
            
            if not np.isfinite(xi) or xi <= 0:
                return 1.0
            
            # Validate: ξ should be in [0.1, 1.0]
            if xi > 1.0:
                return 1.0
            
            if xi < 0.1:
                if verbose:
                    print(f"  WARNING: Very strong contraction ξ={xi:.6f} at rf={rf:.3e}. Capping at 0.1")
                return 0.1
            
            return xi
            
        except Exception as e:
            if verbose:
                print(f"  ERROR: Exception in xi_of_r for rf={rf:.3e}: {e}")
            return 1.0
    
    # Main function
    if np.isscalar(r):
        xi = xi_of_r(r)
        ri = r / xi
        return norm * xi**(-3) * rho_nfw(ri, r_s, rho0, r_tr)
    
    out = np.zeros_like(r)
    xi_vals = np.zeros_like(r)
    
    for i, rf in enumerate(r):
        xi = xi_of_r(rf)
        xi_vals[i] = xi
        ri = rf / xi
        out[i] = norm * xi**(-3) * rho_nfw(ri, r_s, rho0, r_tr)
    
    if verbose:
        print(f"\n=== Adiabatic Contraction Results ===")
        print(f"Contraction parameter ξ statistics:")
        print(f"  Min ξ:  {xi_vals.min():.6f} (strongest contraction)")
        print(f"  Max ξ:  {xi_vals.max():.6f}")
        print(f"  Mean ξ: {xi_vals.mean():.6f}")
        n_total = len(xi_vals)
        n_no_contract = np.sum(np.isclose(xi_vals, 1.0, atol=1e-3))
        print(f"  ξ ≈ 1 (no contraction): {100*n_no_contract/n_total:.1f}% ({n_no_contract}/{n_total} points)")
        
        idx_min = np.argmin(xi_vals)
        idx_max = np.argmax(xi_vals)
        print(f"  Strongest contraction at r = {r[idx_min]:.4f} Mpc/h (ξ = {xi_vals[idx_min]:.4f})")
        print(f"  Weakest contraction at r = {r[idx_max]:.4f} Mpc/h (ξ = {xi_vals[idx_max]:.4f})")
        
        idx_almost_one = np.where(xi_vals > 0.99)[0]
        if len(idx_almost_one) > 0:
            r_transition = r[idx_almost_one[0]]
            print(f"  Contraction negligible (ξ > 0.99) beyond r ≈ {r_transition:.3f} Mpc/h")
    
    return out