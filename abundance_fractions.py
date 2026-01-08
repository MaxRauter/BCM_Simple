import numpy as np

# --------------------------------------------------------------------
# Abundance Fractions
# --------------------------------------------------------------------

def f_bgas(M, fbar_loc, Mc=1.2e14, beta=0.6):
    """
    Calculate the bound gas fraction as a function of halo mass.
    
    This function implements Equation 2.19 which describes how the 
    bound gas fraction varies with halo mass.
    
    Parameters
    ----------
    M : float or numpy.ndarray
        Halo mass in solar masses.
    fbar_loc : float
        Local baryon fraction (Omega_b/Omega_m).
    Mc : float, optional
        Characteristic mass in solar masses. Default is 1.2e14.
    beta : float, optional
        Power-law slope. Default is 0.6.
    
    Returns
    -------
    float or numpy.ndarray
        Bound gas fraction relative to the total mass.
        
    Notes
    -----
    The bound gas fraction is calculated as:
    f_bgas(M) = (Omega_b/Omega_m) / (1 + (Mc/M)^beta)
    """
    return fbar_loc / (1 + (Mc/M)**beta)

def g_func(x, alpha=-1.779, delta=4.394, gamma=0.547):
    """
    Helper function for calculating the central galaxy abundance.
    
    This implements Equation 2.20's internal function.
    
    Parameters
    ----------
    x : float or numpy.ndarray
        Log10 of the normalized mass (log10(M/M1)).
    alpha : float, optional
        Model parameter. Default is -1.779.
    delta : float, optional
        Model parameter. Default is 4.394.
    gamma : float, optional
        Model parameter. Default is 0.547.
    
    Returns
    -------
    float or numpy.ndarray
        Functional value used in stellar fraction calculation.
        
    Notes
    -----
    The function is defined as:
    g(x) = -log10(10^alpha * x + 1) + delta * (log10(1 + exp(x)))^gamma / (1+exp(10*-x))
    mistake in schneider paper, should be log10(1 + exp(10 * (-x))) not log10(1 + exp(10 * x)), as in referance paper.
    """
    return - np.log10(10**(alpha * x) + 1) + delta * (np.log10(1 + np.exp(x)))**gamma / (1 + np.exp(10**(-x)))

def f_cgal(M, epsilon=0.023, M1=1.526e11): 
    """
    Calculate the central galaxy stellar fraction as a function of halo mass.
    
    This function implements Equation 2.20 which models the stellar content
    of central galaxies.
    
    Parameters
    ----------
    M : float or numpy.ndarray
        Halo mass in solar masses.
    epsilon : float, optional
        Normalization parameter. Default is 0.023.
    M1 : float, optional
        Characteristic mass in solar masses. Default is 1.526e11.
    
    Returns
    -------
    float or numpy.ndarray
        Central galaxy stellar fraction relative to the total mass.
        
    Notes
    -----
    The central galaxy stellar fraction is calculated as:
    fcgal(M) = epsilon * (M1/M) * 10^(g(log10(M/M1)) - g(0))
    where g is the helper function g_func.
    """
    x = np.log10(M/M1)
    return epsilon * (M1/M) * 10**( g_func(x) - g_func(0) )

def f_egas(fbgas, fcgal, fbar_loc):
    """
    Calculate the ejected gas fraction.
    
    This function implements Equation 2.21 which represents gas that has been
    ejected from the halo due to feedback processes.
    
    Parameters
    ----------
    fbgas : float or numpy.ndarray
        Bound gas fraction.
    fcgal : float or numpy.ndarray
        Central galaxy stellar fraction.
    fbar_loc : float
        Local baryon fraction (Omega_b/Omega_m).
    
    Returns
    -------
    float or numpy.ndarray
        Ejected gas fraction relative to the total mass.
        
    Notes
    -----
    The ejected gas fraction is calculated as:
    f_egas(M) = fbar - f_bgas(M) - f_cgal(M)
    It represents baryons that have been expelled from the halo.
    """
    return fbar_loc - fbgas - fcgal

def f_rdm(fbar_loc):
    """
    Calculate the relaxed dark matter fraction.
    
    Parameters
    ----------
    fbar_loc : float
        Local baryon fraction (Omega_b/Omega_m).
    
    Returns
    -------
    float
        Relaxed dark matter fraction relative to the total mass.
        
    Notes
    -----
    The relaxed dark matter fraction is calculated as:
    f_rdm = 1 - fbar
    It represents the dark matter component that has reached equilibrium.
    """
    return 1 - fbar_loc
