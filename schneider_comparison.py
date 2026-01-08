import numpy as np
import profiles as dp
import utils as ut
from scipy.interpolate import interp1d

class SchneiderExamples:
    """
    A class for reproducing the Schneider and Teyssier 2015 examples for BCMs.
    """
    def __init__(self, verbose=False): 
        # Initializes Paramters that are same for all cases from the paper
        import params as par
        self.M200 = par.DEFAULTS['M200']
        self.r200 = par.DEFAULTS['r200']
        self.c = par.DEFAULTS['c']
        self.h = par.DEFAULTS['h']
        self.z = par.DEFAULTS['z']
        self.Om = par.DEFAULTS['Omega_m']
        self.Ol = 1 - self.Om
        self.Ob = par.DEFAULTS['Omega_b']
        self.fbar = self.Ob / self.Om
        self.r_ej = par.DEFAULTS['r_ej_factor'] * self.r200
        self.R_h = par.DEFAULTS['R_h_factor'] * self.r200
        self.r_s = par.DEFAULTS['r_s_factor'] * self.r200
        self.r_tr = par.DEFAULTS['r_tr_factor'] * self.r200
        self.verbose = verbose
    
    def calculate_single_case(self,f,r_min=0.001, r_max=None, n_points=1000):
        if r_max is None:
            r_max = 200 * self.r200 
            
        self.f_rdm = f['f_rdm']
        self.f_bgas = f['f_bgas']
        self.f_cgal = f['f_cgal']
        self.f_egas = f['f_egas']
        
        self.rmax = r_max
        
        # Create a radius array
        self.r_vals = ut.create_radius_array(r_min, r_max, n_points)
        
        # Calculate target NFW mass profile
        M_nfw = self._calc_NFW_target_mass()
        
        # Calculate normalizations
        norm_bgas, norm_egas, norm_cgal, norm_rdm_fixed_xi = self._calculate_normalizations()
        if self.verbose:
            print(f"Component normalizations to contain M200:")
            print(f"  bgas: {norm_bgas:.3e}")
            print(f"  egas: {norm_egas:.3e}")
            print(f"  cgal: {norm_cgal:.3e}")
        
        # Calculate density profiles
        rho_dmo_vals, rho_nfw_vals, rho_bkg_vals, y_bgas_vals, y_egas_vals, y_cgal_vals, y_rdm_vals, rho_bcm = self._compute_density_profiles(norm_bgas, norm_egas, norm_cgal, norm_rdm_fixed_xi)
        
        # Calculate mass profiles
        M_dmo, M_bkg, M_rdm, M_bgas, M_egas, M_cgal, M_bcm = self._compute_mass_profiles(rho_dmo_vals, rho_bkg_vals, y_rdm_vals, y_bgas_vals, y_egas_vals, y_cgal_vals, rho_bcm, M_nfw)
        
        #M_b = M_bgas + M_egas + M_cgal
        #M_i = M_nfw
        # Calculate RDM profile
        #y_rdm_vals2, M_rdm2 = self._calculate_rdm(M_i, M_b)
        
        # Calculate displacement
        f_inv_bcm = interp1d(M_bcm, self.r_vals, bounds_error=False, fill_value="extrapolate")
        disp = self._compute_displacement(M_dmo, f_inv_bcm)        
        
        # Store results in the components dictionary
        self.components = {
            'M200': self.M200,
            'r200': self.r200,
            'r_s': self.r_s,
            'rho_dmo': rho_dmo_vals,
            'rho_bcm': rho_bcm,
            'rho_bkg': rho_bkg_vals,
            'rdm': y_rdm_vals,
            'bgas': y_bgas_vals,
            'egas': y_egas_vals,
            'cgal': y_cgal_vals,
            'M_dmo': M_dmo,
            'M_rdm': M_rdm,
            'M_bgas': M_bgas,
            'M_egas': M_egas,
            'M_cgal': M_cgal,
            'M_bcm': M_bcm,
            'M_bkg': M_bkg,
            'M_nfw': M_nfw,
            'disp': disp
        }
        
        return self
    
    def _calc_NFW_target_mass(self, inf = True):
        """
        Calculate the target mass for the NFW profile.
        
        This method integrates the NFW profile over the entire radius range
        to determine the total mass of the halo.
        
        Returns
        -------
        numpy.ndarray
            Array of cumulative NFW masses at each radius.
            
        Notes
        -----
        Also sets the class attributes rho0 and fixed_M_tot.
        """

        # Integrate NFW profile over a large range to approximate total mass
        self.rho0 = ut.bracket_rho0(self.M200, self.r_s, self.r_tr, self.r200)
        rho_nfw = np.array([dp.rho_nfw(r, self.r_s, self.rho0, self.r_tr) for r in self.r_vals])
        self.y_target = rho_nfw
        M_nfw = dp.mass_profile(self.rmax,dp.rho_nfw,r_s = self.r_s, rho0 = self.rho0, r_tr=self.r_tr)
        M_nfw2 = ut.cumul_mass(self.r_vals, rho_nfw)[-1]
        M_tot = M_nfw
        print(f"Compare M_nfw:old:{M_nfw2}\n new:{M_nfw}")
        #rho_nfw2 = np.array([dp.rho_nfw(r, self.r_s, self.rho0, self.r_tr) for r in #self.r_vals])
        #M_nfw2 = ut.cumul_mass(self.r_vals, rho_nfw2)
        #M_tot2 = dp.M_tot_truncated(self.rho0,self.r_s, self.r_tr)
        #print(f"Fixed M_tot: {M_tot:.3e}, M_tot2: {M_tot2:.3e}")
        self.fixed_M_tot = M_tot
        return M_nfw
    
    def _calculate_normalizations(self):
        """
        Calculate normalization factors for density components.
        
        This method computes the normalization constants needed for each
        component of the BCM to contain the correct fraction of the total mass.
        
        Returns
        -------
        tuple of float
            Normalization constants for (bgas, egas, cgal, rdm).
        """
        mass = self.fixed_M_tot
        #mass = self.M200
        
        norm_bgas = ut.normalize_component(
            lambda r, r_s, r200, y0, c, rho0, r_tr: dp.y_bgas(r, r_s, r200, y0, c, rho0, r_tr), 
            (self.r_s, self.r200, 1.0, self.c, self.rho0, self.r_tr), self.f_bgas * mass, self.r200
        )
        norm_egas = ut.normalize_component(
            lambda r, M_tot, r_ej: dp.y_egas(r, M_tot, r_ej), 
            (1.0, self.r_ej), self.f_egas * mass, self.r200
        )
        norm_cgal = ut.normalize_component(
            lambda r, M_tot, R_h: dp.y_cgal(r, M_tot, R_h), 
            (1.0, self.R_h), self.f_cgal * mass, self.r200
        )
        norm_yrdm_fixed_xi = ut.normalize_component(
            lambda r, r_s, rho0, r_tr, xi: dp.y_rdm_fixed_xi(r, r_s, rho0, r_tr, xi), 
            (self.r_s, self.rho0, self.r_tr, 0.85), self.f_rdm * mass, self.r200
        )
        
        return norm_bgas, norm_egas, norm_cgal, norm_yrdm_fixed_xi
    
    def _compute_density_profiles(self, norm_bgas, norm_egas, norm_cgal, norm_rdm_fixed_xi):
        """
        Compute density profiles for all components.
        
        This method calculates the density profiles for each BCM component
        using the provided normalization constants.
        
        Parameters
        ----------
        norm_bgas : float
            Normalization constant for baryonic gas.
        norm_egas : float
            Normalization constant for ejected gas.
        norm_cgal : float
            Normalization constant for central galaxy.
        norm_rdm_fixed_xi : float
            Normalization constant for remaining dark matter.
        M_nfw : numpy.ndarray
            Cumulative mass profile of the NFW halo.
            
        Returns
        -------
        tuple
            All density profiles: (rho_dmo, rho_nfw, rho_bkg, y_bgas, y_egas, y_cgal, y_rdm, rho_bcm).
            
        Notes
        -----
        Also sets the class attribute profiles with all calculated profiles.
        """        
        rho_dmo_vals = np.array([dp.rho_nfw(r, self.r_s, self.rho0, self.r_tr) + 
                               dp.rho_background(r, 1) for r in self.r_vals])
        rho_nfw_vals = np.array([dp.rho_nfw(r, self.r_s, self.rho0, self.r_tr) 
                               for r in self.r_vals])
        rho_bkg_vals = np.array([dp.rho_background(r, 1) for r in self.r_vals])
        
        y_bgas_vals = np.array([dp.y_bgas(r, self.r_s, self.r200, norm_bgas, self.c, self.rho0, self.r_tr) 
                              for r in self.r_vals])
        y_egas_vals = np.array([dp.y_egas(r, norm_egas, self.r_ej) 
                              for r in self.r_vals])
        y_cgal_vals = np.array([dp.y_cgal(r, norm_cgal, self.R_h) 
                              for r in self.r_vals])
        y_rdm_vals_fixed_xi = np.array([dp.y_rdm_fixed_xi(r, self.r_s, self.rho0, self.r_tr, norm_rdm_fixed_xi)
                                for r in self.r_vals])
        
        fractions = [self.f_bgas, self.f_egas, self.f_cgal,self.f_rdm] 
        profiles = [y_bgas_vals, y_egas_vals, y_cgal_vals, y_rdm_vals_fixed_xi]

        new_profiles = self._correction_factors_baryons(
            fractions, 
            profiles
        )
        for old, new ,frac in zip(profiles, new_profiles,fractions):
            if self.verbose:
                print(f"Old profile: {old[0]:.3e}, New profile: {new[0]:.3e}")

        y_bgas_vals2, y_egas_vals2, y_cgal_vals2, y_rdm_vals_fixed_xi2 = new_profiles
        y_bgas_vals, y_egas_vals, y_cgal_vals = new_profiles[:3]

        #y_bgas_vals2, y_cgal_vals2,y_egas_vals2,y_rdm_vals_fixed_xi2 = self.#correction_factors_baryons(
        #    [self.f_bgas, self.f_cgal,self.f_egas,self.f_rdm], 
        #    [y_bgas_vals, y_cgal_vals, y_egas_vals, y_rdm_vals_fixed_xi],inf=False
        #)
        
        #print(f"Comparison between bgas normalization at r_200 and infinity:")
        #print(f"  r_200: {ut.cumul_mass_single(self.r_vals[-1],y_bgas_vals2,self.r_vals):.3e}, inf: {ut.cumul_mass_single(self.r_vals[-1],y_bgas_vals,self.r_vals):.3e}")
        #y_cgal_vals = y_cgal_vals2
        #y_egas_vals = y_egas_vals2
        #y_bgas_vals = y_bgas_vals2
        # baryon components for xi
        # Note: y_bgas, y_egas, and y_cgal are already normalized
        baryons = [(self.r_vals, y_cgal_vals), 
            (self.r_vals, y_bgas_vals),
            (self.r_vals, y_egas_vals)
            ]
        
        # Calculate unnormalized profile
        rho_dm_contracted = dp.y_rdm_ac(self.r_vals, self.r_s, self.rho0, self.r_tr, 
                                    norm=1.0, a=0.68, f_cdm=0.839, 
                                    baryon_components=baryons, verbose=self.verbose)
        
        # Calculate total mass and correction factor
        M_contracted_inf = ut.cumul_mass(self.r_vals, rho_dm_contracted)[-1]
        #M_contracted_inf = new_mass[-1]
        
        target_mass = self.f_rdm * self.fixed_M_tot
        correction_factor = target_mass / M_contracted_inf
        
        if self.verbose:
            print(f"RDM mass correction factor: {correction_factor:.4f}")

        # Apply correction
        rho_dm_contracted *= correction_factor
        #new_mass *= correction_factor
        y_rdm_vals = rho_dm_contracted
        
        # Alternatively, use the fixed xi profile
        #y_rdm_vals = y_rdm_vals_fixed_xi

        rho_bcm = y_rdm_vals + y_bgas_vals + y_egas_vals + y_cgal_vals + rho_bkg_vals

        def comp_mass(rho1, rho2, frac):
            M1 = ut.cumul_mass(self.r_vals, rho1)[-1]
            M2 = ut.cumul_mass(self.r_vals, rho2)[-1]
            #print(f"Mass of component 1: {M1:.3e}, Mass of component 2: {M2:.3e}, should be {frac * self.fixed_M_tot:.3e}")

        for rho, rho2, frac in zip([y_bgas_vals, y_egas_vals, y_cgal_vals, y_rdm_vals],
                                    [y_bgas_vals2, y_egas_vals2, y_cgal_vals2, y_rdm_vals],
                                    [self.f_bgas, self.f_egas, self.f_cgal, self.f_rdm]):
            comp_mass(rho, rho2, frac)

        
        self.profiles = {
            'rho_dmo': rho_dmo_vals,
            'rho_nfw': rho_nfw_vals,
            'rho_bkg': rho_bkg_vals,
            'y_bgas': y_bgas_vals,
            'y_egas': y_egas_vals,
            'y_cgal': y_cgal_vals,
            'y_rdm': y_rdm_vals,
            'rho_bcm': rho_bcm,
        }
        
        return rho_dmo_vals, rho_nfw_vals, rho_bkg_vals, y_bgas_vals, y_egas_vals, y_cgal_vals, y_rdm_vals, rho_bcm
    
    def _compute_mass_profiles(self, rho_dmo_vals, rho_bkg_vals, y_rdm_vals, y_bgas_vals, y_egas_vals, y_cgal_vals, rho_bcm, M_nfw):
        """
        Compute cumulative mass profiles for all components.
        
        This method integrates the density profiles to obtain the enclosed
        mass as a function of radius for each component.
        
        Parameters
        ----------
        rho_dmo_vals : numpy.ndarray
            DMO density profile.
        rho_bkg_vals : numpy.ndarray
            Background density profile.
        y_rdm_vals : numpy.ndarray
            Remaining dark matter density profile.
        y_bgas_vals : numpy.ndarray
            Baryonic gas density profile.
        y_egas_vals : numpy.ndarray
            Ejected gas density profile.
        y_cgal_vals : numpy.ndarray
            Central galaxy density profile.
        rho_bcm : numpy.ndarray
            Total BCM density profile.
        M_nfw : numpy.ndarray
            Cumulative NFW mass profile.
            
        Returns
        -------
        tuple
            All mass profiles: (M_dmo, M_bkg, M_rdm, M_bgas, M_egas, M_cgal, M_bcm).
            
        Notes
        -----
        Also sets the class attribute masses with all calculated mass profiles.
        Calls _check_masses to validate the results and _print_masses_at_infinity
        if verbose is True.
        """
        M_dmo = ut.cumul_mass(self.r_vals, rho_dmo_vals)
        M_bkg = ut.cumul_mass(self.r_vals, rho_bkg_vals)
        M_rdm = ut.cumul_mass(self.r_vals, y_rdm_vals)
        M_bgas = ut.cumul_mass(self.r_vals, y_bgas_vals)
        M_egas = ut.cumul_mass(self.r_vals, y_egas_vals)
        M_cgal = ut.cumul_mass(self.r_vals, y_cgal_vals)
        M_bcm = ut.cumul_mass(self.r_vals, rho_bcm)
        self.masses = {
            'M_dmo': M_dmo,
            'M_bkg': M_bkg,
            'M_rdm': M_rdm,
            'M_bgas': M_bgas,
            'M_egas': M_egas,
            'M_cgal': M_cgal,
            'M_bcm': M_bcm,
            'M_nfw': M_nfw,
        }
        
        return M_dmo, M_bkg, M_rdm, M_bgas, M_egas, M_cgal, M_bcm
    
    def _correction_factors_baryons(self, fractions, profiles,inf=True):
        """
        Apply mass correction factors to baryonic components.
        
        This method adjusts the density profiles to ensure each component
        contains the exact fraction of the total mass.
        
        Parameters
        ----------
        fractions : list of float
            List of mass fractions [f_rdm, f_bgas, f_cgal, f_egas].
        profiles : list of numpy.ndarray
            List of density profiles [rdm, bgas, egas, cgal].
            
        Returns
        -------
        list of numpy.ndarray
            Corrected density profiles.
        """
        cor_profiles = []
        for i in range(len(fractions)):
            mass = ut.cumul_mass(self.r_vals, profiles[i])[-1]
            # guard against zero/invalid mass to avoid inf/NaN
            if (not np.isfinite(mass)) or (mass <= 0.0):
                if True:
                    print(f"Warning: component {i} mass within r200 is non-finite or <=0 ({mass}). Zeroing corrected profile.")
                cor_profiles.append(np.zeros_like(profiles[i]))
                continue
            correction = (fractions[i] * self.fixed_M_tot) / mass
            cor_profiles.append(correction*profiles[i])
        return cor_profiles
    
    def _compute_displacement(self, M_dmo, f_inv_bcm):
        """
        Compute displacement field for particles.
        
        This method calculates how much each particle needs to be moved
        to transform the DMO density profile into the BCM profile.
        
        Parameters
        ----------
        M_dmo : numpy.ndarray
            DMO cumulative mass profile.
        f_inv_bcm : callable
            Function that returns radius given a mass value.
            
        Returns
        -------
        numpy.ndarray
            Displacement values at each radius.
        """
        disp = np.zeros_like(self.r_vals)
        for i, r in enumerate(self.r_vals):
            M_target = M_dmo[i]
            r_bcm_val = f_inv_bcm(M_target)
            disp[i] = r_bcm_val - r
        return disp