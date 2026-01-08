import numpy as np
import matplotlib.pyplot as plt
import schneider_comparison as sim
from params import DEFAULTS, CASE_PARAMS
import argparse
import sys

def verify_schneider(verbose=False, save_path="schneider_match.png"):
    """
    Verify BCM implementation against Schneider & Teyssier 2016 Figure 1.
    
    Reproduces the three test cases (a, b, c) from the original paper, showing:
    - Row 1: Density profiles for all components
    - Row 2: Cumulative mass profiles
    - Row 3: Particle displacement functions
    
    Parameters
    ----------
    verbose : bool, optional
        Print detailed calculation information (default: False)
    save_path : str, optional
        Output figure filename (default: "schneider_match.png")
    """
    print("=" * 70)
    print("Verifying BCM Implementation")
    print("Target: Schneider & Teyssier 2016 Figure 1")
    print("=" * 70)
    
    # Initialize model
    bcm = sim.SchneiderExamples(verbose=verbose)
    cases = list(CASE_PARAMS.keys())
    
    # Create figure with 3 rows (density, mass, displacement) × 3 columns (cases a, b, c)
    fig, axes = plt.subplots(3, len(cases), figsize=(18, 18))
    
    # Process each case
    for col, case_name in enumerate(cases):
        print(f"\n--- Case ({case_name}) ---")
        print(f"Fractions: {CASE_PARAMS[case_name]}")
        
        # Calculate BCM profiles
        bcm.calculate_single_case(CASE_PARAMS[case_name])
        
        # Plot results
        _plot_density_profiles(axes[0, col], bcm, case_name)
        _plot_mass_profiles(axes[1, col], bcm, case_name)
        _plot_displacement(axes[2, col], bcm, case_name)
    
    # Finalize figure
    fig.suptitle(
        "BCM Verification: Cases (a, b, c) from Schneider & Teyssier 2016", 
        fontsize=20, 
        fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print("\n" + "=" * 70)
    print(f"✓ Verification plot saved: {save_path}")
    print("=" * 70)

def _plot_density_profiles(ax, bcm, case_name):
    """Plot density profiles for all BCM components."""
    comp = bcm.components
    r = bcm.r_vals
    
    # Plot each component
    ax.loglog(r, comp['rho_dmo'], 'b-', lw=2, label='DM-only (NFW+Background)')
    ax.loglog(r, comp['rdm'], 'b--', alpha=0.8, label='Relaxed DM')
    ax.loglog(r, comp['bgas'], 'g--', alpha=0.8, label='Bound gas')
    ax.loglog(r, comp['egas'], 'r--', alpha=0.8, label='Ejected gas')
    ax.loglog(r, comp['cgal'], 'm--', alpha=0.8, label='Central galaxy')
    ax.loglog(r, comp['rho_bkg'], 'y--', alpha=0.6, label='Background')
    ax.loglog(r, comp['rho_bcm'], 'r-', lw=3, label='BCM Total')
    
    # Reference lines
    ax.axvline(bcm.r200, color='gray', linestyle=':', lw=1.5, label='r₂₀₀')
    ax.axvline(comp['r_s'], color='gray', linestyle='--', lw=1.5, label='rₛ')
    
    # Formatting
    ax.set_xlabel("Radius [Mpc/h]", fontsize=11)
    ax.set_ylabel("Density [M☉/h/Mpc³]", fontsize=11)
    ax.set_title(f"Case ({case_name}): Density Profiles", fontsize=12, fontweight='bold')
    ax.set_xlim(1e-2, 3e1)
    ax.set_ylim(2e9, 7e16)
    ax.legend(fontsize=7, loc='best', framealpha=0.9)
    ax.grid(True, which="both", ls=":", alpha=0.3)

def _plot_mass_profiles(ax, bcm, case_name):
    """Plot cumulative mass profiles for all BCM components."""
    comp = bcm.components
    r = bcm.r_vals
    
    # Plot each component
    ax.loglog(r, comp['M_dmo'], 'b-', lw=2, label='DM-only')
    ax.loglog(r, comp['M_rdm'], 'b--', alpha=0.8, label='Relaxed DM')
    ax.loglog(r, comp['M_bgas'], 'g--', alpha=0.8, label='Bound gas')
    ax.loglog(r, comp['M_egas'], 'r--', alpha=0.8, label='Ejected gas')
    ax.loglog(r, comp['M_cgal'], 'm--', alpha=0.8, label='Central galaxy')
    ax.loglog(r, comp['M_bkg'], 'y--', alpha=0.6, label='Background')
    ax.loglog(r, comp['M_bcm'], 'r-', lw=3, label='BCM Total')
    
    # Reference line
    ax.axvline(bcm.r200, color='gray', linestyle=':', lw=1.5, label='r₂₀₀')
    
    # Formatting
    ax.set_xlabel("Radius [Mpc/h]", fontsize=11)
    ax.set_ylabel("Cumulative Mass [M☉/h]", fontsize=11)
    ax.set_title(f"Case ({case_name}): Mass Profiles", fontsize=12, fontweight='bold')
    ax.set_xlim(1e-2, 1e2)
    ax.set_ylim(7e11, 7e15)
    ax.legend(fontsize=7, loc='best', framealpha=0.9)
    ax.grid(True, which="both", ls=":", alpha=0.3)

def _plot_displacement(ax, bcm, case_name):
    """Plot particle displacement function."""
    comp = bcm.components
    r = bcm.r_vals
    disp = comp['disp']
    
    # Separate positive and negative displacements
    disp_positive = np.where(disp > 0, disp, np.nan)
    disp_negative = np.where(disp < 0, -disp, np.nan)
    
    # Plot displacements
    ax.loglog(r, disp_positive, 'b-', lw=2, label='Outward (positive)')
    ax.loglog(r, disp_negative, 'b--', lw=2, label='Inward (negative)')
    
    # Reference line
    ax.axvline(bcm.r200, color='gray', linestyle=':', lw=1.5, label='r₂₀₀')
    
    # Formatting
    ax.set_xlabel("Radius [Mpc/h]", fontsize=11)
    ax.set_ylabel("|Displacement| [Mpc/h]", fontsize=11)
    ax.set_title(f"Case ({case_name}): Displacement", fontsize=12, fontweight='bold')
    ax.set_xlim(1e-2, 1e2)
    ax.set_ylim(1e-4, 1.1)
    ax.legend(fontsize=7, loc='best', framealpha=0.9)
    ax.grid(True, which="both", ls=":", alpha=0.3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BCM Schneider verification")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("-o", "--output", default="schneider_match.png", help="Output filename")
    args = parser.parse_args()

    print("\nRunning Schneider & Teyssier 2015 verification...")
    verify_schneider(verbose=args.verbose, save_path=args.output)
    sys.exit(0)