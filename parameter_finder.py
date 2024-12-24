
from pyscf import __config__
__config__.B3LYP_WITH_VWN5 = False
import pyscf.pbc.gto as pbcgto
from pyscf.pbc import dft
from scipy.optimize import minimize_scalar
import numpy as np

def objective_function(a):
    """
    Calculate the total energy of a cubic Pt cell with lattice parameter a.
    
    Args:
        a (float): Lattice parameter in Angstroms
        
    Returns:
        float: Total energy of the system
    """
    try:
        cell = pbcgto.Cell()
        cell.atom = 'Pt 0 0 0'
        cell.a = [[a, 0, 0], [0, a, 0], [0, 0, a]]  # Simple cubic lattice
        cell.basis = 'LanL2DZ'
        cell.exp_to_discard = 0.1
        
        # Add essential parameters
        cell.pseudo = 'lanl2dz'  # Pseudopotential for Pt
        cell.dimension = 3
        cell.verbose = 0  # Reduce output
        cell.unit = 'angstrom'  # Specify units
        
        cell.build()

        mf = dft.RKS(cell)
        mf.xc = 'B3LYP'
        # Add convergence parameters
        mf.max_cycle = 100
        mf.conv_tol = 1e-6
        
        energy = mf.kernel()
        
        if not mf.converged:
            return float('inf')  # Return high value if SCF didn't converge
            
        return float(energy)
    
    except Exception as e:
        print(f"Error occurred for a = {a}: {str(e)}")
        return float('inf')  # Return high value if calculation fails

def optimize_lattice():
    """
    Optimize the lattice parameter of the cubic Pt cell.
    """
    # Set reasonable bounds for lattice parameter (in Angstroms)
    initial_guess = 4  # Close to experimental Pt lattice constant
    bounds = (3.0, 5.0)
    
    result = minimize_scalar(
        objective_function,
        bounds=bounds,
        method='bounded',
        options={'xatol': 1e-3}
    )
    
    if result.success:
        print(f"Optimization successful!")
        print(f"Optimal lattice parameter: {result.x:.4f} Angstroms")
        print(f"Final energy: {result.fun:.6f} Ha")
    else:
        print("Optimization failed!")
        print(result.message)
    
    return result


def objective_function_FCC(a):
    """
    Calculate the total energy of a cubic Pt cell with lattice parameter a.
    
    Args:
        a (float): Lattice parameter in Angstroms
        
    Returns:
        float: Total energy of the system
    """
    try:
        cell = pbcgto.Cell()
        cell.atom = 'Pt 0 0 0'
        cell.a = [[0, a/2, a/2], [a/2, 0, a/2], [a/2, a/2, 0]]  # Simple cubic lattice
        cell.basis = 'LanL2DZ'
        cell.exp_to_discard = 0.1
        
        # Add essential parameters
        cell.pseudo = 'lanl2dz'  # Pseudopotential for Pt
        cell.dimension = 3
        cell.verbose = 0  # Reduce output
        cell.unit = 'angstrom'  # Specify units
        
        cell.build()

        mf = dft.RKS(cell)
        mf.xc = 'B3LYP'
        # Add convergence parameters
        mf.max_cycle = 100
        mf.conv_tol = 1e-6
        
        energy = mf.kernel()
        
        if not mf.converged:
            return float('inf')  # Return high value if SCF didn't converge
            
        return float(energy)
    
    except Exception as e:
        print(f"Error occurred for a = {a}: {str(e)}")
        return float('inf')  # Return high value if calculation fails

def optimize_latticeFCC():
    """
    Optimize the lattice parameter of the cubic Pt cell.
    """
    # Set reasonable bounds for lattice parameter (in Angstroms)
    initial_guess = 4  # Close to experimental Pt lattice constant
    bounds = (3.0, 5.0)
    
    result = minimize_scalar(
        objective_function_FCC,
        bounds=bounds,
        method='bounded',
        options={'xatol': 1e-3}
    )
    
    if result.success:
        print(f"Optimization successful!")
        print(f"Optimal lattice parameter: {result.x:.4f} Angstroms")
        print(f"Final energy: {result.fun:.6f} Ha")
    else:
        print("Optimization failed!")
        print(result.message)
    
    return result


if __name__ == "__main__":
    result = optimize_lattice()
    result2 = optimize_latticeFCC()