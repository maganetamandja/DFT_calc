from pyscf import __config__
__config__.B3LYP_WITH_VWN5 = False
import pyscf.pbc.gto as pbcgto

from pyscf import  dft

import numpy as np


def simple_cubic():

    cell = pbcgto.Cell()
    cell.atom = 'Pt 0 0 0' # Example: Pt atom
    a = 3.9242
    cell.a = [[a, 0, 0], # Lattice vectors
            [0, a, 0],
            [0, 0, a]]
    cell.basis = 'LanL2DZ'
    cell.build()

    mf = dft.RKS(cell)
    mf.xc = 'B3LYP'
    mf.kernel()


    print("Data for face simple cubic")
    print("Total energy:", mf.e_tot)
    print("Orbital energies:", mf.mo_energy)
    print("Mulliken populations:", mf.mulliken_pop())

def face_centered_cubic():
    cell = pbcgto.Cell()
    cell.atom = 'Pt 1 1 1' # Example: Copper atom
    a = 3.9242 # Lattice constant for Oxygen https://www.webelements.com/oxygen/crystal_structure.html
    cell.a = [[0, a/2., a/2.], # Lattice vectors
          [a/2., 0, a/2.],
          [a/2., a/2., 0]]
    cell.basis = 'LanL2DZ'
    cell.build()

    mf = dft.RKS(cell)
    mf.xc = 'B3LYP'
    mf.kernel()


    print("Data for face centered cubic")
    print("Total energy:", mf.e_tot)
    print("Orbital energies:", mf.mo_energy)
    print("Mulliken populations:", mf.mulliken_pop())

def hexagonal_close_packed():

    cell = pbcgto.Cell()
    cell.atom = 'Pt 0 0 0; O 1/3 2/3 1/2' # Example: Magnesium atoms
    a = 3.9242 # Lattice constant 'a' for oxigen
    c = 3.9242 # Lattice constant 'c' for oxygen
    cell.a = [[a, 0, 0], # Lattice vectors
            [-a/2., np.sqrt(3)/2.*a, 0],
            [0, 0, c]]
    cell.basis = 'LanL2DZ'
    cell.build()

    mf = dft.RKS(cell)
    mf.xc = 'B3LYP'
    mf.kernel()

    print("Data for Hexagonal close packed ")
    print("Total energy:", mf.e_tot)
    print("Orbital energies:", mf.mo_energy)
    print("Mulliken populations:", mf.mulliken_pop())

print(simple_cubic())
print(face_centered_cubic())
print(hexagonal_close_packed())