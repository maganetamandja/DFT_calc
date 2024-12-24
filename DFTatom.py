from pyscf import gto, dft

my_atom = gto.Mole()
my_atom.atom = 'O 0 0 0'  # Oxygen atom at the origin
my_atom.basis = '6-31g'
my_atom.build()

mf = dft.RKS(my_atom)
mf.xc = 'B3LYP'
mf.kernel()

print("Total energy:", mf.e_tot)
print("Orbital energies:", mf.mo_energy)
print("Mulliken populations:", mf.mulliken_pop())