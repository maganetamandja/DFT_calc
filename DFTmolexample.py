from pyscf import gto, dft

mol = gto.Mole()
mol.atom = [
    ['O', (0., 0., 0.)],
    ['H', (0., -0.757, 0.587)],
    ['H', (0., 0.757, 0.587)]]
mol.basis = '6-31g'
mol.build()

mf = dft.RKS(mol)
mf.xc = 'B3LYP'
mf.kernel()

print("Total energy:", mf.e_tot)
print("Orbital energies:", mf.mo_energy)
print("Mulliken populations:", mf.mulliken_pop())