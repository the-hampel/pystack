from h5 import HDFArchive
from triqs_dft_tools.sumk_dft import SumkDFT
from triqs_dft_tools.sumk_dft_tools import SumkDFTTools
import triqs.utility.mpi as mpi

from triqs.operators import *
from triqs.gf import *
from triqs_dft_tools.converters import Wannier90Converter

import numpy as np

np.set_printoptions(precision=4, suppress= True)

if mpi.is_master_node():
    Converter = Wannier90Converter(seedname='wannier90', rot_mat_type='hloc_diag')
    Converter.convert_dft_input()

sum_k = SumkDFTTools(hdf_file = 'wannier90.h5' , use_dft_blocks = False)

sum_k.set_mu(0.016795)
# sum_k.calc_mu()

Sigma = sum_k.block_structure.create_gf(beta=100)
sum_k.put_Sigma([Sigma])
Gloc = sum_k.extract_G_loc()[0]

density_shell = np.real(Gloc.total_density())
mpi.report('Total charge of impurity problem = {:.4f}'.format(density_shell))
density_mat = Gloc.density()
mpi.report('Density matrix:')
for key, value in density_mat.items():
    mpi.report(key)
    mpi.report(np.real(value))

mpi.report('calculating local Hamiltonian')
atomic_levels = sum_k.eff_atomic_levels()[0]
for name, matrix in atomic_levels.items():
    mpi.report(matrix.real)

