from h5 import HDFArchive
from triqs_dft_tools.sumk_dft import SumkDFT
from triqs_dft_tools.sumk_dft_tools import SumkDFTTools
import triqs.utility.mpi as mpi

from triqs.operators import *
from triqs.gf import *
from triqs_dft_tools.converters import Wannier90Converter

import numpy as np

np.set_printoptions(precision=4, suppress=True)

################
### Params #####
seed = 'ce2o3'
store = True
################

Converter = Wannier90Converter(seedname=seed, rot_mat_type='hloc_diag')
Converter.convert_dft_input()

sum_k = SumkDFT(hdf_file=seed+'.h5', use_dft_blocks=False)

# sum_k.set_mu(14.737167)
sum_k.calc_mu()

Sigma = sum_k.block_structure.create_gf(beta=100)
sum_k.put_Sigma([Sigma])
Gloc = sum_k.extract_G_loc()

if mpi.is_master_node():
    Gloc_new = sum_k.analyse_block_structure_from_gf(Gloc, threshold=1e-04)

    for ishell, gloc in enumerate(Gloc_new):
        print(f'correlated shell {ishell}:')
        density_shell = np.real(gloc.total_density())

        print('Total charge of impurity problem = {:.4f}'.format(density_shell))
        density_mat = gloc.density()
        print('Density matrix:')
        for key, value in density_mat.items():
            print(key)
            print(np.real(value))
            if np.any(np.imag(value) > 1e-4):
                print('Im:')
                print(np.imag(value))

        print('---')
        print('calculating local Hamiltonian in block structure:')
        atomic_levels = sum_k.eff_atomic_levels()[ishell]
        solver_eal = sum_k.block_structure.convert_matrix(atomic_levels, space_from='sumk')
        for name, matrix in solver_eal.items():
            print(name)
            print(matrix.real)
            if np.any(np.imag(matrix) > 1e-4):
                print('Im:')
                print(np.imag(matrix))
        print('---------------------------------------------------')

        if store:
            with HDFArchive(seed+'.h5', 'a') as h5:
                h5['Gloc_iw'] = Gloc_new
                h5['Hloc'] = solver_eal
