import sys
import glob

import triqs.utility.mpi as mpi

from triqs_dft_tools.sumk_dft import SumkDFT
from triqs_dft_tools.sumk_dft_tools import SumkDFTTools
from h5 import HDFArchive
from triqs.gf import *


h5_file = glob.glob(sys.argv[1])[0]

sum_k = SumkDFTTools(hdf_file = h5_file)


zero_Sigma_w = [sum_k.block_structure.create_gf(ish=iineq, gf_function=GfReFreq,
                                                window = [-4,4],
                                                n_points = 1001)
                for iineq in range(sum_k.n_inequiv_shells)]
sum_k.put_Sigma(zero_Sigma_w)

sum_k.calc_mu()

G_loc_all = sum_k.extract_G_loc(iw_or_w='w', broadening = 0.01, transform_to_solver_blocks=False)
dens_mat = [G_loc_all[iineq].density() for iineq in range(sum_k.n_inequiv_shells)]


DOS, DOSproj, DOSproj_orb = sum_k.dos_wannier_basis(broadening=0.01, with_Sigma=True, with_dc=False, save_to_file=True)

if mpi.is_master_node():
    with HDFArchive(h5_file,'a') as h5:
        h5['A_w'] = DOS
        h5['A_w_proj'] = DOSproj
        h5['A_w_proj_orb'] = DOSproj_orb

