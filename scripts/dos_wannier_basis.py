import sys
import glob

import triqs.utility.mpi as mpi

from triqs_dft_tools.sumk_dft import SumkDFT
from triqs_dft_tools.sumk_dft_tools import SumkDFTTools
from h5 import HDFArchive
from triqs.gf import *


h5_file = glob.glob(sys.argv[1])[0]
# extract Aw from Sigma
# h5_file = '/home/alex/work/molybdenates/SrMoO3/subspace_comp/pd_t2g_U3_J0.5_Held_ndmft/vasp.h5'

with HDFArchive(h5_file,'r') as h5:
    Sigma_w = h5['DMFT_results']['last_iter']['Sigma_w_0']
    # Sigma_w_2 = h5['DMFT_results']['last_iter']['Sigma_w_1']
    chemical_potential =  h5['DMFT_results']['last_iter']['chemical_potential']
    dc_imp = h5['DMFT_results']['last_iter']['DC_pot']
    dc_energ = h5['DMFT_results']['last_iter']['DC_energ']
    block_structure = h5['DMFT_input']['block_structure']

sum_k = SumkDFTTools(hdf_file = h5_file)


corr_to_inequiv = sum_k.corr_to_inequiv

sum_k.block_structure = block_structure

sum_k.corr_to_inequiv = corr_to_inequiv
sum_k.set_Sigma([Sigma_w])
sum_k.set_mu(chemical_potential)
sum_k.set_dc(dc_imp,dc_energ)

# print sum_k.Sigma_imp_w

DOS, DOSproj, DOSproj_orb = sum_k.dos_wannier_basis(broadening=0.01, with_Sigma=True, with_dc=True, save_to_file=True)

# if mpi.is_master_node():
    # with HDFArchive(h5_file,'a') as h5:
        # h5['DMFT_results']['last_iter']['A_w'] = DOS
        # h5['DMFT_results']['last_iter']['A_w_proj'] = DOSproj
        # h5['DMFT_results']['last_iter']['A_w_proj_orb'] = DOSproj_orb

