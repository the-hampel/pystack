import sys
import glob

import triqs.utility.mpi as mpi

from triqs_dft_tools.sumk_dft import SumkDFT
from triqs_dft_tools.sumk_dft_tools import SumkDFTTools
from h5 import HDFArchive
from triqs.gf import *


h5_file = glob.glob(sys.argv[1])[0]

with HDFArchive(h5_file,'r') as h5:
    Sigma_w = h5['DMFT_results']['last_iter']['Sigma_freq_0']
    chemical_potential =  h5['DMFT_results']['last_iter']['chemical_potential_post']
    dc_imp = h5['DMFT_results']['last_iter']['DC_pot']
    dc_energ = h5['DMFT_results']['last_iter']['DC_energ']
    block_structure = h5['DMFT_input']['block_structure']

mesh = MeshReFreq(window=(-12,12), n_w=3001)
sum_k = SumkDFTTools(hdf_file = h5_file, mesh=mesh)

corr_to_inequiv = sum_k.corr_to_inequiv
sum_k.block_structure = block_structure

sum_k.corr_to_inequiv = corr_to_inequiv
sum_k.set_Sigma([Sigma_w])
sum_k.set_mu(chemical_potential)
sum_k.set_dc(dc_imp,dc_energ)

sum_k.calc_mu(precision=0.0001, iw_or_w='w', broadening=0.0, beta=100)

with HDFArchive(h5_file,'a') as h5:
    h5['DMFT_results']['last_iter']['chemical_potential_post'] = sum_k.chemical_potential

