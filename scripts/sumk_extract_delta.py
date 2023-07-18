import sys
import glob

import triqs.utility.mpi as mpi

from triqs_dft_tools.sumk_dft import SumkDFT
from triqs_dft_tools.sumk_dft_tools import SumkDFTTools
from h5 import HDFArchive
from triqs.gf import *


h5_file = glob.glob(sys.argv[1])[0]
eta = 0.02
site = 0 

# with HDFArchive(h5_file,'r') as h5:
#     Sigma_w = h5['DMFT_results']['last_iter']['Sigma_freq_0']
#     chemical_potential =  h5['DMFT_results']['last_iter']['chemical_potential_post']
#     dc_imp = h5['DMFT_results']['last_iter']['DC_pot']
#     dc_energ = h5['DMFT_results']['last_iter']['DC_energ']
#     block_structure = h5['DMFT_input']['block_structure']

mesh = MeshReFreq(window=(-12,12), n_w=3001)
sum_k = SumkDFTTools(hdf_file = h5_file, mesh=mesh)

chemical_potential = 10.6412

# sum_k.set_Sigma([Sigma_w])
# sum_k.set_dc(dc_imp,dc_energ)
sum_k.set_mu(chemical_potential)

# sum_k.calc_mu(precision=0.0001, iw_or_w='w', broadening=0.0)

G_loc_all = sum_k.extract_G_loc(broadening = eta, with_Sigma=False)
Sigma_freq = G_loc_all[site].copy()
Sigma_freq << 0.0+0.0j
G0_freq = Sigma_freq.copy()
Delta_freq = Sigma_freq.copy()

dens_mat = [G_loc_all[iineq].density() for iineq in range(sum_k.n_inequiv_shells)]

sumk_eal = sum_k.eff_atomic_levels()[site]
G0_freq << inverse(Sigma_freq + inverse(G_loc_all[site]))

for name, g0 in G0_freq:
    solver_eal = sum_k.block_structure.convert_matrix(sumk_eal, space_from='sumk', ish_from=sum_k.inequiv_to_corr[site])[name]
    Delta_freq[name] << Omega + 1j * eta - inverse(g0) - solver_eal

if mpi.is_master_node():
    with HDFArchive('delta_out.h5','a') as h5:
        h5['delta'] = Delta_freq
        h5['G0'] = G0_freq
        h5['Gloc'] = G_loc_all[site]
        h5['eal'] = sumk_eal
