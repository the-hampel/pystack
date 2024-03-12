#!/bin/python
import numpy as np
from scipy.optimize import brentq

from timeit import default_timer as timer
from triqs_dft_tools.sumk_dft import *
from triqs_dft_tools.sumk_dft_tools import *
from h5 import HDFArchive
from triqs.lattice.utils import TB_from_wannier90, k_space_path
from triqs.sumk import SumkDiscreteFromLattice


# --------------------------------------------------------------------------------------------

def lambda_matrix(lam_xy, lam_z): # This is the definition by Hugo, Robert, Gernot
    lam_loc = np.zeros((6,6),dtype=complex)
    lam_loc[0,4] =  -1j*lam_xy/2.0
    lam_loc[0,5] =     lam_xy/2.0
    lam_loc[1,2] =  -1j*lam_z/2.0
    lam_loc[1,3] =   1j*lam_xy/2.0
    lam_loc[2,3] =    -lam_xy/2.0
    lam_loc[4,5] =   1j*lam_z/2.0
    lam_loc = lam_loc + np.transpose(np.conjugate(lam_loc))
    return lam_loc

def make_sigma_block(sigma_in):

    sigma = Gf(mesh=sigma_in.mesh, target_shape=[6, 6])
    sigma.data[:,0,0] = sigma_in['up_0'].data[:,0,0]
    sigma.data[:,1,1] = sigma_in['up_1'].data[:,0,0]
    sigma.data[:,2,2] = sigma_in['up_2'].data[:,0,0]
    sigma.data[:,3,3] = sigma_in['down_0'].data[:,0,0]
    sigma.data[:,4,4] = sigma_in['down_1'].data[:,0,0]
    sigma.data[:,5,5] = sigma_in['down_2'].data[:,0,0]

    sigma_out = BlockGf(name_list=['up'], block_list=[sigma], make_copies=True)

    return sigma_out

def compute_dens(mu, sk, sigma, dc, n_elect):

    dens =  sk(mu = mu + dc, Sigma = sigma).total_density()

    mpi.report(f'mu: {mu:8.4f} n_elect: {n_elect:4.2f} dens: {dens.real:7.4f}')
    return dens.real - n_elect

# --------------------------------------------------------------------------------------------

betas = [100]
n_iw = [2525]
#n_iw = [1025] * 8
window = [-1., 1.]
n_orb = 6
n_k = 20
n_elect = 4.0
cfs = 0.03


calc_mode = 'DFT_DMFT'
calc_mode = 'DFT'
tb_mode = 'dlam'
#tb_mode = 'tb'
write_file = f'./mus_{calc_mode}_{tb_mode}_cfs{cfs:.2f}_nk{n_k}.txt'

# --------------------------------------------------------------------------------------------
# Wannier90 without SOC

seed = 'w2w'
path = '/mnt/ceph/users/sbeck/materials/sro/SRO_ARPES_T/W90_data/w2w/'

h_loc = np.zeros((n_orb, n_orb), dtype=complex)

add_cfs = np.diag([cfs/2, -cfs/2, -cfs/2] * 2)
h_loc += add_cfs

tb = TB_from_wannier90(path=path, seed=seed, extend_to_spin=True, add_local=h_loc)

d_lam_loc = lambda_matrix(0.20,0.20)
mpi.report('Lambda + dLambda Matrix: \n', d_lam_loc)
h_loc += d_lam_loc

tb_dlam = TB_from_wannier90(path=path, seed=seed, extend_to_spin=True, add_local=h_loc)
tb_use = tb_dlam if tb_mode == 'dlam' else tb

# --------------------------------------------------------------------------------------------

tot_start_time = timer()

with open(write_file, 'w') as f:
    f.write(f'# mus {calc_mode}\n')

sum_k = SumkDiscreteFromLattice(lattice=tb_use, n_points=n_k)

for ibeta, beta in enumerate(betas):

    if calc_mode == 'DFT_DMFT':

        dc_val = Sigma_block = None
        if mpi.is_master_node():
            path = f'/mnt/ceph/users/sbeck/materials/sro/unstrained/DMFT/b{np.floor(beta):.0f}/Sr2RuO4.h5'

            with HDFArchive(path, 'r') as h5:
                Sigma_avg = h5['DMFT_results']['Sigma_iw_avg']
                dc_avg = h5['DMFT_results']['dc_avg']
                mu_avg = h5['DMFT_results']['mu_avg']

            Sigma_block = make_sigma_block(Sigma_avg)
            dc_val = dc_avg[0]['up'][0,0]

        dc_val = mpi.bcast(dc_val)
        Sigma_block = mpi.bcast(Sigma_block)

    else:

        Sigma_single = Gf(mesh=MeshImFreq(beta=beta, n_iw=n_iw[ibeta], S='Fermion'), target_shape=[n_orb, n_orb])
        Sigma_block = BlockGf(name_list = ['up'], block_list = [Sigma_single], make_copies = True)
        dc_val = 0.0

    mu = brentq(compute_dens, window[0], window[1], (sum_k, Sigma_block, dc_val, n_elect), xtol=1e-8)

    if mpi.is_master_node():
        print(f'{beta}', f'{mu:.7f}')

        with open(write_file, 'a') as f:
            f.write(f'{beta} {mu:.7f}\n')

mpi.barrier()
mpi.report('total time for calc_mu: {:.2f} s'.format(timer() - tot_start_time))
