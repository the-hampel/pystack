#!/usr/bin/python3
# pyright: reportUnusedExpression=false

from timeit import default_timer as timer
import numpy as np
from scipy.optimize import brentq

# triqs imports
from h5 import HDFArchive
from triqs.utility import mpi

from triqs.gf import MeshDLRImFreq, Gf, MeshImFreq
from triqs.lattice.utils import TB_from_wannier90
from triqs.plot.mpl_interface import oplot, plt

np.set_printoptions(precision=4, suppress=True, linewidth=200)

# ------------------------------#
# parameters
w90_seed = 'lvo_rot'
w90_path = './'
n_orb = 12
sites = 1
n_elect = 3.0
spin = None

k_dim = 21
beta = 1000.0
# dlr_wmax = [1,2,4,5,8,10,12,15,20,25,30,40,50,80,100]
dlr_wmax = [5]
dlr_eps = 1e-10

# window to seach for mu
search_win = [8, 12]

store = f'b{beta:.0f}_k{k_dim:.0f}_res.h5'
# ------------------------------#

if spin:
    spin_factor = 1
    n_orb = 2 * n_orb
else:
    spin_factor = 2


def calc_Gloc(mu, w_mat, w_k, h_k_mat_slice, S_dlr_iw):
    Gloc = S_dlr_iw.copy()
    Gloc.zero()

    mu_mat = mu * np.eye(n_orb)
    # loop over slice on each rank and calc Gloc contribution
    for h_k_mat in h_k_mat_slice:
        Gloc.data[:, :, :] += wk * np.linalg.inv(w_mat + mu_mat[None, ...] - h_k_mat[None, ...] - S_dlr_iw.data)

    # gather results
    Gloc << mpi.all_reduce(Gloc)

    return Gloc

def calc_target_dens(mu, w_mat, w_k, h_k_mat_slice, S_dlr_iw, n_elect):

    dens = calc_Gloc(mu, w_mat, w_k, h_k_mat_slice, S_dlr_iw).total_density().real

    mpi.report(f'mu: {mu:8.8f} dens: {dens:7.8f}')
    return dens - n_elect

def calc_mu(n_elect, S_reg, dlr_wmax, dlr_eps, beta, h_k_mat_slice, w_k):

    # omega mesh, mu, and eta matrix vectors for each frequency
    w_mesh = MeshDLRImFreq(beta=beta, statistic='Fermion', w_max=dlr_wmax, eps=dlr_eps)
    # comment this in if you want to use a regular mesh
    # w_mesh = MeshImFreq(beta=beta, statistic='Fermion', n_iw=5000)

    w_mesh_arr = np.array([iwn.value for iwn in w_mesh.values()])
    mpi.report(f'Largest DLR frequency: {w_mesh_arr[-1].imag:.4f}')
    w_mat = np.array([w.value * np.eye(n_orb) for w in w_mesh])

    # local Gf for spectral function
    Gloc = Gf(mesh=w_mesh, target_shape=[n_orb, n_orb])

    # S_dlr_iw = Gloc.copy()
    S_dlr_iw = S_reg

    # for iwn in w_mesh:
    #     S_dlr_iw[iwn] = S_reg(iwn.value)

    # find mu
    time_start_mu = timer()
    mu = brentq(calc_target_dens, search_win[0], search_win[1], (w_mat, w_k, h_k_mat_slice, S_dlr_iw, n_elect), xtol=1e-8)

    mpi.barrier()
    mpi.report('total time for calc_mu: {:.2f} s'.format(timer() - time_start_mu))

    mpi.report(f'#####\nfinal μ={mu:.5f} for n={n_elect:.3f}\n#####\n')

    Gloc = calc_Gloc(mu, w_mat, w_k, h_k_mat_slice, S_dlr_iw)

    if mpi.is_master_node():
        tot_dens = Gloc.total_density().real
        dens_mat = Gloc.density().real
        print(f'total density: {tot_dens:.4f}')
        print(np.diag(dens_mat))
        print('density matrix:')
        print(dens_mat)

        if store:
            with HDFArchive(store, 'a') as h5:
                h5['Gloc'] = Gloc

    return mu, Gloc


start_time = timer()

# load wannier90 file and create k_mesh
TB = TB_from_wannier90(seed=w90_seed, path=w90_path, extend_to_spin=spin)

d_start_time = timer()

k_spacing = np.linspace(0, 1, k_dim, endpoint=False)
k_array = np.array(np.meshgrid(k_spacing, k_spacing, k_spacing)).T.reshape(-1, 3)
n_k = k_dim**3

# load self-energy data
# s_xy = np.loadtxt('sigma_xy.dat')
# s_xz = np.loadtxt('sigma_xz.dat')
# a1g = np.loadtxt('S_a1g_b1000.dat')
# egpi = np.loadtxt('S_egpi_b1000.dat')
a1g = np.load('S_a1g_b1000_dlr.npy', allow_pickle=True)
egpi = np.load('S_egpi_b1000_dlr.npy', allow_pickle=True)

# S_reg = Gf(mesh=MeshImFreq(beta, 'Fermion', a1g.shape[0]//2), target_shape=[n_orb, n_orb])
S_reg = Gf(mesh=MeshDLRImFreq(beta, 'Fermion', dlr_wmax[0], dlr_eps), target_shape=[n_orb, n_orb])
# mpi.report(f'Largest reg mesh frequency: {a1g[-1,0]:.4f}')

# mid = len(S_reg.mesh) // 2

# fill Sigma from file
# S_reg.data[:, 0, 0] = a1g[:, 1] + 1j * a1g[:, 2]
# S_reg.data[:, 1, 1] = a1g[:, 1] + 1j * a1g[:, 2]
# S_reg.data[:, 2, 2] = a1g[:, 1] + 1j * a1g[:, 2]
# S_reg.data[:, 3, 3] = a1g[:, 1] + 1j * a1g[:, 2]
# S_reg.data[:, 4, 4] = egpi[:, 1] + 1j * egpi[:, 2]
# S_reg.data[:, 5, 5] = egpi[:, 1] + 1j * egpi[:, 2]
# S_reg.data[:, 6, 6] = egpi[:, 1] + 1j * egpi[:, 2]
# S_reg.data[:, 7, 7] = egpi[:, 1] + 1j * egpi[:, 2]
# S_reg.data[:, 8, 8] = egpi[:, 1] + 1j * egpi[:, 2]
# S_reg.data[:, 9, 9] =  egpi[:, 1] + 1j * egpi[:, 2]
# S_reg.data[:, 10, 10] = egpi[:, 1] + 1j * egpi[:, 2]
# S_reg.data[:, 11, 11] = egpi[:, 1] + 1j * egpi[:, 2]
for i in range(4):
    S_reg.data[:, i, i] = a1g[:,0,0]
for i in range(4, 12):
    S_reg.data[:, i, i] = egpi[:,0,0]

# Subtract dc_pot
# S_reg -= 2.8301177882202526 # beta=40
S_reg -= 2.8200574314727165

# S_reg.data[:mid, 0, 0] = s_xz[::-1, 1] - 1j * s_xz[::-1, 2]
# S_reg.data[:mid, 1, 1] = s_xz[::-1, 1] - 1j * s_xz[::-1, 2]
# S_reg.data[:mid, 2, 2] = s_xy[::-1, 1] - 1j * s_xy[::-1, 2]

mpi.report('time for setup: {:.2f} s'.format(timer() - start_time))

trafo_start_time = timer()

k_array_slice = mpi.slice_array(k_array)
h_k_mat_slice = TB.fourier(k_array_slice)
wk = 1 / n_k

mpi.barrier()
mpi.report('time for Fourier trafo to H(k): {:.2f} s'.format(timer() - trafo_start_time))

if mpi.is_master_node():
    with open('wmax_scan.dat', 'w') as f:
        f.write('# wmax mu\n')

for wmax in dlr_wmax:
    mpi.report(f'DLR wmax={wmax}')
    mu, Gloc = calc_mu(n_elect, S_reg, wmax, dlr_eps, beta, h_k_mat_slice, wk)

    if mpi.is_master_node():
        with open('wmax_scan.dat', 'a') as f:
            f.write(f'{wmax:.1f} {mu:.6f}\n')

mpi.barrier()
mpi.report('overall time: {:.2f} s'.format(timer() - start_time))
