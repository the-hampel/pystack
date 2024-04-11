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

np.set_printoptions(precision=4, suppress=True)

# ------------------------------#
# parameters
w90_seed = 'sro'
w90_path = './'
n_orb = 3
sites = 1
n_elect = 0.5
spin = None

k_dim = 21
beta = 5.0
dlr_wmax = 5
dlr_eps = 1e-8

# window to seach for mu
search_win = [10, 20]

store = f'b{beta:.0f}_k{k_dim:.0f}_res.h5'
# ------------------------------#

if spin:
    spin_factor = 1
    n_orb = 2 * n_orb
else:
    spin_factor = 2


def calc_Gloc(mu, w_mat, h_k_mat_slice, S_dlr_iw):
    Gloc = S_dlr_iw.copy()
    Gloc.zero()

    mu_mat = mu * np.eye(n_orb)
    # loop over slice on each rank and calc Gloc contribution
    for h_k_mat in h_k_mat_slice:
        Gloc.data[:, :, :] += wk * np.linalg.inv(w_mat + mu_mat[None, ...] - h_k_mat[None, ...] - S_dlr_iw.data)

    # gather results
    Gloc << mpi.all_reduce(Gloc)

    return Gloc

def calc_target_dens(mu, w_mat, h_k_mat_slice, S_dlr_iw, n_elect):

    dens = calc_Gloc(mu, w_mat, h_k_mat_slice, S_dlr_iw).total_density().real

    mpi.report(f'mu: {mu:8.8f} dens: {dens:7.8f}')
    return dens - n_elect


start_time = timer()

# load wannier90 file and create k_mesh
TB = TB_from_wannier90(seed=w90_seed, path=w90_path, extend_to_spin=spin)

d_start_time = timer()

k_spacing = np.linspace(0, 1, k_dim, endpoint=False)
k_array = np.array(np.meshgrid(k_spacing, k_spacing, k_spacing)).T.reshape(-1, 3)
n_k = k_dim**3

# # omega mesh, mu, and eta matrix vectors for each frequency
w_mesh = MeshDLRImFreq(beta=beta, statistic='Fermion', w_max=dlr_wmax, eps=dlr_eps)
# comment this in if you want to use a regular mesh
# w_mesh = MeshImFreq(beta=beta, statistic='Fermion', n_iw=501)

w_mesh_arr = np.array([iwn.value for iwn in w_mesh.values()])
n_omega = len(w_mesh)
w_mat = np.array([w.value * np.eye(n_orb) for w in w_mesh])
wk = 1 / n_k

# local Gf for spectral function
Gloc = Gf(mesh=w_mesh, target_shape=[n_orb, n_orb])

s_xy = np.loadtxt('sigma_xy.dat')
s_xz = np.loadtxt('sigma_xz.dat')

S_reg = Gf(mesh=MeshImFreq(beta, 'Fermion', s_xy.shape[0]), target_shape=[n_orb, n_orb])

mid = len(S_reg.mesh) // 2

# fill Sigma from file
S_reg.data[mid:, 0, 0] = s_xz[:, 1] + 1j * s_xz[:, 2]
S_reg.data[mid:, 1, 1] = s_xz[:, 1] + 1j * s_xz[:, 2]
S_reg.data[mid:, 2, 2] = s_xy[:, 1] + 1j * s_xy[:, 2]

S_reg.data[:mid, 0, 0] = s_xz[::-1, 1] - 1j * s_xz[::-1, 2]
S_reg.data[:mid, 1, 1] = s_xz[::-1, 1] - 1j * s_xz[::-1, 2]
S_reg.data[:mid, 2, 2] = s_xy[::-1, 1] - 1j * s_xy[::-1, 2]

S_dlr_iw = Gloc.copy()

for iwn in w_mesh:
    S_dlr_iw[iwn] = S_reg(iwn.value)


# oplot(S_reg[0,0])
# oplot(S_dlr[0,0])
# plt.show()

mpi.report('time for setup: {:.2f} s'.format(timer() - start_time))

trafo_start_time = timer()

k_array_slice = mpi.slice_array(k_array)
h_k_mat_slice = TB.fourier(k_array_slice)

mpi.barrier()
mpi.report('time for Fourier trafo to H(k): {:.2f} s'.format(timer() - trafo_start_time))

# find mu
time_start_mu = timer()
mu = brentq(calc_target_dens, search_win[0], search_win[1], (w_mat, h_k_mat_slice, S_dlr_iw, n_elect), xtol=1e-8)

mpi.barrier()
mpi.report('total time for calc_mu: {:.2f} s'.format(timer() - time_start_mu))

mpi.report(f'#####\nfinal μ={mu:.5f} for n={n_elect:.3f}\n#####\n')

Gloc = calc_Gloc(mu, w_mat, h_k_mat_slice, S_dlr_iw)

if mpi.is_master_node():
    tot_dens = Gloc.total_density().real
    dens_mat = Gloc.density().real
    print(f'total density: {tot_dens:.4f}')
    print('density matrix:')
    print(dens_mat)

    if store:
        with HDFArchive(store, 'a') as h5:
            h5['Gloc'] = Gloc

mpi.barrier()
mpi.report('overall time: {:.2f} s'.format(timer() - start_time))
