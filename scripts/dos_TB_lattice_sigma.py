from triqs.lattice.utils import TB_from_wannier90, k_space_path
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from triqs.gf import MeshReFreq, Gf
from triqs.utility import mpi
from timeit import default_timer as timer
from h5 import HDFArchive
from triqs_dft_tools.sumk_dft import SumkDFT

#------------------------------#
# parameters
k_dim = 61
omega_range = 2.5
n_omega = 3001
dft_fermi = 10.6412
n_orb = 12
sites = 4
eta = 0.0
w90_seed = 'lvo'
w90_path = '/mnt/ceph/users/ahampel/LiV2O4/dft/wan_nscf_t2g/'
h5 = 'lvo_k41.h5'
spin = None
store = 'Aw_sigma.h5'
plot = None
# ------------------------------#

if spin:
    spin_factor = 1
    n_orb = 2*n_orb
else:
    spin_factor = 2

start_time = timer()

# load wannier90 file and create k_mesh
TB = TB_from_wannier90(seed=w90_seed, path=w90_path,
                       extend_to_spin=spin)

d_start_time = timer()

k_spacing = np.linspace(0, 1, k_dim, endpoint=False)
k_array = np.array(np.meshgrid(k_spacing, k_spacing, k_spacing)).T.reshape(-1, 3)
n_k = k_dim**3

# # omega mesh, mu, and eta matrix vectors for each frequency
w_mesh = MeshReFreq(omega_min=-omega_range, omega_max=omega_range, n_max=n_omega)
w_mesh_arr = np.linspace(-omega_range, omega_range, n_omega)
w_mat = np.array([w.value * np.eye(n_orb) for w in w_mesh])
eta_mat = 1j * eta * np.eye(n_orb)
wk = 1/n_k

# local Gf for spectral function
Gloc = Gf(mesh=w_mesh, target_shape=[n_orb, n_orb])

sigma_imp_list = []
dc_imp_list = []
with HDFArchive(h5, 'r') as ar:
    for icrsh in range(ar['dft_input']['n_inequiv_shells']):
        sigma_imp_list.append(ar['DMFT_results']['last_iter']['Sigma_maxent_0'])

    for ish in range(ar['dft_input']['n_corr_shells']):
        dc_imp_list.append(ar['DMFT_results']['last_iter']['DC_pot'][ish])

    mu_dmft = ar['DMFT_results']['last_iter']['chemical_potential_post']

    sum_k = SumkDFT(h5, mesh=sigma_imp_list[0].mesh)
    sum_k.block_structure = ar['DMFT_input/block_structure']
    sum_k.deg_shells = ar['DMFT_input/deg_shells']
    sum_k.set_mu = mu_dmft
    # set Sigma and DC into sum_k
    sum_k.dc_imp = dc_imp_list
    sum_k.put_Sigma(sigma_imp_list)

    # use add_dc function to rotate to sumk block structure and subtract the DC
    sigma_sumk = sum_k.add_dc()

    # now upfold with proj_mat to band basis, this only works for the
    # case where proj_mat is equal for all k points (wannier mode)
    sigma = Gf(mesh=sigma_imp_list[0].mesh, target_shape=[n_orb, n_orb])
    for ish in range(ar['dft_input']['n_corr_shells']):
         sigma += sum_k.upfold(ik=0, ish=ish, bname='up', gf_to_upfold=sigma_sumk[ish]['up'], gf_inp=sigma)

sigma_mat = np.zeros((n_omega, n_orb, n_orb), dtype=complex)
w_mesh_dmft = np.linspace(sigma.mesh.omega_min, sigma.mesh.omega_max, len(sigma.mesh))
mu_mat = mu_dmft * np.eye(n_orb)
# mu_mat = dft_fermi * np.eye(n_orb)

for orb1 in range(n_orb):
	for orb2 in range(n_orb):
		sigma_mat[:, orb1, orb2] = np.interp(w_mesh_arr, w_mesh_dmft, sigma.data[:, orb1, orb2])

mpi.report('time for setup: {:.2f} s'.format(timer() - start_time))

# Loop on k points via MPI
trafo_start_time = timer()

k_array_slice = mpi.slice_array(k_array)
h_k_mat_slice = TB.fourier(k_array_slice)

mpi.barrier()
mpi.report('time for Fourier trafo to H(k): {:.2f} s'.format(timer() - trafo_start_time))

k_start_time = timer()
# loop over slice on each rank and calc Gloc contribution
for h_k_mat in h_k_mat_slice:
	# Gloc.data[:, :, :] += wk * np.linalg.inv(w_mat + mu_mat[None, ...] - h_k_mat[None, ...] + eta_mat[None, ...])
	Gloc.data[:, :, :] += wk * np.linalg.inv(w_mat + mu_mat[None, ...] - h_k_mat[None, ...] + eta_mat[None, ...] - sigma_mat)

# gather results
Gloc << mpi.all_reduce(Gloc)

mpi.barrier()
mpi.report('time for k sum: {:.2f} s'.format(timer() - k_start_time))

# calc spectral function
Aw = -1.0 / np.pi * spin_factor * np.trace(Gloc.data, axis1=1, axis2=2).imag

if mpi.is_master_node():
    if plot:
        fig, ax = plt.subplots(1, dpi=150, figsize=(7, 3))
        ax.plot(w_mat[:, 0, 0], Aw)
        ax.set_ylabel(r'A($\omega$)')
        ax.set_xlabel(r'$\omega$')
        ax.set_xlim(-omega_range, omega_range)
        plt.savefig(plot, bbox_inches='tight', transparent=True, pad_inches=0.05)
        print('dos plot saved as: '+w90_seed+'_dos.pdf')

    if store:
        with HDFArchive(store, 'a') as h5:
            h5['A_w'] = Aw
            h5['Gloc'] = Gloc
            h5['mesh'] = np.linspace(w_mesh.omega_min, w_mesh.omega_max, len(w_mesh))
        print('spectral function stored in: '+store)

mpi.barrier()
mpi.report('overall time: {:.2f} s'.format(timer() - start_time))
