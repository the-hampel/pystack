from triqs.lattice.utils import TB_from_wannier90, k_space_path
import matplotlib.pyplot as plt
import numpy as np
from triqs.gf import MeshReFreq, Gf
from triqs.utility import mpi
from timeit import default_timer as timer
from h5 import HDFArchive

#------------------------------#
# parameters
k_dim = 41
omega_range = 3
n_omega = 1001
dft_fermi = 7.5689906322
mu = 0.0
n_orb = 3
eta = 0.05
w90_seed = 'wannier90'
w90_path = '/mnt/home/ahampel/work/GW+DMFT/GW-benchmark/SMO/triqs-cubic/scf/'
spin = None
store = 'dos.h5'
plot = w90_seed+'_dos.pdf'
# ------------------------------#

if spin:
    spin_factor = 1
    n_orb = 2*n_orb
else:
    spin_factor = 2

start_time = timer()

# load wannier90 file and create k_mesh
TB = TB_from_wannier90(seed=w90_seed, path=w90_path,
                       extend_to_spin=spin, add_local=np.diag([-dft_fermi] * n_orb))

d_start_time = timer()

k_spacing = np.linspace(0, 1, k_dim, endpoint=False)
k_array = np.array(np.meshgrid(k_spacing, k_spacing, k_spacing)).T.reshape(-1, 3)
n_k = k_dim**3

# # omega mesh, mu, and eta matrix vectors for each frequency
w_mesh = MeshReFreq(omega_min=-omega_range, omega_max=omega_range, n_max=n_omega)
w_mat = np.array([w.value * np.eye(n_orb) for w in w_mesh])
mu_mat = mu * np.eye(n_orb)
eta_mat = 1j * eta * np.eye(n_orb)
wk = 1/n_k

# local Gf for spectral function
Gloc = Gf(mesh=w_mesh, target_shape=[n_orb, n_orb])

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
    Gloc.data[:, :, :] += wk * np.linalg.inv(w_mat[:] + mu_mat - h_k_mat + eta_mat)

# gather results
Gloc << mpi.all_reduce(mpi.world, Gloc, lambda x, y: x+y)

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
        print('spectral function stored in: '+store)

mpi.barrier()
mpi.report('overall time: {:.2f} s'.format(timer() - start_time))
