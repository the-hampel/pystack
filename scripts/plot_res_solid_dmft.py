import numpy as np
import sys
import glob

import matplotlib
# matplotlib.use('QtAgg')
import matplotlib.pyplot as plt

from h5 import HDFArchive
from triqs.plot.mpl_interface import plt, oplot
from triqs.gf.descriptors import Fourier
from triqs.gf import *

np.set_printoptions(precision=4, suppress=True, linewidth=1040)

print('plot solid_dmft results. Pass as argument the name of h5 file and quantitiy to plot. Options are:\nSigma_freq, Delta_time, G0_freq, Gimp_freq, Gimp_l, Gimp_time')

def _mesh_to_np_arr(mesh):
    from triqs.gf import MeshImTime, MeshReFreq, MeshImFreq

    if isinstance(mesh, MeshReFreq):
        mesh_arr = np.linspace(mesh.omega_min, mesh.omega_max, len(mesh))
    elif isinstance(mesh, MeshImFreq):
        mesh_arr = np.linspace(mesh(mesh.first_index()).imag, mesh(mesh.last_index()).imag, len(mesh))
    elif isinstance(mesh, MeshImTime):
        mesh_arr = np.linspace(0, mesh.beta, len(mesh))
    else:
        raise AttributeError('input mesh must be either MeshReFreq, MeshImFreq, or MeshImTime')

    return mesh_arr

h5_file = glob.glob(sys.argv[1])[0]

try:
    obj = sys.argv[2]
except:
    obj = 'Sigma_freq'

try:
    it = sys.argv[3]
    it = 'it_'+str(it)
except:
    it = 'last_iter'

print(f'plotting {obj} at iteration {it}')

with HDFArchive(h5_file, 'r') as ar:
    n_shells = ar['dft_input/n_inequiv_shells']
    deg_shell_list = ar['DMFT_input/deg_shells']
    print(deg_shell_list)
    obj_list = []
    for i_shell in range(n_shells):
        obj_list.append(ar[f'DMFT_results/{it}'][f'{obj}_{i_shell}'])


for i_imp, gf_obj in enumerate(obj_list):
    print(rf'Impurity {i_imp}')
    mesh = _mesh_to_np_arr(gf_obj.mesh)
    n_blocks = len(deg_shell_list[i_imp])

    for i_block, blocks in enumerate(deg_shell_list[i_imp]):
        # only plot the first of the deg blocks
        block = blocks[0]
        print(rf'Block {block}')
        n_orb = gf_obj[block].target_shape[0]
        figsize = (n_orb*4, n_orb*3)
        for part in ['Re', 'Im']:
            fig, ax = plt.subplots(n_orb, n_orb, figsize=figsize, dpi=150, squeeze=False, sharex=True)
            fig.subplots_adjust(wspace=0.3,hspace=0.1)
            plt.suptitle(rf'{part} {obj} | imp {i_imp} | block {block}')

            for i_orb in range(n_orb):
                for j_orb in range(n_orb):
                    if part == 'Im':
                        ax[i_orb, j_orb].plot(mesh, gf_obj[block].data[:, i_orb, j_orb].imag, '-', ms=1)
                    else:
                        ax[i_orb, j_orb].plot(mesh, gf_obj[block].data[:, i_orb, j_orb].real, '-', ms=1)
                    if isinstance(gf_obj.mesh, MeshImFreq) and i_orb == n_orb-1:
                        ax[i_orb, j_orb].set_xlim(0, mesh[-1])
                        ax[i_orb, j_orb].set_xlabel(r'$\nu_n$ (eV)')
                    if isinstance(gf_obj.mesh, MeshImTime) and i_orb == n_orb-1:
                        ax[i_orb, j_orb].set_xlim(mesh[0], mesh[-1])
                        ax[i_orb, j_orb].set_xlabel(r'$\tau$')


plt.show()
