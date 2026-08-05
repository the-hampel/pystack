import os
import sys
import math
import numpy as np
np.set_printoptions(precision=4,suppress=True, linewidth=140)
import matplotlib.pyplot as plt
from triqs.gf import *
# import custom solid_dmft
sys.path.insert(0, "/Users/ahampel/git/triqs/solid_dmft/python")

from solid_dmft.postprocessing import plot_correlated_bands as pcb
print(pcb.__file__)


# unfolding K points represented in ortho rel coords
G = [0.0, 0.0, 0.0]
X = [0.5, 0.0, 0.0]
Xbar = [0.0, 0.5, 0.0]
Z = [0.0, 0.0, 0.5]

dft_fermi = 13.2581623
w90_dict = {'w90_path': '/Users/ahampel/HiDrive/users/ahampel/flatiron/LaNiO3_arpes/w90/Pm-3m/f/',
            'w90_seed': 'lano',
            'add_spin': False, 'add_lambda': None,
            'n_orb': 2,
            'mu_tb' : dft_fermi}

# plotting options
plot_dict = {'colorscheme_bands': 'Greys',
             'colorscheme_alatt': 'Spectral_r',
             'colorscheme_kslice': 'Spectral_r',
             'colorbar': True,
             'vmin': 0.0, 'vmax' : 1.0}


sigma_dict = {'spin': 'up',
              'dc' : [{'up' : np.zeros((2,2))}],
              'mu_dmft' : dft_fermi,
              'w_mesh': {'window': [-1.0, 1.0], 'n_w': int(3001)},
              'linearize': False}

mesh = MeshReFreq(n_w=3001, window=(-1.0, 1.0))
sw = Gf(mesh=mesh, target_shape=[2, 2])
Sw_bgf = BlockGf(name_list=['up'], block_list=[sw])

tb_slice = {'bands_path': [('Xbar', 'G'),('G', 'X')], 'n_k': 100,
            'kz': 0.0, 'G': G, 'X': X, 'Xbar': Xbar, 'Z': Z}

# proj = [0]
proj = [1]
tb_slice_pcb, alatt, freq_dict_slice = pcb.get_dmft_bands(with_sigma=Sw_bgf,
                                                             fermi_slice=True,
                                                             add_mu_tb=False,
                                                             proj_on_orb=proj,
                                                             eta=0.01,
                                                             **w90_dict, **tb_slice, **sigma_dict)


fig, ax = plt.subplots(1,2,dpi=100,figsize=(14,5))

pcb.plot_kslice(fig, ax[0], alatt, tb_slice_pcb, freq_dict_slice, n_orb=w90_dict['n_orb'],
                tb_dict=tb_slice, tb=True, alatt=False, quarter=[0,1,2,3], **plot_dict)

pcb.plot_kslice(fig, ax[1], alatt, tb_slice_pcb, freq_dict_slice, n_orb=w90_dict['n_orb'],
                tb_dict=tb_slice, tb=False, alatt=True, quarter=[0,1,2,3], **plot_dict)

fig.suptitle(f'proj on orb {proj[0]}')
plt.show()
