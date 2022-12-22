import numpy as np
import sys
import glob


from h5 import HDFArchive

import triqs.utility.mpi as mpi
from triqs.gf import *

from triqs_cthyb.solver import Solver

np.set_printoptions(precision=4, suppress=True, linewidth=1040)


################
### params #####
rank = None   # specify rank to run if wanted, otherwise set to None
shells = None # specify shells to run cthyb, set to None to run all
h5_file = None
################

if not h5_file:
    try:
        h5_file = glob.glob(sys.argv[1])[0]
    except:
        raise ValueError('you have to specify the h5 file to start the debug run')

solver_list = []
param_list = []
with HDFArchive(h5_file, 'r') as ar:
    n_shells = ar['dft_input/n_inequiv_shells']
    for i_shell in range(n_shells):
        solver_list.append(ar['DMFT_input/solver/it_-1'][f'S_{i_shell}'])
        param_list.append(ar['DMFT_input/solver/it_-1'][f'solve_params_{i_shell}'])
    h_int = ar['DMFT_input']['h_int']
    mpi_size = ar['DMFT_input/solver/it_-1/mpi_size']

if not mpi_size == mpi.size:
    mpi.report(f'\n!!!! WARNING: number of mpi ranks differs. Current {mpi.size}, run on h5 had {mpi_size}.\n Results will differ.')


# for i_shell in range(n_shells):
for i_shell in shells:
    mpi.report(f'\n-----\n impurity {i_shell}/{n_shells}\n parameter list:')
    mpi.report(param_list[i_shell])
    param_list[i_shell].pop('perform_tail_fit', False)
    if rank:
        mpi.report(f'running with random seed init for rank {rank}')
        assert mpi.size == 1, 'rank random seed init only works for serial run'
        param_list[i_shell]['random_seed'] = int(34788 + 928374 * rank)
    solver_list[i_shell].solve(h_int=h_int[i_shell], **param_list[i_shell])
