import os
import matplotlib.pyplot as plt

import numpy as np
import scipy as sp
import scipy.integrate as integrate

def set_plot_params():
    import matplotlib as mpl
    import seaborn as sns
    import scienceplots
    np.set_printoptions(precision=6, suppress=True, linewidth=200)
    # set matplotlib parameters
    # notebook switches of latex rendering
    plt.style.use(['science','notebook'])
    sns.set_palette('muted')
    mpl.rcParams['figure.facecolor'] = (1,1,1,0)
    mpl.rcParams['figure.edgecolor'] = (1,1,1,0)
    mpl.rcParams['axes.facecolor'] = (1,1,1,0)

def rotation_matrix(axis, theta):
    import numpy as np
    import math
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta degree.
    """
    # convert degree to radians
    theta = theta * np.pi / 180.0
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def find_supercell(prim_cell, super_cell):

    '''
    Given two paths to a structure file, returns the transformation M from
    primitive to supercell and the corresponding reciprocal transformation
    matrix. For more info see:
    math.stackexchange.com/questions/1078157/components-of-vector-in-dual-basis-transform-covariantly

    and

    github.com/QijingZheng/VaspBandUnfolding/blob/master/unfold.py

    Let M the transformation matrix between supercell and primitive cell,
    and A the column wise supercell vectors and a the column wise primitive
    cell vectors. Then M is given as:
            M = np.dot(A, np.linalg.inv(a))

    In real space, the basis vectors of Supercell (A) and those of the
    primitive cell (a) satisfy:
            A = np.dot(M, a);      a = np.dot(np.linalg.inv(M), A)

    Whereas in reciprocal space
            b = np.dot(M.T, B);    B = np.dot(np.linalg.inv(M).T, b)

    uses pymatgen to load structure files and match the structures

    '''

    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure

    PC = Structure.from_file(prim_cell)
    SC = Structure.from_file(super_cell)

    sm = StructureMatcher(primitive_cell=False, attempt_supercell=True)

    trans, shift, match_atoms = sm.get_transformation(SC, PC)

    k_trans = np.linalg.inv(trans).T

    return trans, k_trans, shift

def ext_fermi_energy(path):
    "extracting the fermi energy"
    doscar = open(path+'/DOSCAR')
    for i in range(5):
         doscar.readline()
    energy = float(doscar.readline().split()[3])
    print('efermi='+'',energy)
    doscar.close()
    return energy;

def pmatrix(a):
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{pmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv +=  [r'\end{pmatrix}']
    return '\n'.join(rv)

def ext_rbasis(path):
    "Calculating the reciprocal basis"
    poscar = open(path+"/POSCAR")
    rl = poscar.readlines()
    a_1 = np.asarray(rl[2].split(),dtype=np.float32)
    a_2 = np.asarray(rl[3].split(),dtype=np.float32)
    a_3 = np.asarray(rl[4].split(),dtype=np.float32)
    poscar.close()
    vol = np.dot(a_1,(np.cross(a_2,a_3)))
    b_1 = np.cross(a_2,a_3) / vol
    b_2 = np.cross(a_3,a_1) / vol
    b_3 = np.cross(a_1,a_2) / vol
    basis = np.array([[b_1],[b_2],[b_3]])
    print('basis of reciprocal lattice:')
    print(basis)
    return basis;


def ext_bands(path, efermi= None, spin= False):
    "extracting the band form the eignval file from VASP"
    if efermi == None:
        efermi = ext_fermi_energy(path)

    basis = ext_rbasis(path)

    #reading the file
    myfile=open(path+'/EIGENVAL')
    data=myfile.readlines()
    myfile.close()

    #extraction of bands
    zeile=data[5].split()
    nkpoints=int(zeile[1])
    nbands=int(zeile[2])

    print ('number of kpoints:',nkpoints)
    print ('number of bands:',nbands)

    nmin=1
    nmax = nbands

    if spin:
        bands_up=open(path+'/bands_up.dat','w')
        bands_dn=open(path+'/bands_dn.dat','w')
    else:
        bands_up=open(path+'/bands.dat','w')

    line = 0
    klen_old = 1000
    kvec_old=np.array([0.0,0.0,0.0])

    for x in range(nkpoints):
        zeile=data[7+x*(nbands+2)].split()
        kvec=np.array([float(zeile[0]),float(zeile[1]),float(zeile[2])])
        kvec = (basis.dot(kvec)).transpose()
        if x == 0:
            klen = 0.0
        else:
            kvec_diff = abs(kvec-kvec_old)
            klen= klen + np.sqrt(kvec_diff.dot(kvec_diff.transpose()))
        if klen - klen_old < 0.00001 and x > 1:
            line = line + 1
        klen_old = klen
        kvec_old = kvec
        value=''
        for y in range(nmin,nmax):
            zeile=data[7+x*(nbands+2)+1+y].split()
            ev = str(float(zeile[1])-efermi)
            value=value+'  '+ev
        bands_up.write(str(float(klen))+'  '+str(line)+'  '+value+'\n')

        if spin:
            value=''
            for y in range(nmin,nmax):
                zeile=data[7+x*(nbands+2)+1+y].split()
                ev = str(float(zeile[2])-efermi)
                value=value+'  '+ev
            bands_dn.write(str(float(klen))+'  '+str(line)+'  '+value+'\n')

    bands_up.close()
    if spin: bands_dn.close()

    return

def legendre_filter(G_tau, order=100, G_l_cut=1e-19):
    """ Filter binned imaginary time Green's function
    using a Legendre filter of given order and coefficient threshold.

    Parameters
    ----------
    G_tau : TRIQS imaginary time Block Green's function
    auto : determines automatically the cut-off nl
    order : int
        Legendre expansion order in the filter
    G_l_cut : float
        Legendre coefficient cut-off
    Returns
    -------
    G_l : TRIQS Legendre Block Green's function
        Fitted Green's function on a Legendre mesh
    """
    from triqs.gf.tools import fit_legendre
    from triqs.gf import BlockGf

    l_g_l = []

    for _, g in G_tau:

        g_l = fit_legendre(g, order=order)
        g_l.data[:] *= (np.abs(g_l.data) > G_l_cut)
        g_l.enforce_discontinuity(np.identity(g.target_shape[0]))

        l_g_l.append(g_l)

    G_l = BlockGf(name_list=list(G_tau.indices), block_list=l_g_l)

    return G_l

def linefit_real_freq(x,y,interval,spacing=50,addspace=0.0):
    def calc_Z(slope):
        Z = 1/(1-slope)
        return Z
    lim_l, lim_r = interval
    indices = np.where(np.logical_and(x>=lim_l, x<=lim_r))
    fit = np.polyfit(x[indices],y[indices],1)
#     print(fit)
    slope = fit[0]
    Z = calc_Z(slope)
    f_x = np.poly1d(fit)
    x_cont = np.linspace(x[indices][0]-addspace,x[indices][-1]+addspace, spacing)
    return x_cont, f_x(x_cont), Z, slope


def extract_bandwidth_h5(h5):

    with HDFArchive(h5,'r') as h5:
        ek = h5['dft_input']['hopping']

    bandwidth = 0.0
    ev_min = 2000
    ev_max = -2000
    for h_k in ek[:,0]:
        ev, _ = np.linalg.eigh(h_k)
        smallest_ev = ev.min()
        largest_ev = ev.max()
        if smallest_ev < ev_min:
            ev_min = ev.min()
        if largest_ev > ev_max:
            ev_max = largest_ev


    return abs(ev_max-ev_min)

def double_counting_ani(U, J, n_orb, occ):

    '''
    small function to calculate double counting via Held / Anisimov formula for Kanamori Hamiltonians

    parameters
    ----------
    U : float
        U value
    J : float
        J value
    n_orb : int
        number of orbitals
    occ : float
        nomber of electrons

    returns
    -------
    dc_pot : float
        double counting potential
    dc_en : float
        double counting energy
    '''

    U_bar = (U + (n_orb-1)*(U-2*J) + (n_orb-1)*(U-3*J))/(2*n_orb -1)

    dc_en = 0.5*U_bar*occ*(occ-1)
    dc_pot = U_bar*(occ-0.5)

    return dc_pot, dc_en

def mesh_to_np_arr(mesh):
    from triqs.gf import MeshImTime, MeshReFreq, MeshImFreq

    if isinstance(mesh, MeshReFreq):
        mesh_arr = np.linspace(mesh.w_min, mesh.w_max, len(mesh))
    elif isinstance(mesh, MeshImFreq):
        mesh_arr = np.linspace(mesh(mesh.first_index()).imag, mesh(mesh.last_index()).imag, len(mesh))
    elif isinstance(mesh, MeshImTime):
        mesh_arr = np.linspace(0, mesh.beta, len(mesh))
    else:
        raise AttributeError('input mesh must be either MeshReFreq, MeshImFreq, or MeshImTime')

    return mesh_arr
