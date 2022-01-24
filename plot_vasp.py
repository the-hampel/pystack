

def smooth(y, box_pts):

    import numpy as np

    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def dos(vasprun_path = './vasprun.xml', t2g_list = [], eg_list = [], emin = -10, emax= 10, y_axis = [0,20], tdos = True, avg_points = 10, file_name=None):
    import matplotlib.pyplot as plt

    import numpy as np
    import collections
    import scipy as sp

    import pymatgen as mg
    import pymatgen.io.vasp.outputs as vio
    from pymatgen.electronic_structure.core import Spin, Orbital
    from pymatgen.electronic_structure import dos

    vasprun = vio.Vasprun(vasprun_path)

    dos_size = len(vasprun.pdos[0][Orbital.dz2][Spin.up][:])

    atom_list = vasprun.atomic_symbols

    atoms = collections.OrderedDict()

    for i in range(0,len(atom_list)):

        if not atom_list[i] in atoms:
            atoms[atom_list[i]] = [i]
        else:
            atoms[atom_list[i]].append(i)

    d_list = ['dxz','dyz','dxy','dx2','dz2']
    f_list = ['f_1','f_2','f_3','f0','f1','f2','f3']

    A_d = np.zeros(dos_size)
    A_f = np.zeros(dos_size)
    A_f_dos = False
    B_t2g = np.zeros(dos_size)
    B_eg = np.zeros(dos_size)
    O_p = np.zeros(dos_size)
    A_d_dn = np.zeros(dos_size)
    A_f_dn = np.zeros(dos_size)
    B_t2g_dn = np.zeros(dos_size)
    B_eg_dn = np.zeros(dos_size)
    O_p_dn = np.zeros(dos_size)

    # is magnetic?
    if Spin.down in vasprun.tdos.densities:
        mag = True
    else:
        mag = False


    for species, atoms in atoms.items():

        if species in ['Mo','V','Cu']:
            B_species = species
            for i in atoms:

                for eg in eg_list:
                    B_eg += np.array(vasprun.pdos[i][Orbital[eg]][Spin.up])

                    if mag:
                        B_eg_dn += np.array(vasprun.pdos[i][Orbital[eg]][Spin.down])

                for t2g in t2g_list:
                    B_t2g += np.array(vasprun.pdos[i][Orbital[t2g]][Spin.up])

                    if mag:
                        B_t2g_dn += np.array(vasprun.pdos[i][Orbital[t2g]][Spin.down])

        elif species == 'O':
            for i in atoms:
                O_p += np.array(vasprun.pdos[i][Orbital.px][Spin.up])
                O_p += np.array(vasprun.pdos[i][Orbital.py][Spin.up])
                O_p += np.array(vasprun.pdos[i][Orbital.pz][Spin.up])

                if mag:
                    O_p_dn += np.array(vasprun.pdos[i][Orbital.px][Spin.down])
                    O_p_dn += np.array(vasprun.pdos[i][Orbital.py][Spin.down])
                    O_p_dn += np.array(vasprun.pdos[i][Orbital.pz][Spin.down])

        elif not species == 'O' and not species in ['Mo','V', 'Cu']:
            A_species = species
            for i in atoms:
                for orb in d_list:
                    A_d += np.array(vasprun.pdos[i][Orbital[orb]][Spin.up])
                    if mag:
                        A_d_dn += np.array(vasprun.pdos[i][Orbital[orb]][Spin.down])

                if np.array(vasprun.pdos[i][Orbital.f_1]):
                    A_f_dos = True
                    for orb in f_list:
                        A_f += np.array(vasprun.pdos[i][Orbital[orb]][Spin.up])
                        if mag:
                            A_f += np.array(vasprun.pdos[i][Orbital[orb]][Spin.down])


    fig, (ax1) = plt.subplots(1,1,sharex="col",figsize=(12,6))

    ax1.fill_between(vasprun.tdos.energies - vasprun.efermi,
                     smooth(vasprun.tdos.densities[Spin.up], avg_points), 0,
                     color='gray', linewidth=0,alpha=0.3, label='tdos')

    ax1.plot(vasprun.tdos.energies - vasprun.efermi , smooth(B_t2g, avg_points),
         "-", label = str(B_species)+r" t$_{2g}$", color = 'C0', linewidth = 2)
    ax1.plot(vasprun.tdos.energies - vasprun.efermi, smooth(B_eg, avg_points),
             "-", label = str(B_species)+r" e$_g$", color = 'C1', linewidth = 2)
    ax1.plot(vasprun.tdos.energies - vasprun.efermi , smooth(O_p, avg_points),
             "-", label = r"$O_p$", color = 'C2', linewidth = 2)

    # A site
    ax1.plot(vasprun.tdos.energies - vasprun.efermi , smooth(A_d, avg_points),
     "-", label = str(A_species)+r"$_d$", color = 'C3', linewidth = 2)

    # A site f dos
    if A_f_dos:
        ax1.plot(vasprun.tdos.energies - vasprun.efermi , smooth(A_f, avg_points),
             "-", label = str(A_species)+r"$_f$", color = 'C4', linewidth = 2)

    if mag:
        ax1.fill_between(vasprun.tdos.energies - vasprun.efermi,
                         -1*smooth(vasprun.tdos.densities[Spin.down], avg_points), 0,
                         color='gray', linewidth=0,alpha=0.3)

        ax1.plot(vasprun.tdos.energies - vasprun.efermi , -1*smooth(B_t2g_dn, avg_points),
             "-", color = 'C0', linewidth = 2)
        ax1.plot(vasprun.tdos.energies - vasprun.efermi, -1*smooth(B_eg_dn, avg_points),
                 "-", color = 'C1', linewidth = 2)
        ax1.plot(vasprun.tdos.energies - vasprun.efermi , -1*smooth(O_p_dn, avg_points),
                 "-", color = 'C2', linewidth = 2)

        # A site
        ax1.plot(vasprun.tdos.energies - vasprun.efermi , -1*smooth(A_d_dn, avg_points),
         "-", color = 'C3' , linewidth = 2)

        # A site f dos
        if A_f_dos:
            ax1.plot(vasprun.tdos.energies - vasprun.efermi , -1*smooth(A_f_dn, avg_points),
                 "-", color = 'C4' , linewidth = 2)




    ax1.set_xlim(emin,emax)
    ax1.set_ylim(y_axis[0],y_axis[1])
    ax1.set_xlabel(r"$E - E_F$ (eV)")
    ax1.set_ylabel(r"$\rho \ \left( states/eV \right)$")
    ax1.vlines(x=0, ymin=-1000, ymax=1000, color="k", lw=2)
    ax1.hlines(y=0, xmin=-1000, xmax=1000, color="k", lw=1)

    ax1.xaxis.set_ticks_position('both')
    ax1.legend(loc='upper right', ncol=1,numpoints=1,handlelength=1,fancybox=True,
               labelspacing=0.2,borderaxespad=0.5,borderpad=0.35,handletextpad=0.4)
    ax1.tick_params(direction='in',pad=2)

    if file_name:
        plt.savefig(file_name,dpi=600, bbox_inches='tight',transparent=True, pad_inches=0.01)
    plt.show()

    return vasprun
