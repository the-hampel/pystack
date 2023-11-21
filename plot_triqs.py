from triqs.gf import *
from h5 import HDFArchive
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import numpy as np
import scipy as sp

from glob import glob
from os.path import basename

import warnings
warnings.filterwarnings("ignore")  # ignore some matplotlib warnings


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def extract_obs(h5):
    with HDFArchive(h5, 'r') as ar:
        obs = ar['DMFT_results']['observables']
        conv_obs = ar['DMFT_results/convergence_obs']

    obs['n_imp'] = len(obs['orb_occ'][:])

    obs['orb_occ_sum'] = []
    obs['orb_gb2_sum'] = []
    obs['orb_mag_mom'] = []
    obs['imp_mag_mom'] = []
    obs['n_orb'] = []
    for imp in range(0, obs['n_imp']):
        obs['orb_occ_sum'].append(np.array(obs['orb_occ'][imp]['up'])
                                  + np.array(obs['orb_occ'][imp]['down']))
        obs['orb_gb2_sum'].append(np.array(obs['orb_gb2'][imp]['up'])
                                  + np.array(obs['orb_gb2'][imp]['down']))
        obs['orb_mag_mom'].append(np.array(obs['orb_occ'][imp]['up'])
                                  - np.array(obs['orb_occ'][imp]['down']))
        obs['imp_mag_mom'].append(np.array(obs['imp_occ'][imp]['up'])
                                  - np.array(obs['imp_occ'][imp]['down']))
        obs['n_orb'].append(obs['orb_occ_sum'][imp].shape[1])

    return obs, conv_obs


def fit_tail(G_inp, w_min, w_max, order=4, known_moments=[], fit_sigma=False):


    if isinstance(G_inp, BlockGf):
        res_list = []
        for block, gf in G_inp:
            res_list.append(fit_tail(gf, w_min, w_max, order, known_moments,fit_sigma))

        return BlockGf(name_list=list(G_inp.indices),block_list=res_list)

    G_iw = G_inp.copy()
    if known_moments==[]:
        # if fitting a self-energy we do not have any prior knowledge on tail
        if fit_sigma:
            shape = [0] + list(G_iw.target_shape)
            known_moments = np.zeros(shape, dtype=complex)
        else:
            known_moments = make_zero_tail(G_iw, 2)

    n_min = int(0.5*(w_min*G_iw.mesh.beta/np.pi - 1.0))
    n_max = int(0.5*(w_max*G_iw.mesh.beta/np.pi - 1.0))
    tail, err = G_iw.fit_hermitian_tail_on_window(n_min=n_min,
                                                  n_max=n_max,
                                                  known_moments=known_moments,
                                                  n_tail_max=4*len(G_iw.mesh),
                                                  expansion_order=order)
    G_iw.replace_by_tail(tail, n_max)

    return G_iw


def extract_Z_visual(h5, order=4, start=0, fitpoints=7, imp=0, plot=False, it='last_iter', xlim=[0, 2], ylim=[-2.0, 0.05]):

    if plot:
        xp = np.linspace(-1, 5, 500)

        fig, (ax1) = plt.subplots(1, 1, figsize=(8, 5), dpi=150)
        fig.subplots_adjust(wspace=0.3)
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)
        ax1.set_ylabel(r"$Im \Sigma (i \omega)$")

    Z_t2g = []
    Z_eg = []
    scat_t2g = []
    scat_eg = []

    if isinstance(h5, str):
        with HDFArchive(h5, 'r') as h5:
            try:
                Sigma_iw = h5['DMFT_results'][it]['Sigma_iw_'+str(imp)]
            except:
                Sigma_iw = h5['DMFT_results'][it]['Sigma_freq_'+str(imp)]
    else:
        Sigma_iw = h5

    # average of up / down
    for blck, S_blck in Sigma_iw:
        if 'up' in blck:
            nblck_no = blck.split('_')[-1]
            S_iw_avg = 0.5*(Sigma_iw[blck] + Sigma_iw['down_'+nblck_no])

            iw = [np.imag(n) for n in S_blck.mesh]
            n_iw0 = int(0.5*len(iw))

            for orb in range(0, S_iw_avg.target_shape[0]):
                Im_S_iw = S_iw_avg[orb, orb].data.imag
                # simple extraction from S_iw_0
                Z_simple = 1/(1 - (Im_S_iw[n_iw0+start]/iw[n_iw0+start]))

                p_fit = np.polyfit(iw[n_iw0+start:n_iw0+start+fitpoints],
                                   Im_S_iw[n_iw0+start:n_iw0+start+fitpoints], order)
                p_der = np.polyder(p_fit)
                Z_fit = 1.0/(1.0 - np.polyval(p_der, 0.0))
                scat_fit = -1*np.polyval(p_fit, 0.0)
                scat_fit_d = np.poly1d(p_fit)

                if Z_simple < 0.85:
                    Z_t2g.append(Z_fit)
                    scat_t2g.append(scat_fit)
                else:
                    Z_eg.append(Z_fit)
                    scat_eg.append(scat_fit)

                if plot:
                    # Sigma
                    ax1.plot(iw, Im_S_iw, 'o', label=orb)
                    ax1.plot(xp, scat_fit_d(xp),
                             '-', lw='1.5')

    if plot:
        ax1.legend(loc='upper right', ncol=1, numpoints=1, handlelength=1, fancybox=True,
                   labelspacing=0.2, borderaxespad=0.5, borderpad=0.35, handletextpad=0.4)
        plt.show()

    return Z_t2g, Z_eg, scat_t2g, scat_eg


def plot_conv_obs(h5, site=0, dpi=120):
    obs, conv_obs = extract_obs(h5)

    markers = ['o', 's', 'x', 'v', '^', '1', '2', '3', '4', '5']
    n_orb = obs['orb_occ'][site]['up'][0].shape[0]

    fig, ax = plt.subplots(nrows=7, dpi=dpi, figsize=(10, 14), sharex=True)
    fig.subplots_adjust(wspace=0.04, hspace=0.05)

    # chemical potential
    ax[0].plot(obs['iteration'], obs['mu'], '-o', color='C3')
    ax[0].set_ylabel(r'$\mu$ (eV)')

    # orb occupation
    for i_orb in range(n_orb):
        ax[1].plot(obs['iteration'], obs['orb_occ_sum'][site][:, i_orb],
                   marker=markers[i_orb], label=f'orb {i_orb}')
    ax[1].set_ylabel('orb occ')
    ax[1].legend()

    # A(w=0)
    for i_orb in range(n_orb):
        ax[2].plot(obs['iteration'], -1*obs['orb_gb2_sum'][site][:, i_orb], marker=markers[i_orb])
    ax[2].set_ylim(0,)
    ax[2].set_ylabel(r'$\bar{A}(\omega=0$)')

    # Z
    Z = 0.5*(np.array(obs['orb_Z'][site]['up'])+np.array(obs['orb_Z'][site]['down']))
    for i_orb in range(n_orb):
        ax[3].plot(obs['iteration'], Z[:, i_orb], marker=markers[i_orb])
    ax[3].set_ylabel(r'QP weight Z')

    # convergence of Weiss field
    ax[4].semilogy(obs['iteration'][1:], conv_obs['d_G0'][site], '-o', color='C4')
    ax[4].set_ylabel(r'$\Delta$ G$_0$')

    # convergence of DMFT self-consistency condition Gimp-Gloc
    ax[5].semilogy(obs['iteration'][1:], conv_obs['d_Gimp'][site], '-o', color='C5')
    ax[5].set_ylabel(r'|G$_{imp}$-G$_{loc}$|')

    # chemical potential diff
    ax[6].semilogy(obs['iteration'][2:], np.abs(
        np.array(obs['mu'][2:])-np.array(obs['mu'][0:-2])), '-o', color='C6')
    ax[6].set_ylabel(r'$\Delta \ \mu$ (eV)')

    ax[-1].set_xticks(range(0, len(obs['iteration'])))
    ax[-1].set_xlabel('Iterations')
    ax[-1].set_xlim(0,)
    ax[-1].xaxis.set_minor_locator(MultipleLocator(1))
    plt.show()

    return obs, conv_obs


def plot_Gl_coeff(h5, block, orb, ax, imp=0, it='last_iter'):
    from triqs.plot.mpl_interface import plt, oplot
    with HDFArchive(h5, 'r') as ar:
        Gl = ar['DMFT_results'][it]['Gimp_l_'+str(imp)]
        S_iw = ar['DMFT_results'][it]['Sigma_freq_'+str(imp)]

    nl = range(0, len(Gl[block][orb, orb].data[:].real), 1)[0::2]
    ax[0].semilogy(nl, (np.abs(Gl[block][orb, orb].data[0::2])), "o-",
                   color='C0', label="$G_l$  even", linewidth=1.5)

    nl_odd = range(0, len(Gl[block][orb, orb].data[:].real), 1)[1::2]
    ax[0].semilogy(nl_odd, (np.abs(Gl[block][orb, orb].data[1::2])),
                   "x-", color='C1', label="$G_l$  odd", linewidth=1.5)

    ax[0].set_xlabel(r"$l$")
    ax[0].set_ylabel(r"$|$G$_{l}|$")

    ax[0].xaxis.set_ticks_position('both')
    ax[0].legend(loc='upper right', ncol=1, numpoints=1, handlelength=1, fancybox=True,
                 labelspacing=0.2, borderaxespad=0.5, borderpad=0.35, handletextpad=0.4)
    ax[0].tick_params(direction='in', pad=2)

    # Sigma
    ax[1].oplot(S_iw[block][orb, orb].imag, '-', color='C3', label='Im')

    ax_twin = ax[1].twinx()
    ax_twin.oplot(S_iw[block][orb, orb].real, '-', color='C2', label='Re')

    ax[1].set_xlim(0, 25)
    ax[1].set_ylabel(r"$Re \Sigma (i \omega)$")
    ax_twin.set_ylabel(r"$Im \Sigma (i \omega)$")

    ax[1].legend(loc='upper left', ncol=1, numpoints=1, handlelength=1, fancybox=True,
                 labelspacing=0.2, borderaxespad=0.5, borderpad=0.35, handletextpad=0.4)
    plt.show()

    return


def plot_G_S(h5, block, orb, ax, imp=0, it='last_iter', w_max=30):

    with HDFArchive(h5, 'r') as ar:
        G_iw = ar['DMFT_results'][it]['Gimp_freq_'+str(imp)]
        S_iw = ar['DMFT_results'][it]['Sigma_freq_'+str(imp)]

    ax[0].oplot(G_iw[block][orb, orb].real, '-', color='C2', label='Re')

    ax1_twin = ax[0].twinx()
    ax1_twin.oplot(G_iw[block][orb, orb].imag, '-', color='C3', label='Im')

    ax[0].set_xlim(0, w_max)
    ax[0].set_ylabel(r"$Re G (i \omega)$")
    ax1_twin.set_ylabel(r"$Im G (i \omega)$")

    ax[0].xaxis.set_ticks_position('both')
    ax[0].legend(loc='upper left', ncol=1, numpoints=1, handlelength=1, fancybox=True,
                 labelspacing=0.2, borderaxespad=0.5, borderpad=0.35, handletextpad=0.4)
    ax1_twin.legend(loc='upper right', ncol=1, numpoints=1, handlelength=1, fancybox=True,
                    labelspacing=0.2, borderaxespad=0.5, borderpad=0.35, handletextpad=0.4)
    ax[0].tick_params(direction='in', pad=2)

    # Sigma
    ax[1].oplot(S_iw[block][orb, orb].real, '-', color='C2', label='Re')

    ax_twin2 = ax[1].twinx()
    ax_twin2.oplot(S_iw[block][orb, orb].imag, '-', color='C3', label='Im')

    ax[1].set_xlim(0, w_max)
    ax[1].set_ylabel(r"$Re \Sigma (i \omega)$")
    ax_twin2.set_ylabel(r"$Im \Sigma (i \omega)$")

    ax[1].legend(loc='upper left', ncol=1, numpoints=1, handlelength=1, fancybox=True,
                 labelspacing=0.2, borderaxespad=0.5, borderpad=0.35, handletextpad=0.4)
    ax_twin2.legend(loc='upper right', ncol=1, numpoints=1, handlelength=1, fancybox=True,
                    labelspacing=0.2, borderaxespad=0.5, borderpad=0.35, handletextpad=0.4)

    return


def plot_pert_order(h5, it='last_iter', o_max=None, dpi=120):
    from triqs_cthyb import Solver

    with HDFArchive(h5, 'r') as ar:
        pert_ord = ar['DMFT_results'][it]['pert_order_imp_0']

    fig, ax = plt.subplots(1, 1, figsize=(9, 4), dpi=dpi, squeeze=False, sharex=True)
    ax = ax.reshape(-1)

    for b in pert_ord:
        if 'down' in b:
            continue
        ax[0].oplot(pert_ord[b], label='block {:s}'.format(b))

    if o_max:
        ax[0].set_xlim(0, o_max)

    return


def lorentzian(x, x0, a, gam):
    return a * gam**2 / (gam**2 + (x - x0)**2)


def smear_PES(x_array, y_array, e_f, eps):
    x_to_modify = np.where(x_array - e_f + eps > 0)[0]
    lor_max = y_array[x_to_modify[0]]
    y_array[x_to_modify] = lorentzian(x_array[x_to_modify], e_f - eps, lor_max, eps)

    return y_array
