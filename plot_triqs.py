import os
import matplotlib.pyplot as plt

import numpy as np
import scipy as sp

from glob import glob
from os.path import basename

import warnings
warnings.filterwarnings("ignore") #ignore some matplotlib warnings

from h5 import HDFArchive
from triqs.gf import *

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def extract_obs(h5):
    with HDFArchive(h5,'r') as ar:
        obs = ar['DMFT_results']['observables']

    obs['n_imp'] = len(obs['orb_occ'][:])

    obs['orb_occ_sum'] = []
    obs['orb_gb2_sum'] = []
    obs['n_orb'] = []
    for imp in range(0,obs['n_imp']):
        obs['orb_occ_sum'].append(np.array(obs['orb_occ'][0]['up'])+np.array(obs['orb_occ'][0]['down']))
        obs['orb_gb2_sum'].append(np.array(obs['orb_gb2'][0]['up'])+np.array(obs['orb_gb2'][0]['down']))
        obs['n_orb'].append(obs['orb_occ_sum'][imp].shape[1])

    return obs

def fit_tail(S_iw, nmin, nmax, order = 4, known_moments= [], block = 'up_0', orb=0, xlim=(0,40), ylim=(-1.5,0.1)):
    
    beta =S_iw[block].mesh.beta
    
    mesh = np.array([w.imag for w in S_iw[block].mesh])

    S_iw_mfit = S_iw.copy()
    print(block)
    if not known_moments:
        shape = [0] + list(S_iw_mfit[block].target_shape)

        known_moments = np.zeros(shape, dtype=np.complex)

    o_min = (2*nmin+1)*np.pi/beta

    o_max = (2*nmax+1)*np.pi/beta
    for block, Gf_bl in S_iw_mfit:
        tail, err = S_iw_mfit[block].fit_hermitian_tail_on_window(n_min = nmin,
                                                      n_max = nmax ,
                                                      known_moments = known_moments,
                                                      n_tail_max = 2 * len(S_iw_mfit.mesh) ,
                                                      expansion_order = order)

        S_iw_mfit[block].replace_by_tail(tail,nmax)


        S_iw_mfit[block].replace_by_tail(tail,nmax)

    fig, (ax1) = plt.subplots(1,1,figsize=(16,10))

    ax1.axvline(x=o_min, color='k',label='window')
    ax1.axvline(x=o_max, color='k')

    ax1.plot(mesh,S_iw[block][orb,orb].data.imag,'o',lw=3,label='raw',markersize=4)
    ax1.plot(mesh,S_iw_mfit[block][orb,orb].data.imag,'-',lw=3,label='fit')

    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_ylabel(r"$Im \Sigma (i \omega)$")
    ax1.set_xlabel(r"$\omega$")

    ax1.legend(loc='lower right', ncol=1,numpoints=1,handlelength=1,fancybox=True,
               labelspacing=0.2,borderaxespad=0.5,borderpad=0.35,handletextpad=0.4)

    plt.show()

    return S_iw_mfit

def extract_Z_visual(h5, order=4, start=0, fitpoints=7, imp=0, plot=False, it='last_iter'):

    if plot:
        xp = np.linspace(-1, 5, 500)
        width = 2*1.07*3.41667

        fig, (ax1) = plt.subplots(1,1,figsize=(1.3*width,1.3*width))
        fig.subplots_adjust(wspace=0.3)
        ax1.set_xlim(0,2)
        ax1.set_ylim(-2.0,0.05)
        ax1.set_ylabel(r"$Im \Sigma (i \omega)$")

    Z_t2g = []
    Z_eg =[]
    scat_t2g = []
    scat_eg =[]
    
    if isinstance(h5,str):
        with HDFArchive(h5,'r') as h5:
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

            for orb in range(0,S_iw_avg.target_shape[0]):
                Im_S_iw = S_iw_avg[orb,orb].data.imag
                # simple extraction from S_iw_0
                Z_simple = 1/(1 - (Im_S_iw[n_iw0+start]/iw[n_iw0+start]) )

                p_fit = np.polyfit(iw[n_iw0+start:n_iw0+start+fitpoints],Im_S_iw[n_iw0+start:n_iw0+start+fitpoints],order)
                p_der = np.polyder(p_fit)
                Z_fit = 1.0/(1.0 - np.polyval(p_der,0.0))
                scat_fit = -1*np.polyval(p_fit,0.0)
                scat_fit_d = np.poly1d(p_fit)

                if Z_simple < 0.85:
                    Z_t2g.append(Z_fit)
                    scat_t2g.append(scat_fit)
                else:
                    Z_eg.append(Z_fit)
                    scat_eg.append(scat_fit)

                if plot:
                    # Sigma
                    ax1.plot(iw,Im_S_iw,'o',label=orb)
                    ax1.plot(xp,scat_fit_d(xp),
                         '-',lw='1.5')


    if plot:
        ax1.legend(loc='upper right', ncol=1,numpoints=1,handlelength=1,fancybox=True,
               labelspacing=0.2,borderaxespad=0.5,borderpad=0.35,handletextpad=0.4)
        plt.show()

    return Z_t2g, Z_eg, scat_t2g, scat_eg

def plot_conv(h5_files):
    if type(h5_files) != list:
        h5_files = [h5_files]

    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(22,14))
    fig.subplots_adjust(wspace=0.15,hspace=0.05)

    for i, h5 in enumerate(h5_files):
        with HDFArchive(h5,'r') as h5:
            conv_obs = h5['DMFT_results']['convergence_obs']

        n_imp = len(conv_obs['d_imp_occ'])


        for imp in range(n_imp):
            ax1.plot(conv_obs['d_imp_occ'][imp],'-o',label=str(i)+' d imp')

        ax1.plot(conv_obs['d_mu'],'-o',label=str(i)+' d mu')


        ax1.set_ylim(0.0,conv_obs['d_mu'][2])

        ax1.set_ylabel(r"delta")


        ax1.legend(loc='lower left', ncol=1,numpoints=1,handlelength=1,fancybox=True,
                   labelspacing=0.2,borderaxespad=0.5,borderpad=0.35,handletextpad=0.4)


        #############

        for imp in range(n_imp):
            ax2.plot(conv_obs['d_Gimp'][imp],'-o',label=str(i)+' dGimp')


        ax2.set_ylim(0.0,conv_obs['d_Gimp'][0][2])
        ax2.set_ylabel(r"delta Gimp")

        ax2.legend(loc='lower left', ncol=1,numpoints=1,handlelength=1,fancybox=True,
                   labelspacing=0.2,borderaxespad=0.5,borderpad=0.35,handletextpad=0.4)

        for imp in range(n_imp):
            ax3.plot(conv_obs['d_G0'][imp],'-o',label=str(i)+' d G0')

        ax3.set_xlabel(r"it")
        ax3.set_ylabel(r"delta G0")
        ax3.set_ylim(0.0,conv_obs['d_G0'][0][3])
        ax3.legend(loc='lower left', ncol=1,numpoints=1,handlelength=1,fancybox=True,
                   labelspacing=0.2,borderaxespad=0.5,borderpad=0.35,handletextpad=0.4)

        for imp in range(n_imp):
            ax4.plot(conv_obs['d_Sigma'][imp],'-o',label=str(i)+' d Sigma')

        ax4.set_xlabel(r"it")
        ax4.set_ylabel(r"delta Sigma")
        ax4.set_ylim(0.0,conv_obs['d_Sigma'][0][2])
        ax4.legend(loc='lower left', ncol=1,numpoints=1,handlelength=1,fancybox=True,
                   labelspacing=0.2,borderaxespad=0.5,borderpad=0.35,handletextpad=0.4)

    plt.show()
    return


def plot_Gl_coeff(h5,block,orb,imp=0,it='last_iter'):
    from triqs.plot.mpl_interface import plt,oplot
    with HDFArchive(h5,'r') as ar:
        Gl = ar['DMFT_results'][it]['Gimp_l_'+str(imp)]
        S_iw = ar['DMFT_results'][it]['Sigma_freq_'+str(imp)]

    # latex columnwidth is 246pt : 3.41667 inch and with tight padding 1.12 is the factor!
    width = 1.7*1.07*3.41667

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(2.6*width,width))

    nl = range(0,len(Gl[block][orb,orb].data[:].real),1)[0::2]
    ax1.semilogy(nl,(np.abs(Gl[block][orb,orb].data[0::2])),"o-", color='C0', label = "$G_l$  even", linewidth = 1.5)

    nl_odd = range(0,len(Gl[block][orb,orb].data[:].real),1)[1::2]
    ax1.semilogy(nl_odd,(np.abs(Gl[block][orb,orb].data[1::2])),"x-", color='C1' ,label = "$G_l$  odd", linewidth = 1.5)


    ax1.set_xlabel(r"$l$")
    ax1.set_ylabel(r"$|$G$_{l}|$")

    ax1.xaxis.set_ticks_position('both')
    ax1.legend(loc='upper right', ncol=1,numpoints=1,handlelength=1,fancybox=True,
               labelspacing=0.2,borderaxespad=0.5,borderpad=0.35,handletextpad=0.4)
    ax1.tick_params(direction='in',pad=2)

    # Sigma
    ax2.oplot(S_iw[block][orb,orb].imag,'-',color='C3',label='Im')

    ax3 = ax2.twinx()
    ax3.oplot(S_iw[block][orb,orb].real,'-',color='C2',label='Re')

    ax2.set_xlim(0,25)
    ax2.set_ylabel(r"$Re \Sigma (i \omega)$")
    ax3.set_ylabel(r"$Im \Sigma (i \omega)$")

    ax2.legend(loc='upper left', ncol=1,numpoints=1,handlelength=1,fancybox=True,
               labelspacing=0.2,borderaxespad=0.5,borderpad=0.35,handletextpad=0.4)
    plt.show()

    return

def plot_G_S(h5,block,orb,imp=0,it='last_iter', w_max=30):

    with HDFArchive(h5,'r') as ar:
        G_iw = ar['DMFT_results'][it]['Gimp_freq_'+str(imp)]
        S_iw = ar['DMFT_results'][it]['Sigma_freq_'+str(imp)]

    # latex columnwidth is 246pt : 3.41667 inch and with tight padding 1.12 is the factor!
    width = 1.7*1.07*3.41667

    fig, (ax1,ax3) = plt.subplots(1,2,figsize=(2.6*width,width))
    fig.subplots_adjust(wspace=0.3)

    ax1.oplot(G_iw[block][orb,orb].real,'-',color='C3',label='Re')

    ax2 = ax1.twinx()
    ax2.oplot(G_iw[block][orb,orb].imag,'-',color='C2',label='Im')


    ax1.set_xlim(0,w_max)
    ax1.set_ylabel(r"$Re G (i \omega)$")
    ax2.set_ylabel(r"$Im G (i \omega)$")

    ax1.xaxis.set_ticks_position('both')
    ax1.legend(loc='upper left', ncol=1,numpoints=1,handlelength=1,fancybox=True,
               labelspacing=0.2,borderaxespad=0.5,borderpad=0.35,handletextpad=0.4)
    ax2.legend(loc='upper right', ncol=1,numpoints=1,handlelength=1,fancybox=True,
               labelspacing=0.2,borderaxespad=0.5,borderpad=0.35,handletextpad=0.4)
    ax1.tick_params(direction='in',pad=2)

    # Sigma
    ax3.oplot(S_iw[block][orb,orb].real,'-',color='C3',label='Re')

    ax4 = ax3.twinx()
    ax4.oplot(S_iw[block][orb,orb].imag,'-',color='C2',label='Im')

    ax3.set_xlim(0,w_max)
    ax3.set_ylabel(r"$Re \Sigma (i \omega)$")
    ax4.set_ylabel(r"$Im \Sigma (i \omega)$")

    ax3.legend(loc='upper left', ncol=1,numpoints=1,handlelength=1,fancybox=True,
               labelspacing=0.2,borderaxespad=0.5,borderpad=0.35,handletextpad=0.4)
    ax4.legend(loc='upper right', ncol=1,numpoints=1,handlelength=1,fancybox=True,
               labelspacing=0.2,borderaxespad=0.5,borderpad=0.35,handletextpad=0.4)
    plt.show()

    return

def lorentzian( x, x0, a, gam ):
    return a * gam**2 / ( gam**2 + ( x - x0 )**2)

def smear_PES(x_array, y_array, e_f, eps):
    x_to_modify = np.where(x_array - e_f + eps > 0)[0]
    lor_max = y_array[x_to_modify[0]]
    y_array[x_to_modify] = lorentzian(x_array[x_to_modify] ,e_f - eps, lor_max, eps)

    return y_array
