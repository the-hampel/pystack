import os
import matplotlib.pyplot as plt

import numpy as np
import scipy as sp
import scipy.integrate as integrate

def calc_Z_B(U_w):
    '''
    Z_B = exp \left[ \frac{1}{\pi} \int_0^\infty d\omega \frac{\text{Im} U(\omega)}{\omega^2} \right]
    '''
    integrant = U_w[:,1].imag/(U_w[:,0]*U_w[:,0])
    integrant[0] = 0.0
    
    Z_B = np.exp(integrate.trapezoid(integrant,U_w[:,0])/np.pi)
    
    return Z_B.real

def U_scr_check(U_w, V):
    integrant = U_w[:,1].imag/U_w[:,0]
    integrant[0] = 0.0 

    U_scr = integrate.trapezoid(integrant,U_w[:,0])*2/np.pi
    
    print('should be:',U_w[0,1].real)
    print('integrated:',(V+U_scr).real)

    return (V+U_scr).real


def calc_U_iw(U_w, beta=40, n_iw = 10001, V=None, plot=True, fit_w0=False, fitpoints=5, shift=None):
    from triqs.gf import MeshImFreq, GfImFreq
    
    iw_mesh = MeshImFreq(beta=beta, S="Boson", n_max=n_iw)
    iw_mesh_np = np.array([w.value.imag for w in iw_mesh])
    U_iw = GfImFreq(mesh =iw_mesh,target_shape=[1,1])
    
    if V:
        U_iw << V


    w_mesh = U_w[:,0]
    
    for i_w in iw_mesh:
        integrant = U_w[:,1].imag*w_mesh/(i_w.value**2-w_mesh**2)
        U_iw[i_w] = U_iw[i_w] - (2/np.pi) * integrate.trapezoid(integrant,w_mesh)
    
    # if no bare V is given extrapolate
    if not V:
        # the smallest Matsubara freq should be roughly linear ibn iwn
        n_iw0 = int(0.5*len(iw_mesh_np))
        start = 1
        p_fit = np.polyfit(iw_mesh_np[n_iw0+start:n_iw0+start+fitpoints],U_iw.data[:,0,0].real[n_iw0+start:n_iw0+start+fitpoints],1)
        fit_line = np.poly1d(p_fit)
        U_iw0_fit = np.polyval(p_fit,0.0)
        V = -1*U_iw0_fit+U_w[0,1].real
        print('extrapolated bare V:',V)
        
        U_iw << U_iw + V
    
    # if fit_w0 is True fit smallest Mat U_iw data linearly instead of using real Uiw data
    if fit_w0:
        # the smallest Matsubara freq should be roughly linear ibn iwn
        n_iw0 = int(0.5*len(iw_mesh_np))
        start = 1
        p_fit = np.polyfit(iw_mesh_np[n_iw0+start:n_iw0+start+fitpoints],U_iw.data[:,0,0].real[n_iw0+start:n_iw0+start+fitpoints],1)
        fit_line = np.poly1d(p_fit)
        U_iw0 = np.polyval(p_fit,0.0)

        print('fitted screened U:',U_iw0)
        U_iw.data[int(len(iw_mesh_np)/2)] = U_iw0
    else:    
        U_iw0 = U_w[0,1].real
        U_iw.data[int(len(iw_mesh_np)/2)] = U_iw0
    
    
    if shift=='V':
        U_iw << U_iw - V
    
    if shift=='U':
        U_iw << U_iw - U_iw0
        
    print('U_iwn0:',U_iw.data[int(len(iw_mesh_np)/2)][0,0].real)
    print('U_iw0+1:',U_iw.data[int(len(iw_mesh_np)/2)+1][0,0].real)
    print('U_iwn:',U_iw.data[-1,0,0].real)

    if plot:
        fig, ax1 = plt.subplots(1,1,sharex="col",figsize=(12,8))

        ax1.plot(iw_mesh_np,U_iw.data[:,0,0].real,'o')

        plt.xlim(0,5)
        plt.ylim(U_iw.data[int(len(iw_mesh_np)/2)].real,U_iw.data[int(len(iw_mesh_np)/2)].real+3)
        plt.ylabel(r'$U_{scr}(i \nu_n)$')
        plt.xlabel(r'$i \nu_n$')


        fig, ax1 = plt.subplots(1,1,sharex="col",figsize=(12,8))

        ax1.plot(iw_mesh_np,U_iw.data[:,0,0].real,'-')

        plt.xlim(0,200)
        plt.ylim(U_iw.data[int(len(iw_mesh_np)/2)].real,U_iw.data[-1].real)
        plt.ylabel(r'$U_{scr}(i \nu_n)$')
        plt.xlabel(r'$i \nu_n$')

    return U_iw