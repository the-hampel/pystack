#!/usr/bin/python
#written by A. Hampel 2015
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pylab import *
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, ScalarFormatter
import argparse
import sys
import os

parser = argparse.ArgumentParser(description='This programm postprocesses vasp calculations')

parser.add_argument("--path",help="path to vasp calculation",default=["./"],nargs='+')

parser.add_argument("--files",help="files with data to plot for DMFT calcs",nargs='+')

parser.add_argument("--bands",help="plotting bandstructure from vasp dir", action="store_true")

parser.add_argument("--wbands",help="plotting wannier bands from wannier90 calculation", action="store_true")

parser.add_argument("--tdos",help="plotting totdos from vasp dir", action="store_true")

parser.add_argument("--ldos",help="plotting local dos from given atoms (numbers) from vasp dir",default=None,nargs='+')

parser.add_argument("--ylim",nargs='+',default='')

parser.add_argument("--xlim",nargs='+',default='')

parser.add_argument("--dos_sum",help="want to have summation of spin resolved dos?",action="store_true")

parser.add_argument("--ldos_mult",help="local dos multiplicator for each ldos",default=None,nargs='+')

args = parser.parse_args()

#checking for arguments
if len(sys.argv)==1:
    parser.print_help()
    sys.exit(1)

#defining constants
fig_width_pt = 1.0*418.25368  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = 1.0*fig_width*golden_mean      # height in inches

#array with all the useful matplotlib colors
colors = ['k','b','r','g','c','m','#ff8400','y','k']

if args.bands or args.wbands:
  fig_size =  [1*fig_width,1*fig_height]
  #fig_size =  [2*fig_height,2*fig_width]
if args.tdos or args.ldos:
  fig_size =  [2*fig_width,1*fig_height]

#setting up canvas
params = {'backend': 'ps',
          'axes.labelsize': 12,
          'font.size': 12,
          'legend.fontsize': 12,
          'xtick.labelsize': 12,
          'ytick.labelsize': 12,
          'text.usetex': False,
          'figure.figsize': fig_size}
plt.rcParams.update(params)

plt.figure(1)
global ax

##################################################################################################

#defining functions
def ext_fermi_energy(i):
    "extracting the fermi energy"
    doscar = open(args.path[i]+'DOSCAR')
    for i in range(5):
         doscar.readline()
    energy = float(doscar.readline().split()[3])
    print 'efermi='+'',energy
    doscar.close()
    return energy;

def get_calc_name(i):
    doscar = open(args.path[i]+'DOSCAR')
    for i in range(4):
         doscar.readline()
    name = doscar.readline().strip().splitlines()
    print 'Name of your calculation:'+'',name
    doscar.close()
    return name;

def ext_rbasis(i):
    "Calculating the reciprocal basis"
    poscar = open(args.path[i]+"POSCAR")
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
    print 'basis of reciprocal lattice:'
    print basis
    return basis;

def ext_dos(i):
    #reading the file
    print 'opening now: '+args.path[i]+'DOSCAR'
    myfile=open(args.path[i]+'DOSCAR')
    data=myfile.readlines()
    myfile.close()


    #extract number of atoms and fermi energy
    zeile=data[0].split()
    natoms=int(zeile[0])
    zeile=data[5].split()
    efermi=float(zeile[3])
    nedos=int(zeile[2])

    print 'number of atoms:',natoms
    print 'fermi energy:',efermi,'eV'
    print 'number of points:',nedos

    #setting up the dos files
    total=open(args.path[i]+'total.dos','w')

    #total dos
    #total.write('!total dos decomposed\n')
    for x in range(nedos):
        zeile=data[6+x].split(None,1)
        en=float(zeile[0])-efermi
        total.write(str(en)+'  '+zeile[1])

    #partial dos of atoms
    for x in range(natoms):
        filename=args.path[i]+'atom_'+str(x+1)+'.dos'
        partialdos=open(filename,'w')
        for y in range(nedos):
            zeile=data[5+(1+nedos)*(x+1)+y+1].split(None,1)
            en=float(zeile[0])-efermi
            partialdos.write(str(en)+'  '+zeile[1])
        partialdos.close()
    return


def ext_bands(i):
    "extracting the band form the eignval file from VASP"
    efermi = ext_fermi_energy(i)
    basis = ext_rbasis(i)

    #reading the file
    myfile=open(args.path[i]+'EIGENVAL')
    data=myfile.readlines()
    myfile.close()

    #extraction of bands
    zeile=data[5].split()
    nkpoints=int(zeile[1])
    nbands=int(zeile[2])

    print 'number of kpoints:',nkpoints
    print 'number of bands:',nbands

    nmin=1
    nmax = nbands

    bands=open('bands.dat','w')

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
            zeile[1] = str(float(zeile[1])-efermi)
            value=value+'  '+zeile[1]
        bands.write(str(float(klen))+'  '+str(line)+'  '+value+'\n')

    bands.close()
    return

def plot_wannier():
    print "You're plotting",nplot,"wannier90 bandstructures for the following input files: ",args.path

    #Loading the BDSTRUC data
    DATA =[]
    for i in np.arange(0,nplot,1):
        DATA.append(np.loadtxt(args.path[i]+'wannier90_band.dat', unpack= True))

    ef = ext_fermi_energy(i)
    #plt.title(r"")
    plt.xlabel(r"Wave vector")
    plt.ylabel(r"Energy $\epsilon - \epsilon_f$ $(eV)$")

    # setting the ticks and limits
    minorLocatory   = MultipleLocator(0.5)
    majorLocatory   = MultipleLocator(1)
    ax.yaxis.set_minor_locator(minorLocatory)
    #ax.yaxis.set_major_locator(majorLocatory)
    if args.ylim:
        ax.axes.set_ylim([float(args.ylim[0]), float(args.ylim[1])])

    #finding and plotting the high symmetry lines and set there  xticks
    tk=[]
    if os.path.isfile(args.path[i]+'bands.dat'):
      bdstruc = loadtxt(args.path[i]+'bands.dat', unpack= True)
      tk.append(bdstruc[0,0])
      j=1
      for i in range(len(bdstruc[0,:])):
          if bdstruc[1,i]>j:
              ax.axvline(x=bdstruc[0,i], color='k')
              tk.append(bdstruc[0,i])
          j=bdstruc[1,i]
      tk.append(bdstruc[0,-1])
      plt.xticks(tk)
      ax.set_xticklabels(labels)
    else:
      print "WARNING: There is no Input for the high symmetry lines, so I use the normal xticks..."

    # manipulating the data for right plotting
    factor = bdstruc[0,-1] / DATA[0][0,-1]
    start=0
    end=0
    for i in arange(0,nplot,1):
      ax.plot(0,0, colors[i+1], linewidth = 1.0, label= get_calc_name(i)[0]+'_wan')
      for j in range(1,len(DATA[i][0,:])):
        if DATA[i][0,j] < DATA[i][0,j-1]:
          end = j
          ax.plot(DATA[i][0,start:end]*factor, DATA[i][1,start:end]-ef, colors[i+1], linewidth = 0.5)
          start = j
        if j+1 == len(DATA[i][0,:]):
          end = j+1
          ax.plot(DATA[i][0,start:end]*factor, DATA[i][1,start:end]-ef, colors[i+1], linewidth = 0.5)

def plot_bands():
    print "You're plotting",nplot,"bandstructure plot(s) with the following input files: ",args.path

    #Loading the BDSTRUC data
    DATA =[]
    for i in np.arange(0,nplot,1):
        DATA.append(np.loadtxt(args.path[i]+'bands.dat', unpack= True))

    #plt.title(r"")
    plt.xlabel(r"Wave vector")
    plt.ylabel(r"Energy $\epsilon - \epsilon_f$ $(eV)$")

    # setting the ticks and limits
    minorLocatory   = MultipleLocator(0.5)
    majorLocatory   = MultipleLocator(1)
    ax.yaxis.set_minor_locator(minorLocatory)
    #ax.yaxis.set_major_locator(majorLocatory)
    if args.ylim:
        ax.axes.set_ylim([float(args.ylim[0]), float(args.ylim[1])])

    #finding and plotting the high symmetry lines and set there  xticks
    tk=[]

    tk.append(DATA[0][0,0])
    j=1
    for i in range(len(DATA[0][0,:])):
      if DATA[0][1,i]>j:
          ax.axvline(x=DATA[0][0,i], color='k')
          tk.append(DATA[0][0,i])
      j=DATA[0][1,i]
    tk.append(DATA[0][0,-1])
    plt.xticks(tk)
    #setting the labels (don't forget to set the right labels)
    ax.set_xticklabels(labels)

    # line at fermi energy
    ax.axhline(y=0, color='k')

    #Plotting the data
    for i in np.arange(0,nplot,1):
      for j in range(2,len(DATA[i][:,0])):
        if j == 2:
          ax.plot(DATA[i][0,:], DATA[i][j,:], colors[i], linewidth = 0.7, label= get_calc_name(i)[0])
        else:
          ax.plot(DATA[i][0,:], DATA[i][j,:], colors[i], linewidth = 0.7)
    return

def plot_tdos():
    print "You're plotting",nplot,"DOS plot(s) with the following input files: ",args.path

    colors = ['k','b','r','g','c','m','#ff8400','y','k']

    ax = subplot(111)
    #Loading the DOS data

    DATA =[]
    for i in np.arange(0,nplot,1):
        DATA.append(np.loadtxt(args.path[i]+'total.dos', unpack= True))

    if args.xlim:
        ax.axes.set_xlim([float(args.xlim[0]), float(args.xlim[1])])
    if args.ylim:
        ax.axes.set_ylim([float(args.ylim[0]), float(args.ylim[1])])

    plt.xlabel(" ")
    plt.ylabel(r"$\rho$ $(\frac{states}{eV})$")
    plt.grid(True)


    majorLocatorx   = MultipleLocator(1)
    majorFormatterx = FormatStrFormatter('%d')
    minorLocatorx   = MultipleLocator(0.5)
    ax.xaxis.set_major_locator(majorLocatorx)
    ax.xaxis.set_minor_locator(minorLocatorx)

    for i in arange(0,nplot,1):
        ax.axvline(x=0, color='k')
        ax.axhline(y=0, color='k')
        if len(DATA[i][:,0]) == 3:
            plt.plot(DATA[i][0,:], DATA[i][1,:], linewidth = 1.5,color=colors[i],label=get_calc_name(i)[0])
        if len(DATA[i][:,0]) == 5:
            print 'spin-resolved DOS'
            if args.dos_sum:
                plt.plot(DATA[i][0,:], DATA[i][1,:]+DATA[i][2,:], linewidth = 1.5,color=colors[i],label=get_calc_name(i)[0])
            else:
                plt.plot(DATA[i][0,:], DATA[i][1,:], linewidth = 1.5,color=colors[i],label=get_calc_name(i)[0])
                plt.plot(DATA[i][0,:], -DATA[i][2,:], linewidth = 1.5,color=colors[i])
    return



def plot_ldos():
    print "You're plotting",nplot,"local DOS plot(s) with the following input files: ",args.path

    colors = ['c','m','b','r','g','#ff8400','y','k']
    ax = subplot(111)

   #Loading the DOS data
    DATA =[]
    for i in np.arange(0,nplot,1):
        efermi = ext_fermi_energy(0)
        DATA.append(np.loadtxt(args.path[0]+'atom_'+args.ldos[i]+'.dos', unpack= True))

    if args.xlim:
        ax.axes.set_xlim([float(args.xlim[0]), float(args.xlim[1])])
    if args.ylim:
        ax.axes.set_ylim([float(args.ylim[0]), float(args.ylim[1])])

    plt.xlabel(" ")
    plt.ylabel(r"$\rho$ $(\frac{states}{eV})$")
    plt.grid(True)

    majorLocatorx   = MultipleLocator(1)
    majorFormatterx = FormatStrFormatter('%d')
    minorLocatorx   = MultipleLocator(0.5)
    ax.xaxis.set_major_locator(majorLocatorx)
    ax.xaxis.set_minor_locator(minorLocatorx)
    sum_dos= []
    sum_dos_up= []
    sum_dos_down= []
    fermi_index = 0
    for i in range(0,len(DATA[0][0,:])):
      if (np.abs(DATA[0][0,i]) - efermi) < 0.001:
        fermi_index = i
        break
      if i == len(DATA[0][0,:])-1:
        raise ValueError( "error: cant find fermi level" )
    if args.ldos_mult:
        if len(args.ldos_mult) != len(args.ldos):
            print "Error: Please ensure that you give one mutliplier for each ldos!"
            sys.exit(1)
        for i in arange(0,nplot,1):
            sum_dos.append(0.0)
            sum_dos_up.append(0.0)
            sum_dos_down.append(0.0)
            ax.axvline(x=0, color='k')
            ax.axhline(y=0, color='k')
            if len(DATA[i][:,0]) == 10:
                plt.plot(DATA[i][0,:]-efermi, float(args.ldos[i])*DATA[i][1,:], linewidth = 1.5,color=colors[i],label=get_calc_name(0)[0]+'-atom_'+args.ldos[i])
                sum_dos[i] = sum_dos[i] + np.sum(DATA[i][1,:])
            if len(DATA[i][:,0]) == 19:
                print 'spin-resolved DOS'
                plt.plot(DATA[i][0,:], float(args.ldos_mult[i])*(DATA[i][1,:]+ DATA[i][3,:]+ DATA[i][5,:]+ DATA[i][7,:]+ DATA[i][9,:]+ DATA[i][11,:]+ DATA[i][13,:]+ DATA[i][15,:]+ DATA[i][17,:]), linewidth = 1.5,color=colors[i],label=get_calc_name(0)[0]+'-atom_'+args.ldos[i])
                plt.plot(DATA[i][0,:], -float(args.ldos_mult[i])*(DATA[i][2,:]+ DATA[i][4,:]+ DATA[i][6,:]+ DATA[i][8,:]+ DATA[i][10,:]+ DATA[i][12,:]+ DATA[i][14,:]+ DATA[i][16,:]+ DATA[i][18,:]), linewidth = 1.5,color=colors[i])
                for l in arange(1,len(DATA[i][0,0:fermi_index])-1,1):
                    sum_dos_up[i] = sum_dos_up[i] + ( (DATA[i][1,l]+ DATA[i][3,l]+ DATA[i][5,l]+ DATA[i][7,l]+ DATA[i][9,l]+ DATA[i][11,l]+ DATA[i][13,l]+ DATA[i][15,l]+ DATA[i][17,l])  * ( abs(DATA[i][0,l+1] - DATA[i][0,l]))  ) #  delta ei 
                    sum_dos_down[i] = sum_dos_down[i] + ( (DATA[i][2,l]+ DATA[i][4,l]+ DATA[i][6,l]+ DATA[i][8,l]+ DATA[i][10,l]+ DATA[i][12,l]+ DATA[i][14,l]+ DATA[i][16,l]+ DATA[i][18,l])  * ( abs(DATA[i][0,l+1] - DATA[i][0,l]))  ) #  delta ei 
        print "integrated pdos for spin-up:", sum_dos_up
        print "integrated pdos for spin-down:", sum_dos_down
    else:
        for i in arange(0,nplot,1):
            sum_dos.append(0.0)
            ax.axvline(x=0, color='k')
            ax.axhline(y=0, color='k')
            if len(DATA[i][:,0]) == 10:
                plt.plot(DATA[i][0,:]-efermi, DATA[i][1,:], linewidth = 1.5,color=colors[i],label=get_calc_name(0)[0]+'-atom_'+args.ldos[i])
                sum_dos[i] = sum_dos[i] + np.sum(DATA[i][1,:])
            if len(DATA[i][:,0]) == 19:
                print 'spin-resolved DOS'
                plt.plot(DATA[i][0,:], DATA[i][1,:]+ DATA[i][3,:]+ DATA[i][5,:]+ DATA[i][7,:]+ DATA[i][9,:]+ DATA[i][11,:]+ DATA[i][13,:]+ DATA[i][15,:]+ DATA[i][17,:], linewidth = 1.5,color=colors[i],label=get_calc_name(0)[0]+'-atom_'+args.ldos[i])
                plt.plot(DATA[i][0,:], -(DATA[i][2,:]+ DATA[i][4,:]+ DATA[i][6,:]+ DATA[i][8,:]+ DATA[i][10,:]+ DATA[i][12,:]+ DATA[i][14,:]+ DATA[i][16,:]+ DATA[i][18,:]), linewidth = 1.5,color=colors[i])
                sum_dos[i] = sum_dos[i] + np.sum(DATA[i][1:,:])
    return

##################################################################################################

ortho=["Z",r"$\Gamma$","X","M",r"$\Gamma$","Y"]
labels = ortho

if args.bands:
    nplot = len(args.path)
    ax = subplot(111)
    for i in range(nplot):
        ext_bands(i)
    plot_bands()
    if args.wbands:
        plot_wannier()

if args.wbands and args.bands==False:
    nplot = len(args.path)
    ax = subplot(111)
    plot_wannier()

if args.tdos:
    nplot = len(args.path)
    for i in range(nplot):
        ext_dos(i)
    plot_tdos()

if args.ldos:
    nplot = len(args.path)
    if nplot > 1:
        print "warning ldos is only implementet for one calculation at a time"
        sys.exit(1)
    nplot = len(args.ldos)
    ext_dos(0)
    plot_ldos()

##################################################################################################

# Now add the legend with some customizations.
plt.legend(loc='upper right', shadow=True, fancybox=True, ncol=1,labelspacing=0.07,borderaxespad=0.3,borderpad=0.3,handletextpad=0.07)

#and finally:
plt.savefig('plot.pdf', bbox_inches='tight',transparent=True, pad_inches=0.05)
#plt.savefig('plot.jpg',dpi=200, bbox_inches='tight', pad_inches=0.05)
plt.show()
