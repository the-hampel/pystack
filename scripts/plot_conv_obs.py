#!/mnt/home/ahampel/.local/bin/python

import sys
import glob
import tkinter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
sys.path.append("/mnt/home/ahampel/Dropbox/git/hampel-pystack")
from tools import set_plot_params
from plot_triqs import plot_conv_obs, plot_G_S

set_plot_params()


h5_file = glob.glob(sys.argv[1])[0]

plot_conv_obs(h5_file)
