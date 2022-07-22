import sys
import glob

import solid_dmft.postprocessing.maxent_sigma as sigma_maxent

h5_file = glob.glob(sys.argv[1])[0]

# use pcb maxent script to continue sigma, self energy stored in the archive again
Sigma_real_freq = sigma_maxent.main(external_path=h5_file,
                                    omega_min=-10, omega_max=10,
                                    maxent_error=0.03, iteration=None,
                                    n_points_maxent=101,
                                    n_points_alpha=50,
                                    analyzer='LineFitAnalyzer',
                                    n_points_interp=2001,
                                    n_points_final=1001,
                                    continuator_type='inversion_dc')[0]
