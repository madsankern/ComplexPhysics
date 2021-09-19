
import numpy as np
import matplotlib.pyplot as plt
import one_d_ising as ising

#########################
## Define param values ##
#########################
# N = 1000
# x_axis = range(N) # This is used to plot the lattice configuration

# Iteration parameters
n_max = 10000
n_0 = 8000


################
## Initialize ##
################

temperature = np.linspace(2.0, 3.0, 20) # Vector of values of inverse beta
mag = np.empty(len(temperature))
N_vals = np.linspace(100, 4000, 10)
T_c = np.empty(len(N_vals))

#####################
## Run Monte Carlo ##
#####################

# Loop over N values
for N_i,j in enumerate(N_vals):
    for i,kbT in enumerate(temperature):

        # Initialize with N_i sites
        state = ising.initialstate(N_i)
        mag_temp = np.zeros(n_max - (n_0 + 1)) + np.nan
        
        # Beta is the inverse temperature
        beta_val = 1.0/kbT

        # Run MC-MC
        for it in range(n_max):

            # Run Monte Carlo
            ising.montecarlo_random(state, beta_val)

            # Begin to store data after n0 mc steps
            if it > n_0:
                mag_temp[it - n_0 - 1] = ising.calcMag(state)

        # Save the resulting energy and magnetization
        mag[i] = np.abs(np.mean(mag_temp)) # Calculate the absolute magnetization as the solution is symmetric

        # Compute the critical temperature
        mag_reverse = mag[::-1]
        temperature_reverse = temperature[::-1]

        for i,m in enumerate(mag_reverse):
            if m > 0.1:
                T_c[j] = (temperature_reverse[i] + temperature_reverse[i-1])/2

                break   