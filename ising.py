#################################################################
## Collection of functions used for simulating the Ising model ##
#################################################################

# Import packages
import numpy as np
import numba as nb

###################
## Initial state ##
###################

# Randomly draws either -1 or +1 on a NxN array. N is the row/column length of the lattice
def initialstate(N):
    
    '''Generates a random spin configuration for initial condition'''
    
    state = 2*np.random.randint(2, size=(N,N)) - 1
    
    return state

##############################
## Markov chain Monte Carlo ##
##############################

# Updates are performed using the Metropolis-Hastings algorithm
@nb.njit
def mcmove(config, beta, J=1, h=0):
    '''Monte Carlo move using the Metropolis-Hastings algorithm'''

    # Calculate the lattice size
    N = config.shape[0] # Assuming the lattice is square
    size = config.size

    # Perform N^2 iterations, corresponding to one Monte Carlo update
    for _ in range(size):
                
            # Draw a random site, by drawing random row and column number
            i,j = np.random.randint(0,N), np.random.randint(0,N)

            # Define the lattice site chosen
            # s =  config[i, j]

            # Compute energy of current configuration
            E_stay = (
                -J*config[i,j]*(
                    config[(i + 1)%N, j] + 
                    config[i, (j + 1)%N] + 
                    config[(i - 1)%N, j] + 
                    config[i, (j - 1)%N])
                    - h*config[i,j]
            )

            # Energy when flipping the spin is just minus the current energy
            E_flip = - E_stay

            # Determine whether to flip the spin or not using Metropolis-Hastings
            if np.random.rand() < np.exp(-beta*(E_flip - E_stay)) :
                config[i,j] *= -1

            # Update lattice site
            # config[i,j] = s
                
    return config

###########################
## Calculate observables ##
###########################

# Calculate energy of lattice configuration
@nb.njit
def calcEnergy(config, J=1, h=0):
    '''Average energy of a given configuration'''
    
    # Initialize
    energy = 0

    # Compute size of the system
    N = config.shape[0] # Assuming the lattice is square
    size = config.size
    
    # Loop over all sites
    for i in range(len(config)):
        for j in range(len(config)):
            
            s = config[i,j]
            nb = config[(i+1)%N, j] + config[i, (j+1)%N] + config[(i-1)%N, j] + config[i,(j-1)%N]
            
            # Energy is from the Hamiltonian. Divide by 4 to avoid double counting (each site has 4 neighbours)
            energy += (-J*nb*s / 4.0 - h*s) / size
            
    return energy

# Magnetization
@nb.njit
def calcMag(config):
    '''Magnetization per site of a given configuration'''
    
    # Compute number of lattice sites
    size = config.size

    # Average magnetization is simply the sum of all spins in the lattice divided by # of elements
    mag = np.sum(config)/size

    return mag