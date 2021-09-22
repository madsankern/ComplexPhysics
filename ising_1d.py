###############################
## Code used for the midterm ##
###############################
# Implements the one-dimensional Ising model with either 2 or 4 nearest neighbours

# Import packages
import numpy as np
import numba as nb

# Generate initial state
def initialstate(N):
    '''Generates a random spin configuration on an 1-dimensional lattice'''

    # Randomly draw +- 1
    state = 2*np.random.randint(2, size = N) - 1

    return state

# Function to assign the random neighbours
@nb.njit
def gen_neighbours(N):
    '''Generate two pairs of random neighbours for each site'''

    # List of all sites in configuration
    sites1 = np.arange(0,N)
    sites2 = np.arange(0,N)

    # Initialize arrays with neighbours
    # Each column consists of one random pair
    pairs1 = np.empty( (2, int(N/2)) )
    pairs2 = np.empty( (2, int(N/2)) )

    # Loop over bonds
    for i in range(int(N/2)):
        
        # Draw two random site interactions
        n11,n12 = np.random.choice(sites1, 2, replace=False)
        pairs1[:,i] = n11,n12

        n21,n22 = np.random.choice(sites2, 2, replace=False)
        pairs2[:,i] = n21,n22
        
        # Remove the chosen elements from list
        sites1 = sites1[(sites1!=n11) & (sites1!=n12)]
        sites2 = sites2[(sites2!=n21) & (sites2!=n22)]

    # Return random neighbours
    return pairs1,pairs2


##############################
## Markov chain Monte Carlo ##
##############################

# Simulates the Ising model using metropolis hastings
@nb.njit
def montecarlo(config, beta, J=1):
    '''Monte Carlo using the Metropolis-Hastings algorithm'''

    # Calculate the lattice size
    N = config.shape[0]
    size = config.size

    # Perform N iterations, corresponding to one Monte Carlo update
    for _ in range(size):

        # Unpackage flip candidate
        i = np.random.randint(N)

        # Compute energy of current configuration
        E_stay = -J*config[i]*(
            config[(i+1)%N] + # %N ensures a closed ring
            config[(i-1)%N]
            )

        # Energy when flipping the spin is just minus the current energy
        E_flip = - E_stay

        # Determine whether to flip the spin or not using Metropolis-Hastings
        if np.random.rand() < np.exp(-beta*(E_flip - E_stay)) :
            config[i] *= -1
                
    return config

# Monte Carlo using the random neighbour model
@nb.njit
def montecarlo_random(config, beta, J=1):
    '''Monte Carlo move using the Metropolis-Hastings algorithm for random neighbour model'''

    # Calculate the lattice size
    N = config.shape[0]
    size = config.size

    # Perform N iterations, corresponding to one Monte Carlo update
    for j in range(size):
                
        # Unpackage flip candidate
        i = np.random.randint(N)
        
        # Draw random neighbours
        nb1,nb2 = np.random.randint(N), np.random.randint(N)

        # Compute energy of current configuration
        E_stay = -J*config[i]*(
            config[(i+1)%N] + # %N ensures a closed ring
            config[(i-1)%N] +
            config[nb1] + # Add random neighbours
            config[nb2]
            )

        # Energy when flipping the spin is just minus the current energy
        E_flip = - E_stay

        # Determine whether to flip the spin or not using Metropolis-Hastings
        if np.random.rand() < np.exp(-beta*(E_flip - E_stay)) :
            config[i] *= -1
                
    return config

###########################
## Calculate observables ##
###########################

# Average energy of lattice configuration
@nb.njit
def calcEnergy(config, J=1):
    '''Average energy of a given configuration'''
    
    # Initialize
    energy = 0

    # Compute size of the system
    N = config.size
    
    # Loop over all sites
    for i in range(len(config)):
        s = config[i]
        nb = config[(i+1)%N] + config[(i-1)%N]

        # Energy equals the Hamiltonian. Divide by 2 to avoid double counting (each site has 2 neighbours)
        energy += (-J*nb*s / 2.0) / N
            
    return energy

# Magnetization
@nb.njit
def calcMag(config):
    '''Magnetization per site of a given configuration'''

    # Compute number of lattice sites
    N = config.size

    # Average magnetization is simply the sum of all spins in the lattice divided by # of elements
    mag = np.sum(config)/N

    return mag