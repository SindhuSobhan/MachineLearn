########################################
##............... PSO ................##
########################################


# Import Libraries
import numpy as np 
import matplotlib.pyplot as plt 

# Define Objective/Cost fucntion
def Cost(x):
    Z = np.linalg.norm(x) ** 2
    return Z

# Define Particle Class
class Particle(object):
    def __init__(self):
        self.position = None                            # Particle Position
        self.velocity = None                            # Particle Velocity
        self.cost = None                                # Cost associated with particle
        self.best = {"cost" : None, "position": None}   # Best Particle position and cost


# Define the optimisation algorithm
class PSO:
    def __init__(self, MaxItr = 100, nPop = 50, VarMax = 10, VarMin = -10, Vel_damp = 0.02 ):
        self.MaxItr = MaxItr                            # Maximum iteration

        self.nVar = 100                                # Number of unknown variables 
        self.VarMin = VarMin                            # Minimum value of Particle Position
        self.VarMax = VarMax                            # Maximum value of Particle Position
        self.VarSize = (1, self.nVar)                   # Dimension of each variable
        self.VelMax = Vel_damp * (VarMax - VarMin)      # Minimum value of Particle Velocity
        self.VelMin = - self.VelMax                     # Minimum value of Particle Velocity

        self.nPop = 50                                  # Number of Paricles (Population)

        self.w = 1                                      # Inertia velocity weight 
        self.wdamp = 0.99                               # Damping ratio for inertia
        self.c1 = 2                                     # Acceleration coefficient 1
        self.c2 = 2                                     # Acceleration Coefficient 2

        # Best Particle (Best particle in the particle population)
        self.globalBest = {"cost" : np.inf, "position" : None} 


    # Initialise Particles
    def initialise_particles(self):
        # Create particles
        self.particle = np.array([Particle() for p in range(self.nPop)])
        # Define particles
        for i in range(self.nPop):
            # Initialise particle position
            self.particle[i].position = np.random.uniform(low = self.VarMin, high = self.VarMax, size = self.VarSize)
            # Initialise particle velocity
            self.particle[i].velocity = np.zeros(self.VarSize)
            # Initialise particle cost
            self.particle[i].cost = Cost(self.particle[i].position)
            
            # Initialise best particle parameters
            self.particle[i].best["cost"] = self.particle[i].cost
            self.particle[i].best["position"] = self.particle[i].position

            # Update global best if particle best better than global best
            if self.particle[i].best["cost"] < self.globalBest["cost"]:
                self.globalBest = self.particle[i].best

        self.BestCosts = np.zeros((self.MaxItr, 1))     # Initialise Best Cost for each iteration


    # Define optimisation function
    def optimise(self):
        self.initialise_particles()                     # Call functions to initialise particles 

        # Run for the number of specified iterations
        for i in range(self.MaxItr):
            # Run for all particles in the population
            for j in range(self.nPop):
                # Update the velocity of the jth particle
                self.particle[j].velocity = self.w * self.particle[j].velocity \
                    + self.c1 * np.random.rand(*self.VarSize) * (self.particle[j].best["position"] - self.particle[j].position) \
                        + self.c2 * np.random.rand(*self.VarSize) * (self.globalBest["position"] - self.particle[j].position)
                # Ensure that the velocity of the jth particle is within the limits specified for velocity
                self.particle[j].velocity = np.maximum(self.particle[j].velocity, self.VelMin)
                self.particle[j].velocity = np.minimum(self.particle[j].velocity, self.VelMax)
                
                # Update the position of the jth particle
                self.particle[j].position = self.particle[j].position + self.particle[j].velocity
                # Ensure that the position of the 
                self.particle[j].position = np.maximum(self.particle[j].position, self.VarMin)
                self.particle[j].position = np.minimum(self.particle[j].position, self.VarMax)

                # Update the cost for particle j
                self.particle[j].cost = Cost(self.particle[j].position)

                # Update best particle position
                if self.particle[j].cost < self.particle[j].best["cost"]:
                    self.particle[j].best["cost"] = self.particle[j].cost
                    self.particle[j].best["position"] = self.particle[j].position
                    # Update global best if particle best better than global best
                    if self.particle[j].best["cost"] < self.globalBest["cost"]:
                        self.globalBest = self.particle[j].best

            # Update Best Costs after each iteration
            self.BestCosts[i] = self.globalBest["cost"]
            # Dampen the Inertia weight
            self.w = self.w * self.wdamp


# Run the PSO class    
if __name__ == '__main__':
    particles = PSO()
    particles.optimise()

    # Print globally best parameters
    print("Best parameters:", particles.globalBest)

    # Plot the best cost values
    plt.semilogy(particles.BestCosts, label =  'Cost')
    plt.title("Best Costs")
    plt.legend()
    plt.grid()  
    plt.show()