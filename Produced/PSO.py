import numpy as np 
import matplotlib.pyplot as plt 


def Cost(x):
    Z = np.linalg.norm(x) ** 2
    return Z


class Particle(object):
    def __init__(self):
        self.position = None
        self.velocity = None
        self.cost = None
        self.best = {"cost" : None, "position": None}



class PSO:
    def __init__(self, MaxItr = 100, nPop = 50):
        self.MaxItr = MaxItr

        self.nVar = 5
        self.VarMin = -10
        self.VarMax = 10
        self.VarSize = (1, self.nVar)

        self.nPop = 50

        self.w = 1
        self.wdamp = 0.99
        self.c1 = 2
        self.c2 = 2

        self.globalBest = {"cost" : np.inf, "position" : None}


    def initialise_particles(self):
        self.particle = np.array([Particle() for p in range(self.nPop)])
        for i in range(self.nPop):
            self.particle[i].position = np.random.uniform(low = self.VarMin, high = self.VarMax, size = self.VarSize)
            self.particle[i].velocity = np.zeros(self.VarSize)
            self.particle[i].cost = Cost(self.particle[i].position)

            self.particle[i].best["cost"] = self.particle[i].cost
            self.particle[i].best["position"] = self.particle[i].position

            if self.particle[i].best["cost"] < self.globalBest["cost"]:
                self.globalBest = self.particle[i].best

        self.BestCosts = np.zeros((self.MaxItr, 1))


    def optimise(self):
        self.initialise_particles()

        for i in range(self.MaxItr):
            for j in range(self.nPop):
                self.particle[j].velocity = self.w * self.particle[j].velocity \
                    + self.c1 * np.random.rand(*self.VarSize) * (self.particle[j].best["position"] - self.particle[j].position) \
                        + self.c2 * np.random.rand(*self.VarSize) * (self.globalBest["position"] - self.particle[j].position)

                self.particle[j].position = self.particle[j].position + self.particle[j].velocity
                self.particle[j].cost = Cost(self.particle[j].position)

                if self.particle[j].cost < self.particle[j].best["cost"]:
                    self.particle[j].best["cost"] = self.particle[j].cost
                    self.particle[j].best["position"] = self.particle[j].position

                    if self.particle[j].best["cost"] < self.globalBest["cost"]:
                        self.globalBest = self.particle[j].best

            self.BestCosts[i] = self.globalBest["cost"]
            self.w = self.w * self.wdamp

    
if __name__ == '__main__':
    particles = PSO()
    particles.optimise()

    print("Best parameters:", particles.globalBest)

    plt.plot(particles.BestCosts, label =  'Cost')
    plt.title("Best Costs")
    plt.legend()
    plt.grid()  
    plt.show()

        





        
