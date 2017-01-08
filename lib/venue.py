import numpy as np
from scipy.special import factorial

def draw_finite(p):
    """
    Draw a sample from a distribution on a finite set
    that takes value k with probability p(k)
    """
    q = np.cumsum(p)
    u = np.random.random()
    i = 0
    while u > q[i]:
        i += 1
    return i



class venue():
    def __init__(self, distrib, parameters, V_max):
        """
        Parameters is a the venue ex:
        powerlaw : [zb, beta]
        poisson : [zb, lambda]
        """
        self.distrib = distrib
        self.parameters = parameters
        self.V_max = V_max


    def mean_value(self):
        parameters = self.parameters
        if self.distrib == "powerlaw":
            zb = parameters[0]
            beta = parameters[1]

            non_null_proba = 1. / np.power(np.arange(self.V_max)+1, beta)
            non_null_proba = (1-zb) * non_null_proba / np.sum(non_null_proba)

            return non_null_proba.T.dot(np.arange(self.V_max)+1)


    def draw(self):
        parameters = self.parameters

        if self.distrib == "powerlaw":
            zb = parameters[0]
            beta = parameters[1]
            if (np.random.random() < zb):
                return 0
            else:
                proba = 1. / np.power(np.arange(self.V_max)+1, beta)
                proba = proba / np.sum(proba)
                return draw_finite(proba) + 1

        elif self.distrib == "poisson":
            zb = parameters[0]
            lamb = parameters[1]
            if (np.random.random() < zb):
                return 0
            else:
                proba = np.power(np.arange(self.V_max)+1, lamb)
                proba = proba / factorial(np.arange(self.V_max)+1)
                proba = proba / np.sum(proba)
                return draw_finite(proba) + 1

    def tail_distrib(self):
        parameters = self.parameters

        if self.distrib == "powerlaw":
            zb = parameters[0]
            beta = parameters[1]

            non_null_proba = 1. / np.power(np.arange(self.V_max)+1, beta)
            non_null_proba = (1-zb) * non_null_proba / np.sum(non_null_proba)

            tail_non_null = np.cumsum(non_null_proba[::-1])[::-1]

            tail = np.ones(self.V_max+1)
            tail[1:] = tail_non_null

            return tail



# import numpy as np
# import venue
# v = venue.venue("powerlaw", [0.7, 0.3], 50)
# v.draw()
# v.tail_distrib()




