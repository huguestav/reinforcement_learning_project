import simulator
import numpy as np
import ML
from allocation import uniform_allocation
from matplotlib import pyplot as plt


np.random.seed(7)

sim = simulator.simulator("powerlaw", 2, 10, np.array([[0.9,0.3],[0.2,0.3]]))

n_venues = sim.n_venues

D = np.zeros((n_venues,sim.V_max+1))
N = np.zeros((n_venues,sim.V_max+1))
# first iteration with a uniform allocation
# Do the euclidean division of V by n_venues
p = sim.V_max / n_venues
r = sim.V_max - p * n_venues

# Allocate uniformly to the different venues
arr = np.arange(n_venues)
np.random.shuffle(arr)
alloc = p * np.ones(n_venues) + (arr < r)

print alloc

rewards = ML.get_reward(alloc,sim,D,N)
# print rewards

T = ML.compute_T(D,N) # reestimate the tail probabilities

# print "D"
# print D
# print "N"
# print N
# print "T"
# print T

allocation = ML.greedyAllocation(T,sim.V_max) # do the allocation accordingly
reward = ML.get_reward(allocation,sim,D,N)

print allocation
#V = simulator.V_max # nothing done on the maximum allocation for the moment
#allocation = greedyAllocation(T,V) # do the allocation accordingly
#reward = get_reward(allocation,simulator,D,N)
