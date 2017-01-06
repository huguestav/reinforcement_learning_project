import simulator
import numpy as np
import ML
from allocation import uniform_allocation
from matplotlib import pyplot as plt


np.random.seed(7)

sim = simulator.simulator("powerlaw", 2, 10, np.array([[0.9,0.3],[0.2,0.3]]))



#### test greedy allocation ####
V = 8
pools = np.zeros([4,9])

pools[0,:] = 10*np.ones(9)
pools[1,:] = np.array(range(8,-1,-1))
pools[2,:] = np.array(range(8,-1,-1))
pools[3,:] = np.array(range(8,-1,-1))

print ML.greedyAllocation(pools,V)
################################

print ML.KM(sim,1000,100).shape
