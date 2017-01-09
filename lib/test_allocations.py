import simulator
import numpy as np
from allocation import *
from exp3 import exp3_allocation, exp3_allocation_change, exp3_allocation_swap
from matplotlib import pyplot as plt
import KM


V = 100
a = [0.2,0.01]
b = [0.8,1.2]

venues_params_1 = np.array([a,b])
venues_params_2 = np.array([b,a])
n_venues = venues_params_1.shape[0]
s1 = simulator.simulator("powerlaw", n_venues, V, venues_params_1)
s2 = simulator.simulator("powerlaw", n_venues, V, venues_params_2)

from KM import greedyAllocation
tail_probas = np.empty((s1.n_venues, s1.V_max+1))
for k in range(s1.n_venues):
    tail_probas[k,:] = s1.venues[k].tail_distrib()

alloc = greedyAllocation(tail_probas, V)
print "optimal alloc", alloc

s1.venues[0].mean_value()
s1.venues[1].mean_value()



T = 2500


# eta = (V * np.log(n_venues)**2 / n_venues * T**(-2))**(1./3)
# print "eta:", eta
# # eta = 0.01
# gamma = 0.01

# rewards, allocations = exp3_allocation_change(s1, s2, T, eta, gamma)


# rewards, alloc = KM.KM_change(s1, s2, T)

_,allocations = KM.KM_optimistic_swap(s1,2*T)
_, allocations_exp3 = exp3_allocation_swap(s1,2*T)

plt.plot(allocations[0,:])
plt.plot(allocations[1,:])

plt.plot(allocations_exp3[0,:])
plt.plot(allocations_exp3[1,:])

#plt.plot(100 * np.ones(2*T))
plt.show()


