import simulator
import numpy as np
from allocation import *
from matplotlib import pyplot as plt
import KM


V = 20
venues_params = np.array([[0.7,0.3],[0.2,0.7], [0.5,0.3]])
n_venues = venues_params.shape[0]
s = simulator.simulator("powerlaw", n_venues, V, venues_params)

T = 1000
n_runs = 100

from ML import greedyAllocation
tail_probas = np.empty((s.n_venues, s.V_max+1))
for k in range(s.n_venues):
    tail_probas[k,:] = s.venues[k].tail_distrib()

alloc = greedyAllocation(tail_probas, V)
print alloc



reward_1 = uniform_allocation(simulator=s, T=T, n_runs=n_runs)
reward_2 = bandit_allocation(simulator=s, T=T, n_runs=n_runs, alpha=0.05)
reward_3 = optimal_allocation(simulator=s, T=T, n_runs=n_runs)

reward_1 = np.cumsum(reward_1)
reward_2 = np.cumsum(reward_2)
reward_3 = np.cumsum(reward_3)

regret_1 = reward_3 - reward_1
regret_2 = reward_3 - reward_2

# regret_1 = min(sum(mean_values), V) * (np.arange(T) + 1) - reward_1
# regret_2 = min(sum(mean_values), V) * (np.arange(T) + 1) - reward_2

# print "optimal:", reward_3[-1]
# print "bandit:", reward_2[-1]

plt.plot(regret_1)
plt.plot(regret_2)
plt.show()

#plt.plot(a)
#plt.plot(b)
plt.plot(c)
plt.plot(d)

plt.show()

