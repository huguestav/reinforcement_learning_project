import simulator
import numpy as np
import ML
from allocation import uniform_allocation
from matplotlib import pyplot as plt


np.random.seed(7)

sim = simulator.simulator("powerlaw", 2, 10, np.array([[0.2,0.3],[0.2,0.3]]))

n_venues = sim.n_venues

rewards2 = ML.KM(sim,T=10000,mc=10)
rewards1 = ML.KM_optimistic(sim,T=10000,mc=10)

plt.plot(np.cumsum(rewards2))
plt.plot(np.cumsum(rewards1))

print np.cumsum(rewards2)[-1]
print np.cumsum(rewards1)[-1]

plt.show()