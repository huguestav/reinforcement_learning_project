import simulator
import numpy as np
from allocation import uniform_allocation, bandit_allocation
from matplotlib import pyplot as plt


s = simulator.simulator("powerlaw", 2, 10, np.array([[0.2,0.3],[0.2,0.3]]))


a = uniform_allocation(simulator=s, T=1000, n_runs=10)

b = bandit_allocation(simulator=s, T=1000, n_runs=10, alpha=0.05)

print np.sum(a)
print np.sum(b)


a = np.cumsum(a)
b = np.cumsum(b)

print "uniform:", a[-1]
print "bandit:", b[-1]

plt.plot(a)
plt.plot(b)
plt.show()


