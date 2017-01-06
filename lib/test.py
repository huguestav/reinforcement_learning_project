import simulator
import numpy as np
from allocation import uniform_allocation, bandit_allocation
from matplotlib import pyplot as plt
import KM

T = 10000
n_mc = 10

s = simulator.simulator("powerlaw", 2, 10, np.array([[0,2],[0.1,4],[0.4,8],[0.2,10],[0.2,1],[0.2,4]]))

np.random.seed(7)
#a = uniform_allocation(simulator=s, T=T, n_runs=n_mc)

np.random.seed(7)
#b = bandit_allocation(simulator=s, T=T, n_runs=n_mc, alpha=0.05)

#a = np.cumsum(a)
#b = np.cumsum(b)

#print "uniform:", a[-1]
#print "bandit:", b[-1]

np.random.seed(7)
c, tail_km = KM.KM(s,T,n_mc)
c = np.cumsum(c)
print "kaplan_meier:", c[-1]

np.random.seed(7)
d = KM.KM_optimistic(s,T,n_mc)

d = np.cumsum(d)
print "kaplan_meier optimistic:", d[-1]


#plt.plot(a)
#plt.plot(b)
plt.plot(c)
plt.plot(d)

plt.show()

