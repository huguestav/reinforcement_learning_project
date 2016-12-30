import simulator
import numpy as np
from allocation import uniform_allocation



s = simulator.simulator("powerlaw", 2, 10, np.array([[0.5,0.3],[0.2,0.5]]))


uniform_allocation(simulator=s, T=20, n_runs=11)





p = np.array([[0.1,0.3],[0.2,0.5]]