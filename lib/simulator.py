
from venue import venue
import copy
import random

class simulator():
    def __init__(self, sim, n_venues, V_max, parameters):
        self.n_venues = n_venues
        self.V_max = V_max

        venues = []
        for i in range(n_venues):
        	v = venue(distrib=sim, parameters=parameters[i], V_max=V_max)
        	venues.append(v)

        self.venues = venues


    def swap_venues(self,idx1,idx2):
    	tmp = copy.copy(self.venues[idx1])
    	self.venues[idx1] = copy.copy(self.venues[idx2])
    	self.venues[idx2]  = tmp
    	return

    def random_swap_venues(self):
    	if self.n_venues == 1:
    		return
    	# choose random indexes
    	idx1,idx2 = random.sample(range(self.n_venues),2)
    	self.swap_venues(idx1,idx2)
    	return
