
from venue import venue


class simulator():
    def __init__(self, sim, n_venues, V_max, parameters):
        self.n_venues = n_venues
        self.V_max = V_max

        venues = []
        for i in range(n_venues):
        	v = venue(distrib=sim, parameters=parameters[i], V_max=V_max)
        	venues.append(v)

        self.venues = venues
