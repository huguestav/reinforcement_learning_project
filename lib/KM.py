### this python file regroups the machine learning algorithms for the dark pool problem
import numpy as np
import math

######################################
############# GENERAL ################
######################################


def greedyAllocation(tail_probas,V):
# input: (precalculated) tail probabilities as a numpy array and V, the volume to allocate

    n_venues = tail_probas.shape[0]
    assert (tail_probas.shape[1] > V), 'not enough precalculated tail probas'

    allocation = np.zeros(n_venues,dtype='int')

    for v in range(V):
        # a la mano method, otherwise results in a argmax problem
        tmp = np.zeros(n_venues)

        for a in range(len(tmp)):
            tmp[a] = tail_probas[a,allocation[a]+1]

        idx = np.argwhere(tmp==np.max(tmp))

        i = idx[np.random.randint(len(idx))]

        allocation[i] += 1

    return allocation


def compute_T(D,N):
    # D[n_venues,V_max+1] where V_max is the maximum possible allocation
    # N[n_venues,V_max+1] where V_max is the maximum possible allocation
    assert D.shape == N.shape, 'D and N array dimension mismatch'

    NN = np.copy(N)

    idxs = (NN == 0)
    NN[idxs] = float('inf')

    z = D/NN
    zz = 1-z

    T = np.ones(zz.shape)

    T[:,1:] = np.cumprod(zz, axis=1)[:,:-1]

    return T

def get_reward(allocation,simulator,D,N):
    # D and N are passed a reference and modified directly
    reward = np.zeros(len(allocation))
    venues = simulator.venues

    assert(len(reward) == len(venues)), "dimension mismatch in get_reward"
    assert(len(allocation) == len(venues)), "dimension mismatch in get_reward"

    for k in range(len(venues)):
        sim_s = venues[k].draw()
        reward[k] = min(allocation[k],sim_s)

        for s in range(simulator.V_max + 1):
            if (reward[k] >= s and allocation[k] > s):
                if (reward[k] == s):
                    D[k,s] += 1
                N[k,s] += 1

    return sum(reward)



##############################################
############## KAPLAN-MEIER ##################
##############################################

def KM_iter(D,N,simulator):
    #print "D"
    #print D
    #print "N"
    #print N
    T = compute_T(D,N) # reestimate the tail probabilities
    V = simulator.V_max # nothing done on the maximum allocation for the moment
    allocation = greedyAllocation(T,V) # do the allocation accordingly
    reward = get_reward(allocation,simulator,D,N)
    return reward

def KM(simulator,T,mc):
    n_venues = len(simulator.venues)

    rewards = np.zeros(T)

    for m in range(mc):

        D = np.zeros([n_venues,simulator.V_max+1])
        N = np.zeros([n_venues,simulator.V_max+1])

        # first iteration with a uniform allocation
        # Do the euclidean division of V by n_venues
        p = simulator.V_max / n_venues
        r = simulator.V_max - p * n_venues

        # Allocate uniformly to the different venues
        arr = np.arange(n_venues)
        np.random.shuffle(arr)
        alloc = p * np.ones(n_venues) + (arr < r)

        rewards[0] += get_reward(alloc,simulator,D,N)

        for t in range(1,T):
            rewards[t] += KM_iter(D,N,simulator)

        tail = compute_T(D,N)

    return rewards/mc, tail


###################################################
############ KAPLAN-MEIER OPTIMISTIC ##############
###################################################
epsilon = 0.5
delta = 0.5


def thr(s,V,epsilon,delta):
    return 128*(float(s*V)/(epsilon))**2 * math.log(2*float(V)/float(delta))

def OptimisticKM(D,N,epsilon,delta):
    #this subroutine is meant to restimate the tail probabilities
    n_venues = N.shape[0]
    Vp1 = N.shape[1]
    T = compute_T(D,N)

    # compute the cut-off point for each venue
    c = np.zeros(n_venues,dtype='int')

    for i in range(n_venues):
        for s in range(Vp1-1,-1,-1):
            if s == 0:
                break
            if N[i,s-1] >= thr(s,Vp1-1,epsilon,delta):
                break
        c[i] = s

    # now that we have the cut-off points, we can modify the T accordingly
    for i in range(n_venues):
        if c[i] < Vp1-1:
            T[i,c[i]+1] = T[i,c[i]]
    return T


def KM_iter_optimistic(D,N,simulator,epsilon,delta):
    T = OptimisticKM(D,N,epsilon,delta) # reestimate the tail probabilities
    V = simulator.V_max # nothing done on the maximum allocation for the moment
    allocation = greedyAllocation(T,V) # do the allocation accordingly
    reward = get_reward(allocation,simulator,D,N)
    return reward

def KM_optimistic(simulator,T,mc):

    n_venues = len(simulator.venues)

    rewards = np.zeros(T)

    for m in range(mc):

        D = np.zeros([n_venues,simulator.V_max+1])
        N = np.zeros([n_venues,simulator.V_max+1])

        # first iteration with a uniform allocation
        # Do the euclidean division of V by n_venues
        p = simulator.V_max / n_venues
        r = simulator.V_max - p * n_venues

        # Allocate uniformly to the different venues
        arr = np.arange(n_venues)
        np.random.shuffle(arr)
        alloc = p * np.ones(n_venues) + (arr < r)

        rewards[0] = get_reward(alloc,simulator,D,N)

        for t in range(1,T):
            rewards[t] += KM_iter_optimistic(D,N,simulator,epsilon,delta)

    return rewards/mc