import numpy as np
from KM import greedyAllocation
import sys


def uniform_allocation(simulator, T, n_runs):
    n_venues = simulator.n_venues
    rewards = np.zeros((T, n_venues))

    for k in range(n_runs):
        print "\r(%d/%d) MC runs" % (k+1,n_runs),
        sys.stdout.flush()
        for t in range(T):
            # Do the euclidean division of V by n_venues
            p = simulator.V_max / n_venues
            r = simulator.V_max - p * n_venues

            # Allocate uniformly to the different venues
            arr = np.arange(n_venues)
            np.random.shuffle(arr)
            alloc = p * np.ones(n_venues) + (arr < r)
            # print "uniform:", alloc

            for i in range(n_venues):
                # Simulate the venue and observe the reward
                sample = simulator.venues[i].draw()
                rewards[t,i] += min(sample, alloc[i])

    print ""
    # Calculate the mean reward on the runs and sum the rewards of the venues
    rewards = rewards / float(n_runs)
    rewards = np.sum(rewards, axis=1)
    return rewards

def bandit_allocation_swap(simulator, T, n_runs=1, n_swaps=5, alpha=0.05):
    TT = np.floor(np.linspace(0,T,n_swaps)).astype(int)
    n_venues = simulator.n_venues
    rewards = np.zeros((T, n_venues))
    allocations = np.zeros((n_venues,T))

    for k in range(n_runs):

        weights = 1. / n_venues * np.ones(n_venues)
        for interval in range(len(TT)-1):
            for t in range(TT[interval],TT[interval+1]):
                #print "\r(%d/%d) MC runs | (%d/%d) time step" % (k+1,n_runs,t,T),
                #sys.stdout.flush()
                # Allocate according to the weights
                alloc = np.floor(simulator.V_max * weights)

                # Allocate the rest randomly
                r = simulator.V_max - np.sum(alloc)
                arr = np.arange(n_venues)
                np.random.shuffle(arr)
                alloc = alloc + (arr < r)
                allocations[:,t] += alloc
                # print "bandit:", alloc

                current_rewards = np.zeros(n_venues)
                for i in range(n_venues):
                    # Simulate the venue and observe the reward
                    sample = simulator.venues[i].draw()
                    current_rewards[i] = min(sample, alloc[i])
                # print "current_rewards:", current_rewards
                # print ""

                rewards[t] += current_rewards

                # Update the weights
                weights += alpha * (current_rewards > 0)
                weights = weights / np.sum(weights)
            simulator.random_swap_venues()

    #print ""
    # Calculate the mean reward on the runs and sum the rewards of the venues
    rewards = rewards / float(n_runs)
    rewards = np.sum(rewards, axis=1)
    allocations = allocations / float(n_runs)
    return rewards, allocations


def bandit_allocation(simulator, T, n_runs, alpha=0.05):
    n_venues = simulator.n_venues
    rewards = np.zeros((T, n_venues))

    for k in range(n_runs):
        print "\r(%d/%d) MC runs" % (k+1,n_runs),
        sys.stdout.flush()
        weights = 1. / n_venues * np.ones(n_venues)
        for t in range(T):
            # Allocate according to the weights
            alloc = np.floor(simulator.V_max * weights)

            # Allocate the rest randomly
            r = simulator.V_max - np.sum(alloc)
            arr = np.arange(n_venues)
            np.random.shuffle(arr)
            alloc = alloc + (arr < r)

            # print "bandit:", alloc

            current_rewards = np.zeros(n_venues)
            for i in range(n_venues):
                # Simulate the venue and observe the reward
                sample = simulator.venues[i].draw()
                current_rewards[i] = min(sample, alloc[i])
            # print "current_rewards:", current_rewards
            # print ""

            rewards[t] += current_rewards

            # Update the weights
            weights += alpha * (current_rewards > 0)
            weights = weights / np.sum(weights)

    print ""
    # Calculate the mean reward on the runs and sum the rewards of the venues
    rewards = rewards / float(n_runs)
    rewards = np.sum(rewards, axis=1)
    return rewards


def optimal_allocation(simulator, T, n_runs):
    V_max = simulator.V_max
    n_venues = simulator.n_venues
    tail_probas = np.empty((n_venues, V_max+1))
    for k in range(n_venues):
        tail_probas[k,:] = simulator.venues[k].tail_distrib()

    rewards = np.zeros((T, n_venues))
    for k in range(n_runs):
        print "\r(%d/%d) MC runs" % (k+1,n_runs),
        sys.stdout.flush()

        for t in range(T):

            # Allocate according to the tail proba
            alloc = greedyAllocation(tail_probas, V_max)
            # print "alloc greedy:", alloc

            current_rewards = np.zeros(n_venues)
            for i in range(n_venues):
                # Simulate the venue and observe the reward
                sample = simulator.venues[i].draw()
                current_rewards[i] = min(sample, alloc[i])

            # print "current_rewards:", current_rewards
            # print ""
            rewards[t] += current_rewards

    print ""
    # Calculate the mean reward on the runs and sum the rewards of the venues
    rewards = rewards / float(n_runs)
    rewards = np.sum(rewards, axis=1)
    return rewards

def optimal_allocation_swap(simulator, T, n_runs=1, n_swaps=5):
    TT = np.floor(np.linspace(0,T,n_swaps)).astype(int)

    V_max = simulator.V_max
    n_venues = simulator.n_venues
    tail_probas = np.empty((n_venues, V_max+1))
    for k in range(n_venues):
        tail_probas[k,:] = simulator.venues[k].tail_distrib()

    rewards = np.zeros((T, n_venues))
    allocations = np.zeros((n_venues,T))

    for k in range(n_runs):

        for interval in range(len(TT)-1):
            for t in range(TT[interval],TT[interval+1]):

                #print "\r(%d/%d) MC runs | (%d/%d) time step" % (k+1,n_runs,t,T),
                #sys.stdout.flush()

                # Allocate according to the tail proba
                alloc = greedyAllocation(tail_probas, V_max)
                # print "alloc greedy:", alloc
                allocations[:,t] += alloc
                current_rewards = np.zeros(n_venues)
                for i in range(n_venues):
                    # Simulate the venue and observe the reward
                    sample = simulator.venues[i].draw()
                    current_rewards[i] = min(sample, alloc[i])

                # print "current_rewards:", current_rewards
                # print ""
                rewards[t] += current_rewards
        simulator.random_swap_venues()
    #print ""
    # Calculate the mean reward on the runs and sum the rewards of the venues
    rewards = rewards / float(n_runs)
    rewards = np.sum(rewards, axis=1)
    allocations =allocations / float(n_runs)
    return rewards, allocations

