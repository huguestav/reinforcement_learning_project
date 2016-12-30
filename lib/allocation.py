import numpy as np


def uniform_allocation(simulator, T, n_runs):
    n_venues = simulator.n_venues
    rewards = np.zeros((T, n_venues))

    for k in range(n_runs):
        for t in range(T):
            # Do the euclidean division of V by n_venues
            p = simulator.V_max / n_venues
            r = simulator.V_max - p * n_venues

            # Allocate uniformly to the different venues
            arr = np.arange(n_venues)
            np.random.shuffle(arr)
            alloc = p * np.ones(n_venues) + (arr < r)

            for i in range(n_venues):
                # Simulate the venue and observe the reward
                sample = simulator.venues[i].draw()
                rewards[t,i] += min(sample, alloc[i])

    # Calculate the mean reward on the runs and sum the rewards of the venues
    rewards = rewards / float(n_runs)
    rewards = np.sum(rewards, axis=1)
    return rewards


