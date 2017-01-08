import numpy as np
import numpy.matlib
import scipy.optimize.nnls as nnls
import itertools


def draw_finite(p):
    """
    Draw a sample from a distribution on a finite set
    that takes value k with probability p(k)
    """
    q = np.cumsum(p)
    u = np.random.random()
    i = 0
    while u > q[i]:
        i += 1
    return i


def build_matrix(d, K, m):
    d_modified = np.copy(d)
    subsets = list(itertools.combinations(range(K), m))
    #
    N = len(subsets) - 1
    M = np.zeros((K, N))
    for j in range(N):
        subset_j = subsets[j]
        M[subset_j,j] = 1
    # Handle the last subset differently
    last_subset = list(subsets[N])
    M[last_subset,:] -= 1
    d_modified[last_subset] -= 1
    return M, d_modified


def sample_subset(d):
    m = np.round(np.sum(d))
    m = int(m)
    K = len(d)
    # print m
    # print d
    # print K
    #
    # Handle side problems
    if m == 0:
        return []
    if K == m:
        return range(K)
    if m == 1 and K == 2:
        b = np.random.random() < d[0]
        if b:
            return [0]
        else:
            return [1]
    #
    # Handle all the normal cases
    M, d_modified = build_matrix(d=d, K=K, m=m)
    distrib = nnls(M,d_modified)[0]
    distrib = list(distrib)
    distrib.append(1 - sum(distrib))
    #
    subsets = list(itertools.combinations(range(K), m))
    subset_idx = draw_finite(p=distrib)
    #
    return list(subsets[subset_idx])


def update_x(x, V, s, alloc, eta, f, v, d):
    n_venues, V_max = x.shape

    # First compute v_0
    v_0 = (np.arange(V_max) + 1) * (np.cumsum(x, axis=1) < f.reshape((len(f),1)))
    v_0 = np.max(v_0, axis=1)
    v_0 = np.min([v_0, V * np.ones(n_venues)], axis=0)
    v_0_rep = np.matlib.repmat(v_0.reshape((n_venues,1)),1,V)

    # Compute g_1 and g_2
    non_zero_d = d != 0
    inverse_d = np.zeros(d.shape)
    inverse_d[non_zero_d] = 1. / d[non_zero_d]
    g_1 = (s >= f) - inverse_d * (s==f) * (alloc==np.ceil(v))
    g_2 = inverse_d * (s>=v) * (alloc==np.ceil(v))
    g_1 = g_1.reshape((n_venues, 1))
    g_2 = g_2.reshape((n_venues, 1))

    # Compute g
    inferior_matrix = (np.arange(V)+1) <= v_0_rep
    g = inferior_matrix * g_1 + (1-inferior_matrix) * g_2
    # print g

    x[:,:V] *= np.exp(eta*g)
    x = x / np.sum(x, axis=0)
    return x



def exp3_allocation(simulator, T, n_runs, eta, gamma):
    """
    eta is the larning rate
    gamma is the threshold
    """
    V_max = simulator.V_max
    n_venues = simulator.n_venues
    venues = simulator.venues

    rewards = np.zeros(T)
    for k in range(n_runs):
        # Initialize
        x = 1. / n_venues * np.ones((n_venues, V_max))

        for t in range(T):
            V = V_max
            # Compute the parameters
            v = np.sum(x[:,:V], axis=1)
            # print v
            # print x
            # v = np.sum(x, axis=1)
            f = np.floor(v)
            d = v - f
            m = np.sum(d)
            # print m
            m = int(np.round(np.sum(d)))
            d = (1-gamma)*d + gamma*m/float(n_venues)
            # Sample over the distribution of theorem 2
            sample = sample_subset(d)
            # print "v:", v
            # print "d:", d
            # print "m:", m
            # print "sample:", sample
            # Compute the allocation
            included = np.array([int(i in sample) for i in range(n_venues)])
            alloc = f + included

            # Play alloc
            s = np.zeros(n_venues)
            for i in range(n_venues):
                # Simulate the venue and observe the reward
                s[i] = simulator.venues[i].draw()
                rewards[t] += min(s[i], alloc[i])
            # print "alloc:", alloc
            # print "s:", s
            # print ""
            # Update x according to equation 4
            x = update_x(x, V, s, alloc, eta, f, v, d)

    # Calculate the mean reward on the runs and sum the rewards of the venues
    rewards = rewards / float(n_runs)
    return rewards

