import numpy as np
import numpy.matlib
import scipy.optimize.nnls as nnls
import itertools
import sys


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


def one_exp3_iteration(simulator, gamma, eta, x):
    (n_venues, V_max) = x.shape

    V = V_max
    v = np.sum(x[:,:V], axis=1)
    f = np.floor(v)
    d = v - f
    m = np.sum(d)
    m = int(np.round(np.sum(d)))
    d = (1-gamma)*d + gamma*m/float(n_venues)

    # Sample over the distribution of theorem 2
    sample = sample_subset(d)

    # Compute the allocation
    included = np.array([int(i in sample) for i in range(n_venues)])
    alloc = f + included

    # Play alloc
    s = np.zeros(n_venues)
    reward = 0
    for i in range(n_venues):
        # Simulate the venue and observe the reward
        s[i] = simulator.venues[i].draw()
        reward += min(s[i], alloc[i])

    # Update x according to equation 4
    x = update_x(x, V, s, alloc, eta, f, v, d)

    return x, alloc, reward



def exp3_allocation(simulator, T, mc, gamma=0.01):
    """
    eta is the larning rate
    gamma is the threshold
    """
    V_max = simulator.V_max
    n_venues = simulator.n_venues
    venues = simulator.venues
    eta = (V_max * np.log(n_venues)**2 / n_venues * T**(-2))**(1./3)

    rewards = np.zeros(T)
    for k in range(mc):
        print "\r(%d/%d) MC runs" % (k+1,mc),
        sys.stdout.flush()
        # Initialize
        x = 1. / n_venues * np.ones((n_venues, V_max))

        for t in range(T):
            x, alloc, reward = one_exp3_iteration(simulator, gamma, eta, x)
            rewards[t] += reward

    print ''
    # Calculate the mean reward on the runs and sum the rewards of the venues
    rewards = rewards / float(mc)
    return rewards

def exp3_allocation_swap(simulator,T, mc=1, n_swaps=5, gamma=0.01):
    """
    eta is the larning rate
    gamma is the threshold
    """
    TT = np.floor(np.linspace(0,T,n_swaps)).astype(int)

    V_max = simulator.V_max
    n_venues = simulator.n_venues
    venues = simulator.venues
    eta = (V_max * np.log(n_venues)**2 / n_venues * T**(-2))**(1./3)

    rewards = np.zeros(T)
    allocations = np.zeros((n_venues, T))

    for k in range(mc):
        # Initialize
        x = 1. / n_venues * np.ones((n_venues, V_max))


        for interval in range(len(TT)-1):
            for t in range(TT[interval],TT[interval+1]):
                x, alloc, reward = one_exp3_iteration(simulator, gamma, eta, x)
                rewards[t] += reward
                allocations[:,t] += alloc
            simulator.random_swap_venues()

        # for t in range(T):
        #     x, alloc, reward = one_exp3_iteration(simulator, gamma, eta, x)
        #     rewards[t] += reward
        #     allocations[:,t] += alloc

        # simulator.random_swap_venues()
        # # swap venues in the simulator
        # for t in range(T,2*T):
        #     x, alloc, reward = one_exp3_iteration(simulator, gamma, eta, x)
        #     rewards[t] += reward
        #     allocations[:,t] += alloc

    # Calculate the mean reward on the runs and sum the rewards of the venues
    allocations = allocations / float(mc)
    rewards = rewards / float(mc)

    return rewards, allocations

def exp3_allocation_change(simulator1, simulator2, T, eta, gamma):
    """
    eta is the larning rate
    gamma is the threshold
    """
    V_max = simulator1.V_max
    n_venues = simulator1.n_venues


    rewards = np.zeros(2*T)
    allocations = np.zeros((n_venues, 2*T))
    x = 1. / n_venues * np.ones((n_venues, V_max))

    for t in range(T):
        x, alloc, reward = one_exp3_iteration(simulator1, gamma, eta, x)
        rewards[t] += reward
        allocations[:,t] += alloc

    for t in range(T):
        x, alloc, reward = one_exp3_iteration(simulator2, gamma, eta, x)
        rewards[T+t] += reward
        allocations[:,T+t] += alloc

    # Calculate the mean reward on the runs and sum the rewards of the venues
    return rewards, allocations


