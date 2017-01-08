##########################################
##########################################

import time
from lib import KM
from lib import allocation as alloc
from lib import simulator as sim
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import math


alloc_methods_names = ["uniform","bandit","KM","KM_optimistic"]

def parseArguments():
    parser = argparse.ArgumentParser(description="Simulate a dark pool problem and run different ML algorithms on them")

    parser.add_argument("-i",
        type=str, help="input config file")

    parser.add_argument("-o",
        type=str, help="output FOLDER path for plots")

    parser.add_argument("-m",
        type=int, nargs="*", help="(optional) Method indexes, default to 0 (uniform allocation)")

    parser.add_argument("-T",
        type=int, help='(optional) time horizon, defaults to 1000')

    parser.add_argument("-mc",
        type=int, help='(optional) number of MC runs, for better regret estimation, defaults to 10')

    args = parser.parse_args()
    return args



##################################################
###################### PLOTS #####################
##################################################

def plot_regret(regret,method_label="method 1"):
    plt.plot(regret,label=method_label)
    plt.xlabel('time')
    plt.ylabel('regret')


##################################################
################# PARSE CONFIGURATIONS ###########
##################################################

def parse_config_file(input_config):
# this method returns the dark pool problem configuration parameters
    model_name_array = []
    n_venues_array = []
    V_array = []
    params_array = []

    f = open(input_config,"r") # open in read mode

    line = f.readline()

    while line:
        ## Read model_name
        model_name = line.rstrip() #to remove the '\n'
        ## Read number of venues
        n_venues = int(f.readline().rstrip())
        ## Read Vmax
        V = int(f.readline().rstrip())
        ## init parameters array
        last_pos = f.tell()
        n_params_per_venue = len(f.readline().rstrip().split(' '))
        params = np.zeros([n_venues,n_params_per_venue])
        f.seek(last_pos)
        ## Read the parameters
        for k in range(n_venues):
            #current_param = np.array(f.readline().rstrip().split(' ')) # need to map to float ?
            current_param = np.array(map(float,f.readline().rstrip().split(' ')))
            params[k,:] = current_param
        line = f.readline()

        # append to returned arrays
        model_name_array.append(model_name)
        n_venues_array.append(n_venues)
        V_array.append(V)
        params_array.append(params)

    f.close()

    return model_name_array,n_venues_array,V_array,params_array

##################################################
###################### MAIN ######################
##################################################

def main():
    args = parseArguments()
    input_config = args.i # for one file
    output_path = args.o
    T = args.T
    n_mc = args.mc
    methods =args.m

    # set default values
    if T is None:
        T = 1000
    if n_mc is None:
        n_mc = 10
    if methods is None:
        methods = 0


    # create output path folders if not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model_name_array,n_venues_array,V_array,params_array = parse_config_file(input_config)

    for k in range(len(model_name_array)):
        print "Computing allocation schemes on model %d" % k
        model_name = model_name_array[k]
        n_venues = n_venues_array[k]
        V = V_array[k]
        params = params_array[k]
        # configure the simulator
        simulator = sim.simulator(model_name,n_venues,V,params)

        # run the simuation for each method
        plt.figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
        plt.title("model: %s, n_venues: %d, V_max: %d" % (model_name,n_venues,V))

        # run the optimal allocation
        print "\rOptimal allocation...    "
        optimal_reward = alloc.optimal_allocation(simulator,T,n_mc)

        for m in methods:
            regret = []

            tic = time.clock()
            if m==0:
                print "\runiform_allocation...    ",
                sys.stdout.flush()
                regret = optimal_reward - alloc.uniform_allocation(simulator,T,n_mc)
            if m==1:
                print "\rbandit_allocation...    ",
                sys.stdout.flush()
                regret = optimal_reward - alloc.bandit_allocation(simulator,T,n_mc,0.05)
            if m==2:
                print "\rKM_allocation...    ",
                sys.stdout.flush()
                regret = optimal_reward - KM.KM(simulator,T,n_mc)
            if m==3:
                print "\rKM_optimistic...    ",
                sys.stdout.flush()
                regret = optimal_reward - KM.KM_optimistic(simulator,T,n_mc)
            tac = time.clock()
            timing = tac - tic

            # plot the results
            plot_regret(np.cumsum(regret),alloc_methods_names[m])

        # output path
        print "\rall allocations computed!"
        output_path_fig = os.path.join(\
            output_path,\
            '%s_%dvenues_%dV(%d).eps'\
            %(model_name,n_venues,V,k))

        plt.grid()
        plt.legend(loc=2)
        plt.savefig(output_path_fig, bbox_inches='tight')
        print "EPS figure saved at: " + output_path_fig

if __name__ == '__main__':
    main()
