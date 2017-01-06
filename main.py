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
        type=int, help="input config file")

    parser.add_argument("-o",
        type=float, help="output FOLDER path for plots")

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
    plt.grid()
    plt.xlabel('time')
    plt.ylabel('regret')


##################################################
################# PARSE CONFIGURATIONS ###########
##################################################

def parse_config_file(input_config):
# this method returns the dark pool problem configuration parameters
    f = open(input_config,"r") # open in read mode

    ## Read model_name
    model_name = f.readline().rstrip() #to remove the '\n'
    ## Read number of venues
    n_venues = int(f.readline().rstrip())
    last_pos = f.tell()
    ## init parameters array
    n_params_per_venue = len(f.readline().rstrip().split(' '))
    params = np.zeros([n_venues,n_params_per_venue])
    f.seek(last_pos)
    ## Read the parameters
    for k in range(n_venues):
        #current_param = np.array(f.readline().rstrip().split(' ')) # need to map to float ?
        current_param = np.array(map(float,f.readline().rstrip().split(' ')))
        params[k,:] = current_param

    f.close()

    return model_name,n_venues,params

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

    # output path
    output_path_fig = os.path.join(output_path,'%s.eps'\
    %input_config.rstrip().rsplit(".",1)[0])

    # configure the simulator
    model_name,n_venues,params = parse_config_file(input_config)
    simulator = sim.simulator(model_name,n_venues,params)

    # run the simuation for each method
    figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    for m in methods:
        regret = []

        tic = time.clock()
        if m==0:
            regret = alloc.uniform_allocation(simulator,T,n_mc)
            break
        if m==1:
            regret = alloc.bandit_allocation(simulator,T,n_mc,0.05)
            break
        if m==2:
            regret = KM.KM(simulator,T,n_mc)
            break
        if m==3:
            regret = KM.KM_optimistic(simulator,T,n_mc)
        tac = time.clock()
        timing = tac - tic

        # plot the results
        plot_regret(regret,alloc_methods_names[m])

    plt.savefig(output_path_fig, bbox_inches='tight')
    print "EPS figure saved at: " + output_path_fig

if __name__ == '__main__':
    main()
