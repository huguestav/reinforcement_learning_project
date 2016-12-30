##########################################
##########################################

import time
from lib import ML
from lib import simulator as sim
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import math


def parseArguments():
    parser = argparse.ArgumentParser(description="Simulate a dark pool problem and run different ML algorithms on them")
    
    parser.add_argument("-i",
        type=int, help="input config file")

    parser.add_argument("-o",
        type=float, help="output path for plots")

    parser.add_argument("-T",
        type=int, help='(optional) time horizon, defaults to 1000')

    parser.add_arguement("-mc",
        type=int, help='(optional) number of MC runs, for better regret estimation, defaults to 10')

    args = parser.parse_args()
    return args



##################################################
###################### PLOTS #####################
##################################################

def plot_regret(regret,method_label="method 1",title="Regret curve on a dark pool allocation problem"):
    figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(regret,label=method_label)
    plt.grid()
    plt.title(title)
    plt.xlabel('time')
    plt.ylabel('regret')


##################################################
################# PARSE CONFIGURATIONS ###########
##################################################

def parse_config_file(input_config):
# this method returns the dark pool problem configuration parameters
    f = open(input_config,"r") # open in read mode
    
    ## Read method 
    method = f.readline()[:-1] #to remove the '\n'
    ## Read number of venues
    n_venues = int(f.readline()[:-1])
    last_pos = f.tell()
    ## init parameters array
    n_params_per_venue = len(f.readline()[:-1].split(' '))
    params = np.zeros([n_venues,n_params_per_venue)
    f.seek(last_pos)
    ## Read the parameters    
    for k in range(n_venues):
        current_param = np.array(f.readline()[:-1].split(' '))        
        params[k,:] = current_param

    f.close()
    
    return methods,n_venues,params

##################################################
###################### MAIN ######################
##################################################

def main():
    args = parseArguments()
    input_config = args.i
    output_path = args.o
    T = args.T
    n_mc = args.mc

    # set default values
    if T is None:
        T = 1000
    if n_mc is None:
        n_mc = 10

    # create output path folders if not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # output path
    output_path_fig = os.path.join(output_path,'%s.eps'\
    %input_config.rsplit(".",1)[0])
    
    # configure the simulator
    method,n_venues,params = parse_config_file(input_config)
    simulator = sim.simulator(method,n_venues,params)

    # run the simuation
    tic = time.clock()
    regret = ML.run_simu(simulator, T, n_mc)
    tac = time.clock()

    timing = tac - tic
        
    # plot the results
    plot_regret(regret,method)
    plt.savefig(output_path_fig, bbox_inches='tight')   
    print "EPS figure saved at: " + output_path_fig

if __name__ == '__main__':
    main()
