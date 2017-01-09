#!/bin/bash

CONFIG_FILES='config/powerlaw1.txt'
OUTPUT_FOLDER='output'

MAIN="main.py"

METHODS="1 2 3 4"
#METHODS="4"

T="2000"
n_mc='30'

for config_file in $CONFIG_FILES; do
    python $MAIN -i $config_file -o $OUTPUT_FOLDER -m $METHODS -T $T -mc $n_mc   
done
