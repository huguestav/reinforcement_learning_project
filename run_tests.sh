#!/bin/bash

CONFIG_FILES='config/*.txt'
OUTPUT_FOLDER='output'

MAIN="main.py"

METHODS="0 1 2 3"

T="10000"
n_mc='10'

for config_file in $CONFIG_FILES; do
    python $MAIN -i $config_file -o $OUTPUT_FOLDER -m $METHODS -T $T -mc $n_mc   
done
