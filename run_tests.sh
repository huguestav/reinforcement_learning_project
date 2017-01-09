#!/bin/bash

CONFIG_FILES='config/*.txt'
OUTPUT_FOLDER='output'

MAIN="main.py"

METHODS="2 3 4"
#METHODS="4"

T="20000"
n_mc='1'

for config_file in $CONFIG_FILES; do
    python $MAIN -i $config_file -o $OUTPUT_FOLDER -m $METHODS -T $T -mc $n_mc   
done
