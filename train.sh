#!/bin/bash

timestamp=$(date +"%m-%d_%H-%M")

log_file="log_$timestamp.txt"

python3 train.py > $log_file

