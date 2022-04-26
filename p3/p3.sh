#!/bin/bash
  
salloc --nodes=1 --ntasks=1 --time=03:00:00

module load anaconda3/2021.11

python3 p3_1.py

python3 p3_2.py

