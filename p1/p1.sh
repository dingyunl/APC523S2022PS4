#!/bin/bash
  
salloc --nodes=1 --ntasks=1 --time=03:00:00

module load anaconda3/2021.11

python3 p1_a.py

python3 p1_b_128.py

python3 p1_b_256.py

python3 p1_b_512.py

