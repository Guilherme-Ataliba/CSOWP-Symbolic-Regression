#!/bin/bash

list=("BFGS" "BFGS_random" "CG" "CG_random" "LS" "random_LS"
        "Nelder-Mead" "Nelder-Mead_random" "NoOpt" "PSO_NEW"
        "differential_evolution" "dual_annealing")
    
for opt in "${list[@]}"; do
    cp articleTests-template.py articleTests/articleTests_$opt.py
    sed -i "s/XYZ/$opt/" articleTests/articleTests_$opt.py
done

