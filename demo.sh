#! usr/bin/env bash
# Generated visual results.
device=0
args=(--rate 0.05 --figs 2 --maskType '2D' --dataset_name 'OASI1_MRB')
CUDA_VISIBLE_DEVICES=${device} python ./test_figures2.py "${args[@]}"
