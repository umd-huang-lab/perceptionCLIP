#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$PWD"


augmentation=$1
template=$2
dataset=$3
save_name=$4

python src/zero_shot_inference/zero_shot_org.py --eval_augmentation="$augmentation" \
--template="$template" \
--save_name="$save_name" \
--model=ViT-B/16 \
--dataset="$dataset" \


