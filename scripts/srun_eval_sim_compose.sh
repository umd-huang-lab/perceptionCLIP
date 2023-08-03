#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$PWD"


augmentation=$1
augmentation_2=$2
template=$3
save_name=$4

python src/zero_shot_inference/eval_similarity.py --eval_augmentation="$augmentation" \
  --template="$template" \
  --save_name="$save_name" \
  --model=ViT-B/16 \
  --eval_augmentation_2="$augmentation_2" \

