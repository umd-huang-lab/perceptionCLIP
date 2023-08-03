#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$PWD"


augmentation=$1
template=$2
save_name=$3
data_path=$4

python src/zero_shot_inference/eval_similarity.py --eval_augmentation="$augmentation" \
  --template="$template" \
  --save_name="$save_name" \
  --model=ViT-B/16 \
  --data_location="$data_path" \
  --dataset=ImageNetC \
  --random_descrip=True \

