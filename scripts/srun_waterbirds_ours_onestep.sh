#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$PWD"


template=$1
model=$2
save_name=$3

python src/zero_shot_inference/perceptionclip_one_step.py --dataset=Waterbirds \
  --template="$template" \
  --save_name="$save_name" \
  --model="$model" \
  --eval_group=True \
  --eval_trainset=True \

