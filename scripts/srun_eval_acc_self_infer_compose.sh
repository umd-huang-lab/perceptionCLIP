#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$PWD"

augmentation=$1
augmentation_2=$2
template0=$3
template1=$4
infer_mode=$5
save_name=$6
temperature=(1)

for i in "${temperature[@]}"; do

  python src/zero_shot_inference/eval_acc_self_infer.py --eval_augmentation="$augmentation" \
    --template0="$template0" \
    --template1="$template1" \
    --save_name="$save_name" \
    --infer_mode="$infer_mode" \
    --model=ViT-B/16 \
    --temperature="$i" \
    --eval_augmentation_2="$augmentation_2" \


done
