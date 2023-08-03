#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$PWD"

factors=$1
model=$2
infer_mode=$3
save_name=$4
temperature=(0.5 0.8 1 3 5 10)

for i in "${temperature[@]}"; do

  python src/zero_shot_inference/perceptionclip_two_step.py --dataset=Waterbirds \
    --save_name="$save_name" \
    --infer_mode="$infer_mode" \
    --model="$model" \
    --temperature="$i" \
    --convert_text=bird \
    --eval_group=True \
    --main_template=waterbirds_main_template \
    --factor_templates=waterbirds_factor_templates \
    --factors="$factors" \
    --eval_trainset=True \


done
