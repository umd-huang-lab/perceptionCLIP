#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$PWD"

dataset=$1
model=$2
main_template=$3
factor_templates=$4
factors=$5
infer_mode=$6
convert_text=$7
save_name=$8

temperature=(0.8 1 2 3 4 5 10 100 1000)

for i in "${temperature[@]}"; do

  python src/zero_shot_inference/perceptionclip_two_step.py --dataset="$dataset" \
    --save_name="$save_name" \
    --infer_mode="$infer_mode" \
    --model="$model" \
    --temperature="$i" \
    --convert_text="$convert_text" \
    --main_template="$main_template" \
    --factor_templates="$factor_templates" \
    --factors="$factors" \


done

