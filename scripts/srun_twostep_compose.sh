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

temperature=(1 3 5 10)

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

