#!/bin/bash

if [ -z "$1" ]; then
echo "Please provide a *.pt file as input"
exit 1
fi

model_file=$1
output_dir=./outputs/test_pointbert_8kpts

CUDA_VISIBLE_DEVICES=0 python main.py --model ULIP2_PointBERT_Colored --npoints 10000 --output-dir $output_dir --evaluate_3d_ulip2 --validate_dataset_name=modelnet40 --test_ckpt_addr $model_file 2>&1 | tee $output_dir/log.txt