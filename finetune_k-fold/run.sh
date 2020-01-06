#!/bin/bash

k_id=0
model_path=../further_pretraining/huggingface/v1/roberta_further_50000

while [ $k_id -lt 10 ]; do
    python train.py \
        --data_dir=data_bin \
        --output_dir=/work/u3110095/roberta-large_k10/${k_id} \
        --log_dir=runs/roberta-large_k10/${k_id} \
        --model_type=roberta \
        --model_name_or_path=${model_path} \
        --per_gpu_train_batch_size=12 \
        --per_gpu_eval_batch_size=12 \
        --gradient_accumulation_steps 8 \
        --learning_rate=1e-5 \
        --adam_epsilon=1e-6 \
        --weight_decay=0.01 \
        --num_train_epochs=5 \
        --warmup_proportion=0.1 \
        --ensemble_id=${ensemble_id} \
        --seed=42 \
        --fp16 || exit
    let ensemble_id=ensemble_id+1
done
