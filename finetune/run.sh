#!/bin/bash

model_path=../further_pretraining/huggingface/v1/roberta_further_50000

python train.py \
    --data_dir=data_bin \
    --output_dir=outputs/roberta-large \
    --log_dir=runs/roberta-large \
    --model_type=roberta \
    --model_name_or_path=${model_path} \
    --per_gpu_train_batch_size=8 \
    --per_gpu_eval_batch_size=8 \
    --gradient_accumulation_steps 32 \
    --learning_rate=1e-5 \
    --adam_epsilon=1e-6 \
    --weight_decay=0.01 \
    --num_train_epochs=5 \
    --warmup_proportion=0.1 \
    --fp16 || exit