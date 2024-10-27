#!/bin/bash
TRAIN_STEPS=100000
EVAL_STEPS=5000
BATCH_SIZE=32
LR=1e-4
WARMUP=1000
GPU_NUM=4

BATCH_SIZE_PER_GPU=$(($BATCH_SIZE/$GPU_NUM))
VALID_BATCH_SIZE_PER_GPU=$(($BATCH_SIZE_PER_GPU*2))
MAX_SAMPLES=1000000

REMOVAL_TYPE=mean_eng
OUTPUT=output
LANG=vi
TAG=OneM-en${LANG}-${LANG}-$MAX_SAMPLES-$REMOVAL_TYPE-$BATCH_SIZE.$LR.$WARMUP

torchrun --nproc_per_node=$GPU_NUM --nnodes 1 --rdzv_backend c10d --master_port 0 run.py --do_adapter_train \
              --training_steps $TRAIN_STEPS \
              --eval_steps $EVAL_STEPS \
              --batch_size $BATCH_SIZE_PER_GPU \
              --valid_batch_size $VALID_BATCH_SIZE_PER_GPU \
              --learning_rate $LR \
              --warmup_steps $WARMUP \
              --output $OUTPUT \
              --exp_tag $TAG \
              --adapter_types "decoder-lang" \
              --langs en,${LANG} \
              --partial_langs ${LANG} \
              --mask_rate 0.25 \
              --adapter_layer_norm true \
              --frozen_list all \
              --max_samples $MAX_SAMPLES \
              --without_language_identity \
              --removal_type $REMOVAL_TYPE \
              --model_name_or_path "google/mt5-large"\
              --language_identity_path ./language_identity
