#!/bin/bash 

EPOCHS=50
BATCH_SIZE=16
GPU_NUM=4
GRAD_ACCUM_STEP=4
LR=3e-5
WARMUP=500

BATCH_SIZE_PER_GPU=$(($BATCH_SIZE/($GPU_NUM*$GRAD_ACCUM_STEP)))
VALID_BATCH_SIZE_PER_GPU=$(($BATCH_SIZE_PER_GPU*2))

OUTPUT=output
TAG=mschema2qa-mr2text-enko-${EPOCHS}.${BATCH_SIZE}.${LR}.${WARMUP}
ADAPTER_DIR=./output/OneM-enen-en-100000-mean_eng-4.1e-4.1000 # Your path to english adapter

torchrun --nproc_per_node=4 --nnodes 1 --rdzv_backend c10d --master_port 0 run.py --do_task_finetune \
              --epochs $EPOCHS \
              --batch_size $BATCH_SIZE_PER_GPU \
              --valid_batch_size $VALID_BATCH_SIZE_PER_GPU  \
              --gradient_accumulation_steps $GRAD_ACCUM_STEP \
              --learning_rate $LR \
              --warmup_steps $WARMUP \
              --output $OUTPUT \
              --exp_tag $TAG \
              --langs en,en \
              --task_lang en \
              --pretrained_adapter_dir ${ADAPTER_DIR} \
              --train_file "./data/mschema2qa/train.json" \
              --valid_file "./data/mschema2qa/test.json" \
              --dataset_type mschema2qa \
              --freeze_option decoder_only
