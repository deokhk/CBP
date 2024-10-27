#!/bin/bash

MODEL_PATH=./output/mschema2qa-mr2text-1K-50.16.3e-5.500/last_checkpoint/
INFERENCE_FILE=./data/mschema2qa/train.json

langs=("zh" "tr" "pl" "it" "fi" "fa" "ar" "ja" "es" "de")
for lang in ${langs[@]}; do
    PRETRAINED_ADAPTER_DIR=./output/OneM-en${lang}-${lang}-1000-mean_eng-32.1e-4.300/
    CUDA_VISIBLE_DEVICES=0 python run.py --do_synthesize \
              --valid_batch_size 32  \
              --model_name_or_path $MODEL_PATH \
              --langs en,${lang} \
              --task_lang $lang \
              --adapter_types "decoder-lang" \
              --pretrained_adapter_dir $PRETRAINED_ADAPTER_DIR \
              --inference_data_file $INFERENCE_FILE \
              --generation_num_beams 4 \
              --repetition_penalty 1.2 \
              --dataset_type mschema2qa \
              --dataset_lang en
done
