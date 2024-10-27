#!/bin/bash
set -e

NUM_GPUS=4
SEED=32

torchrun --nproc_per_node=$NUM_GPUS --nnodes 1 --rdzv_backend c10d --master_port 0 text2sql.py \
    --effective_batch_size 32 \
    --gradient_accumulation_steps 8 \
    --learning_rate 3e-5 \
    --epochs 50 \
    --seed ${SEED} \
    --save_path "./models/mt5-large-xspider-CPT-seed32" \
    --model_name_or_path "google/mt5-large" \
    --mode train \
    --multilingual_pt \
    --multi_pt_dataset_path_list "/home1/deokhk_1/research/XLang-NL2SQL/output/mschema2qa-mr2text-100K-50.16.3e-5.500/last_ckpt/filtered_generated_predictions_ar_beam_4_from_train.json,/home1/deokhk_1/research/XLang-NL2SQL/output/mschema2qa-mr2text-100K-50.16.3e-5.500/last_ckpt/filtered_generated_predictions_de_beam_4_from_train.json,/home1/deokhk_1/research/XLang-NL2SQL/output/mschema2qa-mr2text-100K-50.16.3e-5.500/last_ckpt/filtered_generated_predictions_es_beam_4_from_train.json,/home1/deokhk_1/research/XLang-NL2SQL/output/mschema2qa-mr2text-100K-50.16.3e-5.500/last_ckpt/filtered_generated_predictions_fa_beam_4_from_train.json,/home1/deokhk_1/research/XLang-NL2SQL/output/mschema2qa-mr2text-100K-50.16.3e-5.500/last_ckpt/filtered_generated_predictions_fi_beam_4_from_train.json,/home1/deokhk_1/research/XLang-NL2SQL/output/mschema2qa-mr2text-100K-50.16.3e-5.500/last_ckpt/filtered_generated_predictions_it_beam_4_from_train.json,/home1/deokhk_1/research/XLang-NL2SQL/output/mschema2qa-mr2text-100K-50.16.3e-5.500/last_ckpt/filtered_generated_predictions_ja_beam_4_from_train.json,/home1/deokhk_1/research/XLang-NL2SQL/output/mschema2qa-mr2text-100K-50.16.3e-5.500/last_ckpt/filtered_generated_predictions_pl_beam_4_from_train.json,/home1/deokhk_1/research/XLang-NL2SQL/output/mschema2qa-mr2text-100K-50.16.3e-5.500/last_ckpt/filtered_generated_predictions_tr_beam_4_from_train.json,/home1/deokhk_1/research/XLang-NL2SQL/output/mschema2qa-mr2text-100K-50.16.3e-5.500/last_ckpt/filtered_generated_predictions_zh_beam_4_from_train.json" \
    --train_filepath "./data/xspider/train_spider_seq2seq.json" \
    --dev_filepath "./data/xspider/dev_spider_seq2seq.json" \
    --wandb_log \
    --dataset_type spider \
    --dataset_lang en

# You can put multi_pt_dataset_path_list as a single string, where each path is separated by a comma.
# This is where you put your synthesized data from the previous step! Make sure to put synthesized utterances from Xspider dataset.