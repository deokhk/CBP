set -e

torchrun --nproc_per_node=4 --nnodes 1 --rdzv_backend c10d --master_port 0 text2sql.py \
    --effective_batch_size 32 \
    --gradient_accumulation_steps 8 \
    --learning_rate 3e-5 \
    --epochs 50 \
    --seed 32 \
    --save_path "./models/mt5-mschema2qa-seed32" \
    --model_name_or_path "google/mt5-large" \
    --mode train \
    --train_filepath "./data/mschema2qa/train.json" \
    --dev_filepath "./data/mschema2qa/test.json" \
    --wandb_log \
    --dataset_type mschema2qa