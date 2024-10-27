set -e

torchrun --nproc_per_node=4 --nnodes 1 --rdzv_backend c10d --master_port 0 text2sql.py \
    --effective_batch_size 32 \
    --gradient_accumulation_steps 8 \
    --learning_rate 3e-5 \
    --epochs 50 \
    --seed 42 \
    --save_path "./models/text2sql-mt5-large_baseline" \
    --model_name_or_path "google/mt5-large" \
    --mode train \
    --train_filepath "./data/xspider/train_spider_seq2seq.json" \
    --dev_filepath "./data/xspider/dev_spider_seq2seq.json" \
    --wandb_log \
    --dataset_type spider