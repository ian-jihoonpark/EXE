# !/bin/bash

CKPT_DIR="/media/storage/checkpoints/OFA/base"
CUDA_VISIBLE_DEVICES=0 python Trainer.py \
--seed 42 \
--mode train \
--model_path "OFA-base" \
--project_name "ICASSP2023_4" \
--cached_dir /media/storage/checkpoints/OFA/base/cached \
--experiment_name "EXE_q_attn_e10_b32_s0" \
--dataset_name vqax \
--max_epochs 10 \
--ngpu 1 \
--warmup_ratio 0.6 \
--checkpoints_dir ${CKPT_DIR} \
--weight_decay 0.0 \
--nle_anno_path "/media/storage/datasets/NLE_annotation/VQA-X/annotated/" \
--nle_image_dir "/media/storage/datasets/image" \
--train_batch_size 16 \
--eval_batch_size 16 \
--learning_rate 2e-5 \
--gradient_accumulation_steps 2 \
--val_check_interval 0.1 \
--max_seq_len 30 \
--n_train_workers 8 \
--n_valid_workers 4 \
--img_size 224 \
--dataset_name vqax \
--AEmode "EA" \
--alignment \
--concentration_attn \
--q_attn \
# --lr_monitor \
# --lr_scheduler \
# --dropout_rate 0.1 \
# --sample_patch_num 32 \