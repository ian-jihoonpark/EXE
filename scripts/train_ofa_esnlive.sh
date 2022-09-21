#!/bin/bash
# --captioning_anno_paths \
# "../captioning_data/nocaps/annotations/filtered_80.46_k-100_p-0.5_t-0.7_n_1.json" \
# --captioning_image_dirs \
# "../captioning_data/nocaps/image" \
CKPT_DIR="/media/storage/checkpoints/OFA/base"

CUDA_VISIBLE_DEVICES=2 python Trainer.py \
--seed 42 \
--mode train \
--model_path "OFA-base" \
--project_name "ICASSP2023" \
--cached_dir /media/storage/checkpoints/OFA/base/cached \
--experiment_name "EXE_esnlive_experiment_s2" \
--dataset_name esnlive \
--max_epochs 1 \
--ngpu 1 \
--warmup_ratio 0.6 \
--checkpoints_dir ${CKPT_DIR} \
--weight_decay 0.0 \
--nle_anno_path "/media/storage/datasets/NLE_annotation/e_SNLI_VE" \
--nle_image_dir "/media/storage/datasets/image/flickr30k" \
--train_batch_size 32 \
--eval_batch_size 32 \
--learning_rate 2e-5 \
--gradient_accumulation_steps 1 \
--val_check_interval 0.1 \
--max_seq_len 40 \
--n_train_workers 4 \
--n_valid_workers 4 \
--img_size 224 \
--AEmode "EA" \