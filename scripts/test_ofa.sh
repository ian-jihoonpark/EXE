#!/bin/bash
CKPT_DIR="./ckpts"

CUDA_VISIBLE_DEVICES=2 python Trainer.py \
--seed 32 \
--mode test \
--model_path "OFA-base" \
--output_dir outputs \
--cached_dir /media/storage/checkpoints/OFA/base/cached \
--load_ckpt_path /media/storage/checkpoints/OFA/base/EXE_abnorm_e30_b32/epoch=01-OFA-base_val_loss=0.850.ckpt \
--test_batch_size 1 \
--ngpu 1 \
--img_size 224 \
--nle_anno_path "/media/storage/datasets/NLE_annotation/VQA-X/annotated" \
--nle_image_dir "/media/storage/datasets/image" \
--vqax_test_anno_path "/media/storage/datasets/NLE_annotation/VQA-X/annotated/vqaX_test.json" \
--top_k 0 \
--top_p 0.9 \
--dataset_name vqax \
--AEmode "EA" \
--alignment \
--concentration_attn