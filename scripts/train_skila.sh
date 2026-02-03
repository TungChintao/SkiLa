#!/bin/bash

# export WANDB_PROJECT="SkiLa-7B-SFT"
export WANDB_DISABLED=true
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=DETAIL


# model configs
MODEL_SIZE='7B'
MODEL_NAME="Qwen/Qwen2.5-VL-${MODEL_SIZE}-Instruct"

SKETCH_ENCODER="google/siglip2-so400m-patch14-384"

PATTERN="vl"
GRAD_CHECK=True

RANDOM_SEED=42
DATA_PATH="datasets/Zebra-CoT/"

GLOBAL_BATCH_SIZE=128      
BATCH_PER_DEVICE=1           
NUM_DEVICES=8
# GRAD_ACCUM_STEPS=16
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

# LLM-related params
LR=1e-5
SKETCH_LR=1e-5

# Ski-related params
SKETCH_LOSS=mse
SKETCH_LAMBDA=0.1
SKETCH_TOKEN_NUM=27

MAX_TOKEN=5120
MIN_TOKEN=128

RUN_NAME="skila/siglipvit_${PATTERN}_sketch${SKETCH_TOKEN_NUM}_${SKETCH_LAMBDA}${SKETCH_LOSS}_${MODEL_SIZE}_lr${LR}_bsz${GLOBAL_BATCH_SIZE}_maxImgToken${MAX_TOKEN}"
OUTPUT_DIR="skila/siglipvit_${PATTERN}_sketch${SKETCH_TOKEN_NUM}_${SKETCH_LAMBDA}${SKETCH_LOSS}_${MODEL_SIZE}_lr${LR}_bsz${GLOBAL_BATCH_SIZE}_maxImgToken${MAX_TOKEN}"


export PYTHONPATH=$(pwd)
deepspeed src/train/train_skila.py \
    --run_name "$RUN_NAME" \
    --deepspeed scripts/zero2.json \
    --pattern $PATTERN \
    --model_id $MODEL_NAME \
    --sketch_encoder $SKETCH_ENCODER \
    --data_path "$DATA_PATH" \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_merger True \
    --freeze_llm False \
    --learning_rate $LR \
    --sketch_lr $SKETCH_LR \
    --sketch_loss $SKETCH_LOSS\
    --sketch_lambda $SKETCH_LAMBDA \
    --sketch_token_num $SKETCH_TOKEN_NUM \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((MIN_TOKEN * 28 * 28)) \
    --image_max_pixels $((MAX_TOKEN * 28 * 28)) \
    --weight_decay 0.1 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 20 \
    --tf32 False \
    --gradient_checkpointing $GRAD_CHECK \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 10 \
    --dataloader_num_workers 8 \
    --random_seed $RANDOM_SEED \
    --report_to tensorboard # or wandb

