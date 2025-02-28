#!/bin/bash
NNODES='1'
NODE_RANK='0'
MASTER_ADDR='127.0.0.1'
MASTER_PORT=12345

export RUN_NAME='Qwen2.5-VL-R1-Critic'

DISTRIBUTED_ARGS="
    --nproc_per_node 7 \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"


torchrun \
    --nproc_per_node="7" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    /opt/tiger/R1-Multimodal-Journey/src/open_r1/grpo_vllm.py \
    --deepspeed /opt/tiger/R1-Multimodal-Journey/local_scripts/zero3_offload.json \
    --output_dir results \
    --model_name_or_path /opt/tiger/Qwen2.5-VL-7B-Instruct \
    --dataset_name /opt/tiger/R1-Multimodal-Journey/local_scripts/geo_data \
    --max_prompt_length 1024 \
    --num_generations 8 \
    --report_to wandb \
    --max_completion_length 1536 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
    --bf16 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 1000000 \
    --num_train_epochs 1 \
    --save_only_model true \
