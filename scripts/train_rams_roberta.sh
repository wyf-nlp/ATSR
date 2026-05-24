#!/bin/bash
LR=3e-5

EXPERT=4
TOPK=1
LAMBDA=0.1
SEEDS=(22 42 66 99 111 1234)

for SEED in "${SEEDS[@]}"
do
    work_path=exps/rams/$SEED
    mkdir -p $work_path

    python -u engine.py \
        --model_type DEEIA \
        --dataset_type rams \
        --model_name_or_path ./roberta-large \
        --role_path ./data/dset_meta/description_rams.csv \
        --prompt_path ./data/prompts/prompts_rams_full.csv \
        --prompt_path1 ./data/prompts/prompts_rams_full_t1.csv \
        --prompt_path2 ./data/prompts/prompts_rams_full_t2.csv \
        --prompt_path3 ./data/prompts/prompts_rams_full_t4.csv \
        --seed $SEED \
        --output_dir $work_path \
        --learning_rate $LR \
        --max_steps 10000 \
        --max_enc_seq_length 500 \
        --max_prompt_seq_length 360 \
        --window_size 250 \
        --lamb 0.1 \
        --use_arg_moe \
        --moe_num_experts $EXPERT \
        --moe_top_k $TOPK \
        --lambd $LAMBD \
        --bipartite \

done
