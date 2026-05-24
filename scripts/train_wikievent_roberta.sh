#!/bin/bash
LR=3e-5

EXPERT=6
TOPK=1
LAMBDA=0.075
ARG_RE=1
SEEDS=(22 42 66 99 111 1234)

for SEED in "${SEEDS[@]}"
do
    work_path=exps/wikievent/$SEED
    mkdir -p $work_path

    python -u engine.py \
        --model_type ATSR \
        --dataset_type wikievent \
        --model_name_or_path ./roberta-large \
        --role_path ./data/dset_meta/description_wikievent.csv \
        --prompt_path ./data/prompts/prompts_wikievent_full.csv \
        --prompt_path1 ./data/prompts/prompts_wikievent_full_t1.csv \
        --prompt_path2 ./data/prompts/prompts_wikievent_full_t2.csv \
        --prompt_path3 ./data/prompts/prompts_wikievent_full_t4.csv \
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
        --device $DEVICE \
        --bipartite \

done
