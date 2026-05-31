# Learning rate 
LR=3e-5

# seed
SEEDS=(22 42 66 99 111 1234)

for SEED in "${SEEDS[@]}"
do

    work_path=exps/mlee/$SEED
    mkdir -p $work_path

    python -u engine.py \
        --dataset_type=MLEE \
        --context_representation=decoder \
        --model_name_or_path=roberta-large \
        --inference_only \
        --inference_model_path exps/mlee/$SEED/checkpoint \
        --role_path=./data/MLEE/MLEE_role_name_mapping.json \
        --ontology_path=./data/templates/ontology_MLEE_full.csv \
        --seed=$SEED \
        --output_dir=$work_path \
        --learning_rate=$LR \
        --batch_size=4 \
        --max_steps=10000 \
        --max_enc_seq_length 500 \
        --max_template_seq_length 360 \
        --warmup_steps 0.2 \
        --bipartite \
        --use_arg_moe \
        --moe_num_experts 4 \
        --moe_top_k 1 \
        --lambd 0.1
done
