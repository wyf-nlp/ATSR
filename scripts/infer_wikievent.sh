# Learning rate 
LR=3e-5

# seed
# SEEDS=(22 42 66 99 111 1234)
SEEDS=(1234)
for SEED in "${SEEDS[@]}"
do

    work_path=exps/wikievent/$SEED
    mkdir -p $work_path

    python -u engine.py \
        --dataset_type=wikievent \
        --context_representation=decoder \
        --model_name_or_path=roberta-large \
        --inference_only \
        --inference_model_path exps/wikievent/$SEED/checkpoint \
        --role_path=./data/dset_meta/description_wikievent.csv \
        --ontology_path=./data/templates/ontology_wikievent_full.csv \
        --template_path1=./data/templates/templates_wikievent_full_t1.csv \
        --template_path2=./data/templates/templates_wikievent_full_t2.csv \
        --template_path3=./data/templates/templates_wikievent_full_t4.csv \
        --seed=$SEED \
        --output_dir=$work_path \
        --learning_rate=$LR \
        --batch_size=4 \
        --max_steps=10000 \
        --max_enc_seq_length 500 \
        --max_template_seq_length 360 \
        --window_size 250 \
        --device cuda:3 \
        --bipartite \
        --lamb 0.1 \
        --use_arg_moe \
        --moe_num_experts 6 \
        --moe_top_k 1 \
        --lambd 0.075
done
