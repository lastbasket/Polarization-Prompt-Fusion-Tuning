#! /bin/bash

GREEN='\033[0;32m'
NC='\033[0m'
MODEL_NAME=${1}

if [ -z "$MODEL_NAME" ]; then
    echo -e ${GREEN}Model name unspecified, by default will be training "PPFT". Other choices are "CompletionFormer".${NC}
    MODEL_NAME=PPFT
fi

TIME=$(date +"%Y-%m-%d-%T")

python main.py --dir_data ./data/hammer_polar \
                --data_name HAMMER \
                --data_txt ./data_paths/hammer_MODE.txt \
                --gpus 0 \
                --loss 1.0*L1+1.0*L2 \
                --batch_size 14 \
                --epochs 250 \
                --log_dir ./experiments/ \
                --save ${MODEL_NAME}_train_${TIME} \
                --model ${MODEL_NAME} \
                --completionformer_mode rgbd \
                --pre_pvt \
                --pre_res \
                --save_full \
                --warm_up \
                --port 29503 \
                --lr 0.00105 \
                --pretrained_completionformer ./ckpts/NYUv2.pt \
                --use_pol \
                --pol_rep leichenyang-7 \
                --data_percentage 1 \