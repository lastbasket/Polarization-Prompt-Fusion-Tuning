#! /bin/bash
TIME=$(date +"%Y-%m-%d-%T")

MODEL_NAME="PPFT" # E.g. PPFT
CKPT_FILE="./ckpts/ppft_final/model.txt"

python main_refactored.py --dir_data ./data/hammer_polar \
                --data_name HAMMER \
                --data_txt ./data_paths/hammer_MODE.txt \
                --gpus 0 \
                --loss 1.0*L1+1.0*L2 \
                --log_dir ./experiments/ \
                --save ${MODEL_NAME}_test_${TIME} \
                --model ${MODEL_NAME} \
                --completionformer_mode rgbd \
                --pre_pvt \
                --pre_res \
                --pretrained_completionformer ./ckpts/NYUv2.pt \
                --use_pol \
                --pol_rep leichenyang-7 \
                --test \
                --data_percentage 1 \
                --pretrain_list_file ${CKPT_FILE}

for D_TYPE in 0 1 2
do
python main_refactored.py --dir_data ./data/hammer_polar \
                --data_name HAMMER \
                --data_txt ./data_paths/hammer_MODE.txt \
                --gpus 0 \
                --loss 1.0*L1+1.0*L2 \
                --log_dir ./experiments/ \
                --save ${MODEL_NAME}_test_${TIME} \
                --model ${MODEL_NAME} \
                --completionformer_mode rgbd \
                --pre_pvt \
                --pre_res \
                --pretrained_completionformer ./ckpts/NYUv2.pt \
                --use_pol \
                --pol_rep leichenyang-7 \
                --test \
                --data_percentage 1 \
                --pretrain_list_file ${CKPT_FILE} \
                --use_single \
                --depth_type ${D_TYPE}
done