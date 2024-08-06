EXP_PATH=./experiments/base_exp/resnet18

for SPLIT_IDX in 0 1 2 3 4; do

        CUDA_VISIBLE_DEVICES=0 python main_osr.py --model_dir ${EXP_PATH} \
                                                    --num_class 20 \
                                                    --split_idx ${SPLIT_IDX} \
                                                    --image_size 64 \
                                                    --batch_size 32 

done