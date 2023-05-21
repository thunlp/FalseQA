#!/bin/bash
  
model_name=opt-2.7b-da

CUDALIST=(0,1,2,3 4,5,6 7 0,1,2,3 4,5,6 7 0,1,2,3 4,5,6 7 )
model_names=(opt-2.7b-da opt-1.3b-da opt-350m-da opt-2.7b-da opt-1.3b-da opt-350m-da opt-2.7b-da opt-1.3b-da opt-350m-da )
# CUDALIST=(0 1 0 1 0 1 0 1 0 1 0 1)
# CUDALIST=(0 1 0 1 0 1 0 1 0 1 0 1)
# SCALES=(512 512 512 256 256 256 128 128 128 32 32 32)
# SCALES=(512 512 512 256 256 256 128 128 128 32 32 32)
# SCALES=(10000 10000 10000)
SEEDS=(4 4 4 13 13 13 34 34 34)
timestamp=$(date "+%Y-%m-%d_%H-%M-%S")
lr=1e-5
epoch=5
# for outid in 0 2 4 6 8 10
for outid in 0 3 6 #1 2 3 4 5 6 7 8 9 10 11
do
    for offset in 0 1 2
    do
        (
            idx=$(( $outid + $offset ))
            seed=${SEEDS[$idx]} 
            gpuid=${CUDALIST[$idx]} 
            model_name=${model_names[$idx]} 
            scale=10000 #${SCALES[$idx]}
            batchsize=32
            echo "idx"$idx
            echo "seed"$seed
            echo "gpuid"$gpuid
            echo "scale"$scale
            echo "batchsize"$batchsize
            CUDA_VISIBLE_DEVICES=$gpuid python exp-2_opt_prompt.py --model_name=$model_name --seed=$seed --batch_size=$batchsize --time_stamp=$time_stamp --lr=$lr --epoch=$epoch --scale=$scale >> "../log_xxx/exp-2/$timestamp"_"train_exp-2_scale"-"$scale"_"$model_name"_"prompt"_$seed".log" 2>&1 
        ) &
    done
    wait
    rm -rf ../trained_model/exp-2/*opt-2.7b*
done
