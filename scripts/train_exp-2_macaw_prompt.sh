model_name=$1


#!/bin/bash


log_directory="log/exp-2"
if [ ! -d "$log_directory" ]; then
  mkdir -p "$log_directory"
  echo "build log_directory successfully"
fi

# CUDALIST=(0,1,2,3 4,5,6,7 0,1,2,3) #0,1 0,1 0,1 0,1 0,1 0,1 0,1 0,1 0,1 0,1 0,1 0,1 0,1)
CUDALIST=(0 1 2)
SCALES=(1187 1187 1187) # 4, 32, 128, 256, 512, 1187
SEEDS=(4 13 34)
timestamp=$(date "+%Y-%m-%d_%H-%M-%S")
lr=1e-4
epoch=5
batchsize=32
max_input_length=200
max_target_length=64

for outid in 0 # 2 3 4 5 #6 7 8 9 10 11
do
    for offset in 0 1 2
    do
        (
            idx=$(( $outid + $offset ))
            seed=${SEEDS[$idx]} 
            gpuid=${CUDALIST[$idx]} 
            scale=${SCALES[$idx]}
            echo "idx"$idx
            echo "seed"$seed
            echo "gpuid"$gpuid
            echo "scale"$scale
            echo "batchsize"$batchsize
            CUDA_VISIBLE_DEVICES=$gpuid python src/exp-2_macaw_prompt.py --model_name=$model_name \
                                                                    --seed=$seed \
                                                                    --max_input_length=$max_input_length \
                                                                    --max_target_length=$max_target_length \
                                                                    --batch_size=$batchsize \
                                                                    --time_stamp=$time_stamp \
                                                                    --lr=$lr --epoch=$epoch --scale=$scale >> "log/exp-2/$timestamp"_"train_exp-2_scale"-"$scale"_"$model_name"_"prompt"_$seed".log" 2>&1 
        )
    done
    wait
    rm -rf trained_model/exp-2/*macaw-11b*
    rm -rf trained_model/exp-2/*t5-3b*
    rm -rf trained_model/exp-2/*macaw-3b*
done
