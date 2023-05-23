#!/bin/bash
  
log_directory="log"
if [ ! -d "$log_directory" ]; then
  mkdir "$log_directory"
  echo "build log_directory successfully"
fi

for model_name in opt-2.7b-da;
do
time_stamp=$(date "+%Y-%m-%d_%H-%M-%S")
    for scale in 1187 256;
    do
    if [ $scale -eq 256 ]; then
        lr=3e-6
        epoch=12
        batch_size=32
    elif [ $scale -eq 1187 ]; then
        lr=6e-6
        epoch=8
        batch_size=32
    fi
    CUDA_VISIBLE_DEVICES=4,5,6,7 python exp-4_replay_opt.py --model_name=$model_name --seed=34 --epoch=$epoch --lr=$lr --time_stamp=$time_stamp --scale=$scale --batch_size=$batch_size >> "../log/exp-4/train_exp-4_scale"-"$scale"_"$model_name"_"once-cat"_"$time_stamp"_"34.log" 2>&1
    CUDA_VISIBLE_DEVICES=4,5,6,7 python exp-4_replay_opt.py --model_name=$model_name --seed=4 --epoch=$epoch --lr=$lr --time_stamp=$time_stamp --scale=$scale --batch_size=$batch_size >> "../log/exp-4/train_exp-4_scale"-"$scale"_"$model_name"_"once-cat"_"$time_stamp"_"4.log" 2>&1
    CUDA_VISIBLE_DEVICES=4,5,6,7 python exp-4_replay_opt.py --model_name=$model_name --seed=13 --epoch=$epoch --lr=$lr --time_stamp=$time_stamp --scale=$scale --batch_size=$batch_size >> "../log/exp-4/train_exp-4_scale"-"$scale"_"$model_name"_"once-cat"_"$time_stamp"_"13.log" 2>&1
    done
done