#!/bin/bash
  
for model_name in macaw-11b;
do
time_stamp=$(date "+%Y-%m-%d_%H-%M-%S")
    for scale in 1187 256;
    do
    if [ $scale -eq 256 ]; then
        lr=2.5e-4
        epoch=3
        batch_size=4
    elif [ $scale -eq 1187 ]; then
        lr=1e-4
        epoch=3
        batch_size=4
    fi
    max_input_length=200
    max_target_length=100
    CUDA_VISIBLE_DEVICES=0,1,2,3 python exp-4_replay_macaw.py --model_name=$model_name --seed=34 --max_input_length=$max_input_length --max_target_length=$max_target_length --batch_size=$batch_size --epoch=$epoch --time_stamp=$time_stamp --lr=$lr --scale=$scale  >> "../log/exp-4/train_exp-4_scale"-"$scale"_"$model_name"_"once"_"$time_stamp"_"34.log" 2>&1
    CUDA_VISIBLE_DEVICES=0,1,2,3 python exp-4_replay_macaw.py --model_name=$model_name --seed=4 --max_input_length=$max_input_length --max_target_length=$max_target_length --batch_size=$batch_size --epoch=$epoch --time_stamp=$time_stamp --lr=$lr --scale=$scale >> "../log/exp-4/train_exp-4_scale"-"$scale"_"$model_name"_"once"_"$time_stamp"_"4.log" 2>&1
    CUDA_VISIBLE_DEVICES=0,1,2,3 python exp-4_replay_macaw.py --model_name=$model_name --seed=13 --max_input_length=$max_input_length --max_target_length=$max_target_length --batch_size=$batch_size --epoch=$epoch --time_stamp=$time_stamp --lr=$lr --scale=$scale >> "../log/exp-4/train_exp-4_scale"-"$scale"_"$model_name"_"once"_"$time_stamp"_"13.log" 2>&1
    wait
    done
done