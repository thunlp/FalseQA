#!/bin/bash
  
for model_name in macaw-11b;
do
time_stamp=$(date "+%Y-%m-%d_%H-%M-%S")
token_loss=0
# token_loss=1
    for scale in 1187 256 32;
    do
    if [ $scale -eq 32 ]; then
        lr=1e-4
        epoch=3
        batch_size=4
        max_input_length=200
        max_target_length=100
    elif [ $scale -eq 256 ]; then
        lr=2.5e-4
        epoch=3
        batch_size=4
        max_input_length=200
        max_target_length=100
    elif [ $scale -eq 1187 ]; then
        lr=1e-4
        epoch=3
        batch_size=4
        max_input_length=200
        max_target_length=100
    fi
    echo "scale"$scale
    CUDA_VISIBLE_DEVICES=4,5,6,7 python exp-3_macaw.py --model_name=$model_name --seed=34 --max_input_length=$max_input_length --max_target_length=$max_target_length --time_stamp=$time_stamp --lr=$lr --epoch=$epoch --scale=$scale --batch_size=$batch_size --token_loss=$token_loss >> "../log/exp-3/train_exp-3_scale"-"$scale"_"$model_name"_"$time_stamp"_"34.log" 2>&1
    CUDA_VISIBLE_DEVICES=4,5,6,7 python exp-3_macaw.py --model_name=$model_name --seed=4 --max_input_length=$max_input_length --max_target_length=$max_target_length --time_stamp=$time_stamp --lr=$lr --epoch=$epoch --scale=$scale --batch_size=$batch_size --token_loss=$token_loss >> "../log/exp-3/train_exp-3_scale"-"$scale"_"$model_name"_"$time_stamp"_"4.log" 2>&1
    CUDA_VISIBLE_DEVICES=4,5,6,7 python exp-3_macaw.py --model_name=$model_name --seed=13 --max_input_length=$max_input_length --max_target_length=$max_target_length --time_stamp=$time_stamp --lr=$lr --epoch=$epoch --scale=$scale --batch_size=$batch_size --token_loss=$token_loss >> "../log/exp-3/train_exp-3_scale"-"$scale"_"$model_name"_"$time_stamp"_"13.log" 2>&1
    wait
    done
done
