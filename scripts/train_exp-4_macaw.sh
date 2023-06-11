#!/bin/bash
  
log_directory="log/exp-4"
if [ ! -d "$log_directory" ]; then
  mkdir -p "$log_directory"
  echo "build log_directory successfully"
fi

token_loss=0
max_input_length=200
max_target_length=100

for model_name in allenai/macaw-11b;
do

if [[ $model_name == *"/"* ]]; then
    model_name_for_log=$(echo "$model_name" | sed 's/\//-/g')
else
    model_name_for_log="$model_name"
fi

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
        for seed in 4 13 34;
        do
        CUDA_VISIBLE_DEVICES=0,1,2,3 python src/exp-4_replay_macaw.py --model_name=$model_name \
                                                                        --seed=$seed \
                                                                        --time_stamp=$time_stamp \
                                                                        --max_input_length=$max_input_length \
                                                                        --max_target_length=$max_target_length \
                                                                        --batch_size=$batch_size \
                                                                        --epoch=$epoch \
                                                                        --token_loss=$token_loss \
                                                                        --lr=$lr --scale=$scale >> "log/exp-4/$time_stamp"_"train_exp-4_scale"-"$scale"_"$model_name_for_log"_$seed".log" 2>&1
        done
    done
done