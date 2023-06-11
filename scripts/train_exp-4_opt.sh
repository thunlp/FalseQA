#!/bin/bash
  
log_directory="log/exp-4"
if [ ! -d "$log_directory" ]; then
  mkdir -p "$log_directory"
  echo "build log_directory successfully"
fi

token_loss=0

for model_name in opt-2.7b-da;
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
        lr=3e-6
        epoch=12
        batch_size=32
    elif [ $scale -eq 1187 ]; then
        lr=6e-6
        epoch=8
        batch_size=32
    fi
        for seed in 4 13 34;
        do
        CUDA_VISIBLE_DEVICES=0,1,2,3 python src/exp-4_replay_opt.py --model_name=$model_name \
                                                                    --time_stamp=$time_stamp \
                                                                    --seed=$seed \
                                                                    --epoch=$epoch \
                                                                    --lr=$lr --scale=$scale \
                                                                    --token_loss=$token_loss \
                                                                    --batch_size=$batch_size >> "log/exp-4/$time_stamp"_"train_exp-4_scale"-"$scale"_"$model_name_for_log"_$seed".log" 2>&1
        done
    done
done