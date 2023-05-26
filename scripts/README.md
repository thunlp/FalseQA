# Scripts

## EXP-2

A script for Macaw/OPT's prompt tuning.

### Hyperparameters

- model_name: Model name for prompt tuning.
- model_path: The path where the original pretrained models lie in.
- use_local: Whether to use local model or not (model_name='allenai/macaw-3b' when use_local else model_path+model_name).
- prefix: A prefix for prompt tuning template.
- time_stamp: A timestamp for training/testing log.
- scale: The scale of the data used for the experiment.

## EXP-3

A script for finetuing Macaw/OPT's to answer questions.

### Hyperparameters

- model_name: Model name for prompt tuning.
- model_path: The path where the original pretrained models lie in.
- use_local: Whether to use local model (i.e. model_name='allenai/macaw-3b' when use_local else model_path+model_name).
- prefix: A prefix for prompt tuning template.
- time_stamp: A timestamp for training/testing log.
- scale: The scale of the data used for the experiment.
- token_loss: Whether use token loss in training or not.
- loss_rate: The weight of the token_loss in training.

## EXP-4

A script for finetuing Macaw/OPT's to answer questions with data replay.

### Hyperparameters

- model_name: Model name for prompt tuning.
- model_path: The path where the original pretrained models lie in.
- use_local: Whether to use local model (i.e. model_name='allenai/macaw-3b' when use_local else model_path+model_name).
- prefix: A prefix for prompt tuning template.
- time_stamp: A timestamp for training/testing log.
- scale: The scale of the data used for the experiment.
- loss_rate: The weight of the token_loss in training.
- update_gate: Update the replay data every update_gate step.