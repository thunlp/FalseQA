import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
from tqdm import tqdm


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["WANDB_DISABLED"] = "true"


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser(description='Input hyper-parameters')
parser.add_argument('--model_name', type=str, default='t5-large')
parser.add_argument('--seed', type=int, default=34)
parser.add_argument('--prefix', type=str, default='potential tricky question: ')
parser.add_argument('--max_input_length', type=int, default=100)
parser.add_argument('--max_target_length', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epoch', type=int, default=2)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--time_stamp', type=str, default='none')
parser.add_argument('--test_only', type=str, default='False')
parser.add_argument('--scale', type=int, default=4)
parser.add_argument('--use_local', type=int, default=0)
parser.add_argument('--model_path', type=str, default='../../plm_cache/')
input_args = parser.parse_args()


# initialize
size_gate = 512
scale = input_args.scale
time_stamp = input_args.time_stamp
model_path = input_args.model_path
model_name = input_args.model_name
prefix = input_args.prefix
max_input_length = input_args.max_input_length
max_target_length = input_args.max_target_length
batch_size = input_args.batch_size
learning_rate = input_args.lr
epoch = input_args.epoch
test_only = input_args.test_only
save_dir = f"../trained_model/exp-2/train_exp-2_scale-{scale}_{model_name}_{time_stamp}_{input_args.seed}"
if test_only == 'True':
    output_file = f"../output/exp-2/test_exp-2_{model_name.replace('train_exp-2_', '')}.csv"
    model_checkpoint = f'../trained_model/exp-2/{model_name}'
    seed = int(model_name[model_name.rfind('_') + 1:])
    if input_args.use_local == 1:
        tokenizer = T5Tokenizer.from_pretrained(f"{model_path}/{model_name}")
    else:
        tokenizer = T5Tokenizer.from_pretrained(model_name)
else:
    output_file = f"../output/exp-2/train_exp-2_scale-{scale}_{model_name}_{time_stamp}_{input_args.seed}.csv"
    model_checkpoint = model_path + f"{model_name}"
    seed = input_args.seed
    if input_args.use_local == 1:
        tokenizer = T5Tokenizer.from_pretrained(f"{model_path}/{model_name}")
    else:
        tokenizer = T5Tokenizer.from_pretrained(model_name)
setup_seed(seed)

# define label word map here
label_map = [
    tokenizer.encode('false', add_special_tokens=False)[0],
    tokenizer.encode('true', add_special_tokens=False)[0]
]


def preprocess_function(examples):
    """
    To preprocess the input in a teacher-forcing style.
    :param examples: input string
    :return: preprocessed inputs
    """
    if 't5' in model_name:
        inputs = [prefix + doc + " <extra_id_0>"
                for doc in examples["question"]]
    elif 'macaw' in model_name:
        inputs = ["$answer$ ; " + "$question$ = " + doc + " <extra_id_0>"
                for doc in examples["question"]]

    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    if 't5' in model_name:
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                ["<extra_id_0> " + "true" if label == 1
                else
                "<extra_id_0>" + "false" for label in examples["label"]],
                max_length=max_target_length, truncation=True)
    elif 'macaw' in model_name:
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                ["<extra_id_0> " + "$answer$ = " + "true" if label == 1
                else
                "<extra_id_0> " + "$answer$ = " + "false" for label in examples["label"]],
                max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


class MySeq2SeqTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        falseqa_label = inputs.pop("label")
        input_ids = inputs.get('input_ids')
        decoder_input_ids = inputs.get('decoder_input_ids')
        attention_mask = inputs.get('attention_mask')
        outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, attention_mask=attention_mask)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            labels = inputs.get('labels')
            logits = outputs.get('logits')
            interest_index = label_map
            loss_fct = nn.CrossEntropyLoss()
            temp_logits = logits[:, 1, :]
            selected_logits = temp_logits[:, interest_index]
            loss = loss_fct(selected_logits, falseqa_label)
            # loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss


def train():
    print("The time stamp is " + time_stamp)
    print("load data start.")

    raw_datasets = load_dataset('csv', data_files={
            'train': f'../dataset/train.csv', 'validation': f'../dataset/valid.csv', 'test': '../dataset/test.csv'
    })

    if scale <= size_gate:
        if scale == 512:
            raw_datasets = load_dataset('csv', data_files={
                'train': f'../dataset/exp-3/train_{scale}shots_{seed}seed.csv', 'validation': '../dataset/valid.csv', 'test': '../dataset/test.csv'
            })
        else:
            raw_datasets = load_dataset('csv', data_files={
                'train': f'../dataset/exp-3/train_{scale}shots_{seed}seed.csv', 'validation': f'../dataset/exp-3/valid_{scale}shots_{seed}seed.csv', 'test': '../dataset/test.csv'
            })
    else:
        raw_datasets = load_dataset('csv', data_files={
            'train': f'../dataset/train.csv', 'validation': f'../dataset/valid.csv', 'test': '../dataset/test.csv'
        })
    print("load data done.")

    print("data process start.")
    encoded_dataset = raw_datasets.map(preprocess_function, batched=True).shuffle().remove_columns(['question', 'answer'])
    train_dataset = encoded_dataset['train']
    valid_dataset = encoded_dataset['validation']
    print("data process done.")


    print("model load start.")
    if '11b' in model_checkpoint:
        model = T5ForConditionalGeneration.from_pretrained(model_checkpoint, torch_dtype=torch.bfloat16)
        print("bffloat16 done")
    else:
        model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)
    # model parallelization
    if torch.cuda.device_count() > 1:
        model.parallelize()
    print("model load done.")
    

    print("set training arguments start.")
    args = Seq2SeqTrainingArguments(
        save_dir,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=epoch,
        predict_with_generate=True,
        load_best_model_at_end=True,
    )
    print("set training arguments done.")


    print("data collator load start.")
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    print("data collator load done.")


    print("trainer load start.")
    trainer = MySeq2SeqTrainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    print("trainer load done.")


    print("model training start.")
    trainer.train()
    print("model training done.")

    print("saving best model start.")
    model.save_pretrained(f"{save_dir}/best")
    tokenizer.save_pretrained(f"{save_dir}/best")
    print("saving best model done.")
    model.cpu()


def test(model_checkpoint):
    test_set = pd.read_csv('../dataset/test.csv')
    test_list = []
    result_list = []
    # model.load_state_dict(torch.load(f"{model_checkpoint}/best/pytorch_model.bin"))
    print(f"load best from {model_checkpoint}")
    model = T5ForConditionalGeneration.from_pretrained(f"{model_checkpoint}/best")
    # generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=0)
    model.cuda()
    if 't5' in model_name:
        for question in test_set['question']:
            test_list.append(prefix + question)
    else:
        for question in test_set['question']:
            test_list.append(f'$answer$ ; $question$ = {question}')
    

    print("generate start.")
    for question in tqdm(test_list):
        input_ids = tokenizer.encode(question + ' <extra_id_0>', return_tensors='pt').cuda()
        decoder_input_ids = tokenizer.encode('<pad> <extra_id_0> ', return_tensors='pt').cuda()
        outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        logits = outputs.get('logits')
        interest_index = label_map
        pred = logits.squeeze()[1, interest_index].argmax().item()
        result_list.append(pred)

    print("generate done.")


    final_list = result_list


    tp = 0
    tn = 0
    for i in range(len(final_list)):
        if final_list[i] == 1 and test_set['label'][i] == 1:
            tp += 1
        elif final_list[i] == 0 and test_set['label'][i] == 0:
            tn += 1
    
    p_sum = 0
    n_sum = 0
    for label in test_set['label']:
        if label == 1:
            p_sum += 1
        else:
            n_sum += 1

    print("TP rate:", tp / p_sum)
    print("TN rate:", tn / n_sum)


    final_result = pd.DataFrame({'question': test_set['question'],
                                'judgement': final_list,
                                'label': test_set['label']})
    final_result.to_csv(output_file)
    print(f'save result in {output_file}')


if test_only == 'True':
    print()
    print(input_args)
    print()

    print('*'*60)
    print('start test...')
    print('*'*60)
    
    test(model_checkpoint)
else:
    print()
    print(input_args)
    print()
    
    print()
    print('*'*60)
    print('start train...')
    print('*'*60)
    print()

    train()

    print('*'*60)
    print('start test...')
    print('*'*60)

    test(save_dir)