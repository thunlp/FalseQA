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
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from transformers import pipeline
from transformers import DataCollatorForLanguageModeling
from transformers.data.data_collator import DataCollatorMixin as HfDataCollatorMixin
from transformers.data.data_collator import torch_default_data_collator
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
parser.add_argument("--model_parallel", type=bool, default=True)
parser.add_argument('--model_name', type=str, default='opt-1.3b-da')
parser.add_argument('--seed', type=int, default=34)
parser.add_argument('--prefix', type=str, default='potential tricky question: ')
parser.add_argument('--prompt_text', type=str, default='')
parser.add_argument('--max_length', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epoch', type=int, default=8)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--time_stamp', type=str, default='none')
parser.add_argument('--test_only', type=str, default='False')
parser.add_argument('--scale', type=int, default=4)
parser.add_argument('--model_path', type=str, default='../../plm_cache/')
input_args = parser.parse_args()


# initialize
size_gate = 512
scale = input_args.scale
time_stamp = input_args.time_stamp
model_path = input_args.model_path
model_name = input_args.model_name
prefix = input_args.prefix
max_length = input_args.max_length
batch_size = input_args.batch_size
model_parallel = input_args.model_parallel
learning_rate = input_args.lr
epoch = input_args.epoch
test_only = input_args.test_only
save_dir = f"../trained_model/exp-2/train_exp-2_scale-{scale}_{model_name}_{time_stamp}_{input_args.seed}"
if test_only == 'True':
    output_file = f"../output/exp-2/test_exp-2_{model_name.replace('train_exp-2_', '')}.csv"
    model_checkpoint = f'../trained_model/exp-2/{model_name}'
    seed = int(model_name[model_name.rfind('_') + 1:])
    tokenizer = AutoTokenizer.from_pretrained(f'{model_checkpoint}/best', use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
else:
    output_file = f"../output/exp-2/train_exp-2_scale-{scale}_{model_name}_{time_stamp}_{input_args.seed}.csv"
    model_checkpoint = model_path + model_name
    seed = input_args.seed
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

setup_seed(seed)

label_map = {
    tokenizer.encode(' false', add_special_tokens=False)[0]: 0,
    tokenizer.encode(' true', add_special_tokens=False)[0]: 1
}

class DataCollator(HfDataCollatorMixin):
    def __init__(self, *args, **kwargs):
        self.return_tensors = 'pt'

    def torch_call(self, features):
        return torch_default_data_collator(features=features)


class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # inputs.pop('Unnamed: 0')
        labels = inputs.pop('labels')
        input_ids = inputs.get('input_ids')
        attention_mask = inputs.get('attention_mask')
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.get('logits')
        target_positions = labels.max(dim=1).indices.tolist()
        target_positions = [id - 1 for id in target_positions]
        interest_index = list(label_map.keys())
        temp_logits = logits[[range(logits.size(0))], target_positions].squeeze()
        try:
            selected_logits = temp_logits[:, interest_index] if temp_logits.dim() != 1 else temp_logits[interest_index].unsqueeze(dim=0)
        except IndexError:
            from IPython import embed; embed()
        final_labels = torch.tensor([label_map[key] for key in labels.max(dim=1).values.tolist()]).cuda()
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(selected_logits, final_labels)

        return (loss, outputs) if return_outputs else loss


def model_to_device(model, model_name):
    if "2.7b" in model_name:
        device_map = None # {
        #     0: [i for i in range(0, 4)],
        #     1: [i for i in range(4, 11)],
        #     2: [i for i in range(11, 18)],
        #     3: [i for i in range(18, 25)],
        #     4: [i for i in range(25, 32)],
        # }
    elif '350m' in model_name:
        device_map = None #{
        #     0: [i for i in range(0, 6)],
        #     1: [i for i in range(6, 12)],
        #     2: [i for i in range(12, 17)],
        #     3: [i for i in range(17, 22)],
        #     5: [i for i in range(22, 24)],
        # }
    elif '1.3b' in model_name:
        device_map = None #{
        #     0: [i for i in range(0, 8)],
        #     1: [i for i in range(8, 16)],
        #     2: [i for i in range(16, 24)],
        # }
    print(device_map)

    if model_parallel:
        print("Begin model parallel")
        model.parallelize(device_map=device_map)
        print("Done model parallel")
    else:
        model = model.cuda()
    return model


def train():
    print("The time stamp is " + time_stamp)
    print("load data start.")
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


    def preprocess_function(examples):
        questions = examples["question"]
        answers = ['true' if label == 1 else 'false' for label in examples['label']]

        model_inputs = [questions[i] + " " + input_args.prompt_text + answers[i] + "</s>" for i in range(len(questions))]

        answers = [" " + answers[i] + "</s>" for i in range(len(questions))]
        tokenized_answers = tokenizer(answers, add_special_tokens=False)
        tokenized_qa = tokenizer(model_inputs)

        # tokenized_questions = tokenizer(questions, padding='max_length', truncation=True, max_length=max_length)
        # tokenized_inputs = tokenizer(model_inputs, padding='max_length', truncation=True, max_length=max_length)
        tokenized_targets = []
        tokenized_inputs = {"labels": [], "input_ids": [], "attention_mask": []}
        max_length_with_pad = max_length
        for qid in range(len(model_inputs)):
            if len(tokenized_answers['input_ids'][qid]) > max_length_with_pad - 10:
                len_q = len(tokenized_qa['input_ids'][qid]) - len(tokenized_answers['input_ids'][qid])
                tokenized_answers['input_ids'][qid] = tokenized_answers['input_ids'][qid][:max_length_with_pad - 10]
                tokenized_answers['attention_mask'][qid] = tokenized_answers['attention_mask'][qid][
                                                        :max_length_with_pad - 10]
                tokenized_qa['input_ids'][qid] = tokenized_qa['input_ids'][qid][:max_length_with_pad - 10 + len_q]
                tokenized_qa['attention_mask'][qid] = tokenized_qa['attention_mask'][qid][:max_length_with_pad - 10 + len_q]
            ta = {k: tokenized_answers[k][qid] for k in tokenized_answers}
            tqa = {k: tokenized_qa[k][qid] for k in tokenized_qa}

            len_a = len(ta['input_ids'])
            len_qa = len(tqa['input_ids'])
            len_q = len_qa - len_a
            if len_a > max_length_with_pad - 10:
                raise RuntimeError("Max length too small")

            to_truncate = len_qa - max_length_with_pad
            if to_truncate > 0:
                input_ids = tqa['input_ids'][:len_q - to_truncate] + tqa['input_ids'][len_q:]
                attention_mask = tqa['attention_mask'][:len_q - to_truncate] + tqa['attention_mask'][len_q:]
                labels = [-100] * (len_q - to_truncate) + tqa['input_ids'][len_q:]
            else:
                input_ids = tqa['input_ids'] + [tokenizer.pad_token_id] * (-to_truncate)
                attention_mask = tqa['attention_mask'] + [0] * (-to_truncate)
                labels = [-100] * len_q + input_ids[len_q:len_qa + 1] + [-100] * (-to_truncate - 1)

            if not (len(input_ids) == len(attention_mask) == len(labels) == max_length_with_pad):
                from IPython import embed;
                embed(header="In preprocess function")
            # assert len(input_ids) == len(attention_mask) == len(labels) == max_length

            tokenized_inputs['labels'].append(labels)
            tokenized_inputs['input_ids'].append(input_ids)
            tokenized_inputs['attention_mask'].append(attention_mask)

        return tokenized_inputs


    encoded_dataset = raw_datasets.map(preprocess_function, batched=True, load_from_cache_file=False).shuffle(seed=seed)
    train_dataset = encoded_dataset['train']
    valid_dataset = encoded_dataset['validation']
    print("data process done.")


    print("model load start.")
    if '350m' in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_checkpoint, ignore_mismatched_sizes=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
    model = model_to_device(model, model_name)
    print("model load done.")
    

    print("set training arguments start.")
    args = TrainingArguments(
        save_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=epoch,
        logging_steps=100,
        load_best_model_at_end=True,
        remove_unused_columns=False
    )
    print("set training arguments done.")


    print("data collator load start.")    
    datacollator = DataCollator()
    print("data collator load done.")


    print("trainer load start.")
    trainer = MyTrainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=datacollator,
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


def test(model_checkpoint):
    test_set = pd.read_csv('../dataset/test.csv')
    test_list = []
    result_list = []
    if '350m' in model_name:
        model = AutoModelForCausalLM.from_pretrained(f'{model_checkpoint}/best', ignore_mismatched_sizes=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(f'{model_checkpoint}/best')
    model.cuda()
    
    
    for question in test_set['question']:
        test_list.append(question)


    print("generate start.")
    with torch.no_grad():
        for question in tqdm(test_list):
            input_ids = tokenizer(question, return_tensors='pt').input_ids.cuda()
            outputs = model(input_ids)
            logits = outputs.get('logits')
            interest_index = list(label_map.keys())
            pred = logits[0, -1, interest_index].argmax(dim=-1).item()
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
