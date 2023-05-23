import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from transformers import pipeline
from transformers import DataCollatorForLanguageModeling
from transformers.data.data_collator import DataCollatorMixin as HfDataCollatorMixin
from transformers.data.data_collator import torch_default_data_collator
import utils
from utils import compute_arc_f1, compute_fpr, compute_rougeL, compute_falseqa_score
from torch.nn import functional as F


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["WANDB_DISABLED"] = "true"


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class DataCollator(HfDataCollatorMixin):
    def __init__(self, *args, **kwargs):
        self.return_tensors = 'pt'

    def torch_call(self, features):
        # from IPython import embed; embed(header="In data collator")
        return torch_default_data_collator(features=features)


class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        global token_loss, classify_token_dict, loss_rate
        outputs = model(**inputs)
        logits = outputs.get('logits')
        if token_loss == 1:
            try:
                label_index_list = []
                for i, label in enumerate(inputs['labels']):
                    if (label == classify_token_dict['tricky']).nonzero(as_tuple=True)[0].size(0) == 1:
                        label_index_list.append((label == classify_token_dict['tricky']).nonzero(as_tuple=True)[0][0].item())
                    else:
                        label_index_list.append((label == classify_token_dict['true']).nonzero(as_tuple=True)[0][0].item())
                cls_token_logits = logits[[range(logits.size(0))], [index - 1 for index in label_index_list], :].squeeze()
                cls_labels = inputs['labels'][[range(logits.size(0))], label_index_list].squeeze()
            except TypeError:
                from IPython import embed; embed()
            cls_loss = F.cross_entropy(cls_token_logits, cls_labels)
            outputs['loss'] = outputs.get('loss') + cls_loss * loss_rate
        loss = outputs.get('loss')
        return (loss, outputs) if return_outputs else loss



parser = argparse.ArgumentParser(description='Input hyper-parameters')
parser.add_argument("--model_parallel", type=bool, default=True)
parser.add_argument('--token_loss', type=int, default=0)
parser.add_argument('--model_name', type=str, default='opt-2.7b-da')
parser.add_argument('--seed', type=int, default=34)
parser.add_argument('--prompt_text', type=str, default='')
parser.add_argument('--max_length', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epoch', type=int, default=8)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--time_stamp', type=str, default='none')
parser.add_argument('--test_only', type=str, default='False')
parser.add_argument('--scale', type=int, default=4)
parser.add_argument('--model_path', type=str, default='')
parser.add_argument('--loss_rate', type=float, default=1.0)
input_args = parser.parse_args()


# initialize
scale = input_args.scale
time_stamp = input_args.time_stamp
model_path = input_args.model_path
model_name = input_args.model_name
max_length = input_args.max_length
batch_size = input_args.batch_size
model_parallel = input_args.model_parallel
learning_rate = input_args.lr
epoch = input_args.epoch
test_only = input_args.test_only
global token_loss
global loss_rate
token_loss = input_args.token_loss
loss_rate = input_args.loss_rate
save_dir = f"../trained_model/exp-3/train_exp-3_scale-{scale}_{model_name}_{time_stamp}_{input_args.seed}"
if test_only == 'True':
    model_checkpoint = f'../trained_model/exp-3/{model_name}'
    seed = int(model_name[model_name.rfind('_') + 1:])
    tokenizer = AutoTokenizer.from_pretrained(f"{model_checkpoint}/best")
else:
    model_checkpoint = model_path + f"{model_name}"
    seed = input_args.seed
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
setup_seed(seed)
global classify_token_dict
classify_token_dict = {
    'tricky': tokenizer.encode(' tricky')[1],
    'true': tokenizer.encode(' true')[1]
}


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
    if scale != 1187:
        raw_datasets = load_dataset('csv', data_files={
            'train': f'../dataset/exp-3/train_{scale}shots_{input_args.seed}seed.csv', 'validation': f'../dataset/exp-3/valid_{scale}shots_{input_args.seed}seed.csv'
        })
    else:
        raw_datasets = load_dataset('csv', data_files={
            'train': '../dataset/exp-3/train_1187.csv', 'validation': '../dataset/exp-3/valid_1187.csv'
        })
    print("load data done.")


    print("data process start.")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    def preprocess_function(examples):
        questions = examples["question"]
        answers = ['tricky question. ' + examples['answer'][i] 
                    if examples['label'][i] == 1 
                    else "true question. " + examples['answer'][i] 
                    for i in range(len(questions))]
        model_inputs = [questions[i] + " " + input_args.prompt_text + answers[i] + "</s>" for i in range(len(questions))]
        
        answers = [" " + answers[i] + "</s>" for i in range(len(questions))]
        tokenized_answers = tokenizer(answers, add_special_tokens=False)
        tokenized_qa = tokenizer(model_inputs)

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
    torch.cuda.empty_cache()
    if '350m' in model_name:
        model = AutoModelForCausalLM.from_pretrained(f'{model_checkpoint}/best', ignore_mismatched_sizes=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(f'{model_checkpoint}/best')
    tokenizer = AutoTokenizer.from_pretrained(f'{model_checkpoint}/best', use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
    def test_iter(test_set_path, task_name):
        if test_only == 'True':
            output_file = f"../output/exp-3/test_exp-3_{task_name}_{model_name.replace('train_exp-3_', '')}.csv"
        else:
            output_file = f"../output/exp-3/train_exp-3_{task_name}_scale-{scale}_{model_name}_{time_stamp}_{input_args.seed}.csv"
        test_set = pd.read_csv(test_set_path)
        test_list = []
        result_list = []

        for question in test_set['question']:
            test_list.append(question)

        print("generate start.")
        result_list = generator(test_list, max_length=None, max_new_tokens=max_length)
        print("generate done.")


        final_list = []
        for i in range(len(result_list)):
            final_list.append(result_list[i][0]["generated_text"].replace(test_list[i], '').strip())

        final_result = pd.DataFrame({'question': test_set['question'],
                                    'answer': final_list,
                                    'label': test_set['label']})
        final_result.to_csv(output_file)
        print(f'save result in {output_file}')
        return output_file
    
    
    print("generate start.")
    arc_result_path = test_iter('../dataset/exp-3/arc_test.csv', 'arc-da')
    falseqa_result_path = test_iter('../dataset/exp-3/falseqa_test.csv', 'falseqa')
    print("generate done.")


    output_file_path_dict = {
        'arc-da': arc_result_path,
        'falseqa': falseqa_result_path
    }

    print("compute result start.")
    rouge_l = compute_rougeL(output_file_path_dict)
    print(f"arc-da Rouge-L: {rouge_l.get('arc-da')}")
    print(f"falseqa Rouge-L: {rouge_l.get('falseqa')}")
    print(f"total Rouge-L: {rouge_l.get('total')}")

    arc_f1 = compute_arc_f1(output_file_path_dict)
    print(f"arc-da F1: {arc_f1.get('arc-da')}")

    fpr = compute_fpr(output_file_path_dict)
    print(f"arc-da FPR: {fpr.get('arc-da')}")
    print(f"falseqa FPR: {fpr.get('falseqa')}")
    print(f"total FPR: {fpr.get('total')}")

    score_dict = compute_falseqa_score(output_file_path_dict)
    print(f'falseqa recall: ', score_dict.get('recall'))
    print(f'falseqa precision: ', score_dict.get('precision'))
    print(f'falseqa accuracy: ', score_dict.get('accuracy'))

    print("compute result end.")


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
