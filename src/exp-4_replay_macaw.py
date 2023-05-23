import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
import utils
from utils import compute_arc_f1, compute_fpr, compute_rougeL, compute_falseqa_score
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn import functional as F
from datasets import Dataset


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["WANDB_DISABLED"] = "true"


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class MySeq2SeqTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        global train_iter_count, already_seen_count, episodic_batch, update_gate
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        falseqa_label = inputs.pop("label")
        if model.training:
            if train_iter_count % update_gate == 0:
                episodic_batch = sample_normal(raw_episodic_dataset, batch_size, train_iter_count, update_gate)
            train_iter_count += 1
            episodic_input_ids = episodic_batch.get('input_ids').cuda()
            episodic_labels = episodic_batch.get('labels').cuda()
            episodic_attention_mask = episodic_batch.get('attention_mask').cuda()
            cat_inputs = torch.cat((episodic_input_ids, inputs['input_ids']))
            cat_labels = torch.cat((episodic_labels, inputs['labels']))
            cat_attention_mask = torch.cat((episodic_attention_mask, inputs['attention_mask']))
            cat_inputs.cuda()
            cat_labels.cuda()
            cat_attention_mask.cuda()
            
            outputs = model(input_ids=cat_inputs, labels=cat_labels, attention_mask=cat_attention_mask)
            logits = outputs.get('logits')
            cls_labels = cat_labels[:, 6]
            cls_loss = F.cross_entropy(logits[:, 6, :], cls_labels)
            outputs["loss"] = cls_loss + outputs.loss
        else:
            print('in evaluate, set train_iter_count to 0.')
            outputs = model(input_ids=inputs['input_ids'], labels=inputs['labels'], attention_mask=inputs['attention_mask'])

            logits = outputs.get('logits')
            cls_labels = inputs['labels'][:, 6]
            cls_loss = F.cross_entropy(logits[:, 6, :], cls_labels)
            outputs["loss"] = cls_loss + outputs.loss
            train_iter_count = 0

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss


parser = argparse.ArgumentParser(description='Input hyper-parameters')
parser.add_argument('--model_name', type=str, default='allenai/macaw-3b')
parser.add_argument('--seed', type=int, default=34)
parser.add_argument('--prefix', type=str, default='potential tricky question: ')
parser.add_argument('--max_input_length', type=int, default=100)
parser.add_argument('--max_target_length', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--epoch', type=int, default=5)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--time_stamp', type=str, default='none')
parser.add_argument('--scale', type=int, default=256)
parser.add_argument('--test_only', type=str, default='False')
parser.add_argument('--update_gate', type=int, default=30)
parser.add_argument('--model_path', type=str, default='')
input_args = parser.parse_args()


# initialize
global already_seen_count, train_iter_count, episodic_batch, sample_id_list, update_gate
already_seen_count = 0
train_iter_count = 0
episodic_batch = None
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
scale = input_args.scale
update_gate = input_args.update_gate
save_dir = f"../trained_model/exp-4/train_exp-4_scale-{scale}_{model_name}_{time_stamp}_{input_args.seed}"
if test_only == 'True':
    model_checkpoint = f'../trained_model/exp-4/{model_name}'
    seed = int(model_name[model_name.rfind('_') + 1:])
    tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
else:
    model_checkpoint = model_path + f"{model_name}"
    seed = input_args.seed
    tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
setup_seed(seed)


def preprocess_function(examples):
    """
    To preprocess the input in a teacher-forcing style.
    :param examples: input string
    :return: preprocessed inputs
    """
    inputs = ["$answer$ ; " + "$question$ = " + doc + " <extra_id_0>"
            for doc in examples["question"]]

    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding='max_length')

    with tokenizer.as_target_tokenizer():
        try:
            labels = tokenizer(
                ["<extra_id_0> " + "$answer$ = tricky question. " + answer if label == 1
                else
                "<extra_id_0> " + "$answer$ = true question. " + answer
                for label, answer in zip(examples["label"], examples['answer'])],
                max_length=max_target_length, truncation=True, padding='max_length')
        except TypeError:
            from IPython import embed; embed()
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def sample_normal(dataset, batch_size, train_iter_count, update_gate):
    global sample_id_list
    sample_dataset = {'question': [], 'answer': [], 'label': []}
    former_id = train_iter_count / update_gate
    latter_id = train_iter_count / update_gate + 1
    fid = former_id * batch_size
    lid = latter_id * batch_size
    if lid > len(sample_id_list):
        print('should not happen.')
        fid = 0
        lid = batch_size
    for count, id in enumerate(sample_id_list[int(fid): int(lid)]):
        sample_dataset['question'].append(dataset['question'][id])
        sample_dataset['answer'].append(dataset['answer'][id])
        sample_dataset['label'].append(dataset['label'][id])
    sample_dataset = Dataset.from_dict(sample_dataset)
    encoded_episodic_dataset = sample_dataset.map(preprocess_function, batched=True, load_from_cache_file=False)
    episodic_dataset = encoded_episodic_dataset.with_format('torch')
    episodic_dataloader = DataLoader(episodic_dataset, batch_size=batch_size)
    return_batch = None
    for batch in episodic_dataloader:
        return_batch = batch
        break
    return return_batch


print("load data start.")
if scale != 1187:
    raw_datasets = load_dataset('csv', data_files={
        'train': f'../dataset/exp-3/train_{scale}shots_{input_args.seed}seed.csv', 'validation': f'../dataset/exp-3/valid_{scale}shots_{input_args.seed}seed.csv'
    })
else:
    raw_datasets = load_dataset('csv', data_files={
    'train': f'../dataset/exp-3/train_1187.csv', 'validation': f'../dataset/exp-3/valid_1187.csv'
})
raw_episodic_dataset = load_dataset('csv', data_files={
    'train': f'../dataset/exp-4/train_arc-da.csv'
})
print("load data done.")

print("data process start.")
encoded_dataset = raw_datasets.map(preprocess_function, batched=True, load_from_cache_file=False).shuffle(seed=seed)
train_dataset = encoded_dataset['train']
valid_dataset = encoded_dataset['validation']


raw_episodic_dataset = raw_episodic_dataset.shuffle(seed=seed)['train']
sample_id_list = random.sample(range(len(raw_episodic_dataset)), len(raw_episodic_dataset))

print("data process done.")


def train():
    print("The time stamp is " + time_stamp)

    print("model load start.")
    if 'macaw-11b' in model_checkpoint:
        model = T5ForConditionalGeneration.from_pretrained(model_checkpoint, torch_dtype=torch.bfloat16)
    else:
        model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)
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
    return model


def test(model_checkpoint, model=None):
    if model is None:
        if 'macaw-11b' in model_checkpoint:
            model = T5ForConditionalGeneration.from_pretrained(f'{model_checkpoint}/best', torch_dtype=torch.bfloat16)
        else:
            model = T5ForConditionalGeneration.from_pretrained(f'{model_checkpoint}/best')
        model.cuda().parallelize()

    generator = model

    def test_iter(test_set_path, task_name):
        if test_only == 'True':
            output_file = f"../output/exp-4/test_exp-4_{task_name}_{model_name.replace('train_exp-4_', '')}.csv"
        else:
            output_file = f"../output/exp-4/train_exp-4_{task_name}_scale-{scale}_{model_name}_{time_stamp}_{input_args.seed}.csv"
        test_set = pd.read_csv(test_set_path)
        test_list = []
        for question in test_set['question']:
            test_list.append(f'$answer$ ; $question$ = {question}')


        result_list = []
        for question in tqdm(test_list):
            input_ids = tokenizer(question, return_tensors='pt').input_ids.cuda()
            outputs = generator.generate(input_ids, max_length=None, max_new_tokens=max_target_length)
            result_list.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
        final_list = []
        for result in result_list:
            final_list.append(result.replace('answer$ = ', '').replace('$', '').strip())
        final_result = pd.DataFrame({'question': test_set['question'],
                                    'answer': final_list,
                                    'label': test_set['label']})
        final_result.to_csv(output_file)
        print(f'save result in {output_file}')
        return output_file


    print("generate start.")
    falseqa_result_path = test_iter('../dataset/exp-4/falseqa_test.csv', 'falseqa')
    arc_result_path = test_iter('../dataset/exp-4/arc_test.csv', 'arc-da')
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


    falseqa_scores = compute_falseqa_score(output_file_path_dict)
    print('falseqa recall:', falseqa_scores.get('recall'))
    print('falseqa precision:', falseqa_scores.get('precision'))
    print('falseqa accuracy:', falseqa_scores.get('accuracy'))

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

    model = train()

    print('*'*60)
    print('start test...')
    print('*'*60)

    test(save_dir, model)