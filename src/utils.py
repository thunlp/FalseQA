from openprompt.utils import crossfit_metrics as metric
import pandas as pd
import numpy as np
# import evaluate

arc_test = pd.read_csv('../dataset/exp-3/arc_test.csv')
falseqa_test = pd.read_csv('../dataset/exp-3/falseqa_test.csv')


def get_output(output_file_path_dict):
    arc_da_path = output_file_path_dict.get('arc-da')
    falseqa_path = output_file_path_dict.get('falseqa')
    arc_da_output = pd.read_csv(arc_da_path)
    falseqa_output = pd.read_csv(falseqa_path)
    return (arc_da_output, falseqa_output)

def lower_obj(obj):
    # return str obj
    if type(obj) == list:
        return [str(item).lower() for item in obj]
    else:
        return str(obj).lower()


def get_output_transfer(output_file_path_dict):
    seen_path = output_file_path_dict.get('seen')
    noseen_path = output_file_path_dict.get('noseen')
    seen_output = pd.read_csv(seen_path)
    noseen_output = pd.read_csv(noseen_path)
    return (seen_output, noseen_output)


def compute_rougeL(output_file_path_dict):
    arc_da_output, falseqa_output = get_output(output_file_path_dict)
    #   arc-da
    rouges_arc_da = []
    for (prediction, target) in zip(arc_da_output['answer'], arc_test['answer']):
        rouges_arc_da.append(metric.get_rouge_over_list(lower_obj(str(prediction)).replace('true question.', '').replace('tricky question.', ''), lower_obj(eval(target)) if target.find('[') != -1 else str(target)))

    #   falseqa
    rouges_falseqa = []
    for (prediction, target) in zip(falseqa_output[falseqa_output.label == 1]['answer'], falseqa_test[falseqa_test.label == 1]['answer']):
        rouges_falseqa.append(metric.get_rouge_over_list(lower_obj(str(prediction)).replace('tricky question.', '').replace('true question.', ''), lower_obj(eval(target)[0:2])))
    return {
        'arc-da': np.mean(rouges_arc_da),
        'falseqa': np.mean(rouges_falseqa),
        'total': np.mean(rouges_falseqa + rouges_arc_da)
    }


def compute_fpr(output_file_path_dict):
    arc_da_output, falseqa_output = get_output(output_file_path_dict)
    arc_fp_count = 0
    falseqa_fp_count = 0
    total_length = len(arc_da_output) + len(falseqa_output[falseqa_output.label == 0])
    arc_fpr = 0
    falseqa_fpr = 0


    #   arc-da
    for i in range(len(arc_da_output)):
        if 'tricky question' in str(arc_da_output['answer'][i]).lower():
            arc_fp_count += 1
    arc_fpr = arc_fp_count / len(arc_da_output)


    #   falseqa
    for i in range(len(falseqa_output)):
        if 'tricky question' in str(falseqa_output['answer'][i]).lower() and falseqa_output['label'][i] == 0:
            falseqa_fp_count += 1
    falseqa_fpr = falseqa_fp_count / len(falseqa_output[falseqa_output.label == 0])


    total_fpr = (arc_fp_count + falseqa_fp_count) / total_length
    return {
        'arc-da': arc_fpr,
        'falseqa': falseqa_fpr,
        'total': total_fpr
    }
    
def compute_tpr(output_file_path_dict):
    arc_da_output, falseqa_output = get_output(output_file_path_dict)
    falseqa_tp_count = 0
    falseqa_tpr = 0


    #   falseqa
    for i in range(len(falseqa_output)):
        if 'tricky question' in str(falseqa_output['answer'][i]).lower() and falseqa_output['label'][i] == 1:
            falseqa_tp_count += 1
    falseqa_tpr = falseqa_tp_count / len(falseqa_output[falseqa_output.label == 1])


    return {
        'falseqa': falseqa_tpr,
    }


def compute_tnr(output_file_path_dict):
    arc_da_output, falseqa_output = get_output(output_file_path_dict)
    falseqa_tn_count = 0
    falseqa_tnr = 0

    #   falseqa
    for i in range(len(falseqa_output)):
        if 'true question' in str(falseqa_output['answer'][i]).lower() and falseqa_output['label'][i] == 0:
            falseqa_tn_count += 1
    falseqa_tnr = falseqa_tn_count / len(falseqa_output[falseqa_output.label == 0])


    return {
        'falseqa': falseqa_tnr,
    }


def compute_arc_f1(output_file_path_dict):
    arc_da_output, falseqa_output = get_output(output_file_path_dict)
    f1s = []
    for (prediction, target) in zip(arc_da_output['answer'], arc_test['answer']):
        f1s.append(metric.get_f1_over_list(lower_obj(str(prediction)).replace('true question.', '').replace('tricky question.', ''), lower_obj(eval(target)) if target.find('[') != -1 else str(target)))
    return {
        'arc-da': np.mean(f1s)
    }


def compute_falseqa_score(output_file_path_dict):
    arc_da_output, falseqa_output = get_output(output_file_path_dict)
    tpr = compute_tpr(output_file_path_dict).get('falseqa')
    tnr = compute_tnr(output_file_path_dict).get('falseqa')
    recall = tpr
    precision = tpr / (tpr+1-tnr)
    accuracy = (tpr + tnr) / 2
    return {
        'recall': recall,
        'precision': precision,
        'accuracy': accuracy
    }

if __name__ == '__main__':
    file_name = input()
    arc_da = file_name.replace('falseqa', 'arc-da')
    prefix = ''
    output_file_path_dict = {
        'arc-da': prefix + arc_da + '.csv',
        'falseqa': prefix + file_name + '.csv',
        }
    print(output_file_path_dict)
    score_dict = compute_falseqa_score(output_file_path_dict)
    print(f'falseqa recall: ', score_dict.get('recall'))
    print(f'falseqa precision: ', score_dict.get('precision'))
    print(f'falseqa accuracy: ', score_dict.get('accuracy'))
    rouge_l = compute_rougeL(output_file_path_dict)
    print(f"falseqa Rouge-L: {rouge_l.get('falseqa')}")
    print(f"arc-da Rouge-L: {rouge_l.get('arc-da')}")

    fpr = compute_fpr(output_file_path_dict)
    print(f"falseqa FPR: {fpr.get('falseqa')}")
    print(f"arc-da FPR: {fpr.get('arc-da')}")

    arc_f1 = compute_arc_f1(output_file_path_dict)
    print(f"arc-da F1: {arc_f1.get('arc-da')}")