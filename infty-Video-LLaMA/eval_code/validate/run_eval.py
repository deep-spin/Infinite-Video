from pathlib import Path
import pandas as pd
from pprint import pprint
from collections import Counter, defaultdict
import argparse
from utils import *
import json

def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
    parser.add_argument("--input_path", required=True, help="videos path")
    parser.add_argument("--dataset", required=True, type=str, help="path to save features.")
    args = parser.parse_args()
    return args

def eval_qa_videomme(data):
    num_valids = 0
    num_corrects = 0
    results = {}
    
    for uid, el in data.items():
        duration = el["duration"]
        
        # Initialize an empty list for each new duration category
        if duration not in results:
            results[duration] = {'valids': 0, 'corrects': 0}
        
        if el['prediction'] == -1:
            continue
        
        # Update the valid count for the overall and for the specific duration
        num_valids += 1
        results[duration]['valids'] += 1
        
        # Update corrects if the prediction matches the answer
        if el['answer'] == el['prediction']:
            num_corrects += 1
            results[duration]['corrects'] += 1
    
    # Compute overall accuracy
    overall_accuracy = num_corrects / len(data) if len(data) > 0 else 0
    
    # Compute accuracy for each duration category
    category_accuracies = {}
    for duration, counts in results.items():
        category_accuracies[duration] = counts['corrects'] / counts['valids'] if counts['valids'] > 0 else 0
    
    stat = {
        'num_total': len(data),
        'num_valids': num_valids,
        'num_corrects': num_corrects,
        'overall_acc': overall_accuracy,
        'category_accuracies': category_accuracies
    }
    print(stat)
    return stat

def eval_qa_videomme_from_file(fp):
    data = load_json(fp)
    if 'data' in data:
        data = data['data']
    eval_qa_videomme(data)

def eval_qa_egoschema(data):
    num_valids = 0
    num_corrects = 0
    for uid, el in data.items():
        if el['prediction'] == -1:
            continue
        num_valids += 1
        if el['answer'] == el['prediction']:
            num_corrects += 1 
    stat = {
        'num_total': len(data),
        'num_valids': num_valids,
        'num_corrects': num_corrects,
        'acc': num_corrects / len(data),
    }
    pprint(stat)
    stat['data'] = data
    return stat

def eval_qa_egoschema_from_file(fp):
    data = load_json(fp)
    if 'data' in data:
        data = data['data']
    eval_qa_egoschema(data)

def eval_qa_nextqa_from_file(anno_file_path, pred_file_path):
    data = load_json(pred_file_path)
    if 'data' in data:
        data = data['data']
    eval_qa_nextqa(anno_file_path, data)

def eval_sum(data):
    num_words_ls = []
    for example in data.values():
        summarization = example['response']
        num_words = len(summarization.replace('.', ' ').replace(',', ' ').replace('\n', ' ').split(' '))
        num_words_ls.append(num_words)
    num_words_series = pd.Series(num_words_ls)
    stat = {
        'min': float(num_words_series.min()),
        'max': float(num_words_series.max()),
        'mean': float(num_words_series.mean()),
        'std': float(num_words_series.std()),
    }
    stat['data'] = data
    sum_data = {uid: el['response'] for uid, el in data.items()}
    return stat, sum_data

def eval_qa_nextqa(anno_file_path, preds):
    '''
    This function was adapted from https://github.com/doc-doc/NExT-QA/blob/main/eval_mc.py
    '''
    with open(preds, 'r') as file:
        preds = json.load(file)
    map_name = {'CW': 'Why', 'CH': 'How', 'TN': 'Bef&Aft', 'TC': 'When', 'DC': 'Cnt', 'DL': 'Loc', 'DO': 'Other', 'C': 'Acc_C', 'T': 'Acc_T', 'D': 'Acc_D'}
    sample_list = pd.read_csv(anno_file_path)
    group = {'CW':[], 'CH':[], 'TN':[], 'TC':[], 'DC':[], 'DL':[], 'DO':[]}
    for id, row in sample_list.iterrows():
        qns_id = str(row['video']) + '_' + str(row['qid'])
        if qns_id not in preds:
            continue
        qtype = str(row['type'])
        #(combine temporal qns of previous and next as 'TN')
        if qtype == 'TP': qtype = 'TN'
        group[qtype].append(qns_id)

    group_acc = {'CW': 0, 'CH': 0, 'TN': 0, 'TC': 0, 'DC': 0, 'DL': 0, 'DO': 0}
    group_cnt = {'CW': 0, 'CH': 0, 'TN': 0, 'TC': 0, 'DC': 0, 'DL': 0, 'DO': 0}
    overall_acc = {'C':0, 'T':0, 'D':0}
    overall_cnt = {'C':0, 'T':0, 'D':0}
    all_acc = 0
    all_cnt = 0
    for qtype, qns_ids in group.items():
        cnt = 0
        acc = 0
        for qid in qns_ids:

            cnt += 1
            answer = preds[qid]['answer']
            pred = preds[qid]['prediction']

            if answer == pred: 
                acc += 1

        group_cnt[qtype] = cnt
        group_acc[qtype] += acc
        overall_acc[qtype[0]] += acc
        overall_cnt[qtype[0]] += cnt
        all_acc += acc
        all_cnt += cnt

    for qtype, value in overall_acc.items():
        group_acc[qtype] = value
        group_cnt[qtype] = overall_cnt[qtype]

    stat = {}
    for qtype in group_acc:
        print(map_name[qtype], end='\t')
    print('')
    for qtype, acc in group_acc.items():
        if group_cnt[qtype] == 0:
            stat[qtype] = 0
            print('{:.2f}'.format(0), end ='\t')
        else:
            stat[qtype] = acc*100.0/group_cnt[qtype]
            print('{:.2f}'.format(acc*100.0/group_cnt[qtype]), end ='\t')
    stat['Acc'] = all_acc*100.0/all_cnt
    print(all_cnt)
    print('Acc: {:.2f}'.format(all_acc*100.0/all_cnt))
    stat['data'] = preds
    return stat


if __name__ == '__main__':

    args = parse_args()
    if args.dataset == "egoschema":
        eval_qa_egoschema_from_file(args.input_path)
    elif args.dataset == "videomme":
        eval_qa_videomme_from_file(args.input_path)
    elif args.dataset == "nextqa":
        eval_qa_nextqa("/mnt/scratch-artemis/saul/next_qa/test.csv", args.input_path)