import os
import json
import numpy as np
from nltk.tokenize import word_tokenize
import collections



######### Config #########

dataset_selection = 0  # 0: TGQA, 1: TimeQA_easy, 2: TimeQA_hard, 3: TempReason_l2, 4: TempReason_l3
model_selection = 0  # 0: None 1: gpt-3.5-turbo, 2: gpt-4-1106-preview, 3: Llama-2-7b-hf, 4: Llama-2-13b-hf, 5: Llama-2-70b-hf  (only need for inference with ICL)

f_SFT_TGLLM = True  # whether use SFT with TGLLM
f_inference_ICL = False  # whether use inference with ICL
f_ppl = False  # whether use perplexity

###########################


dataset_name = ['TGQA', 'TimeQA', 'TimeQA', 'TempReason', 'TempReason'][dataset_selection]
split_name = ['', '_easy', '_hard', '_l2', '_l3'][dataset_selection]
model_name = [None, 'gpt-3.5-turbo', 'gpt-4-1106-preview', 'Llama-2-7b-hf', 'Llama-2-13b-hf', 'Llama-2-70b-hf'][model_selection]



if f_SFT_TGLLM:
    folder_path = f'../results/{dataset_name}_TGR{split_name}'

if f_inference_ICL:
    folder_path = f'../results/{dataset_name}_ICL{split_name}_{model_name}'

if f_ppl:
    folder_path += '_ppl'








def calculate_EM(a_gold, a_pred):
    # remove spaces and convert to lower case
    return a_gold.replace(' ', '').lower() == a_pred.replace(' ', '').lower()


def calculate_F1(a_gold, a_pred):
    # token-level F1
    gold_toks = word_tokenize(a_gold)
    pred_toks = word_tokenize(a_pred)

    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1



def parse_generation(pred):
    '''
    Parse the generated answer based on rules
    '''
    for start_identifier in ['Answer:', 'answer is']:
        if start_identifier in pred:
            pred = pred.split(start_identifier)[-1].strip()
            break

    for end_identifier in ['Test:']:
        if end_identifier in pred:
            pred = pred.split(end_identifier)[0].strip()
        break

    if '\n' in pred:
        pred = pred.split('\n')[-1].strip()

    if '(' in pred:
        pred = pred[len(pred.split('(')[0]) + 1:]

    if ')' in pred:
        pred = pred[:- (len(pred.split(')')[-1]) + 1)]

    if len(pred)>0 and pred[-1] in [')', '.']:
        pred = pred[:-1]

    pred = pred.strip()
    return pred




EM_dict = {}
f1_score_dict = {}


num_question_cat = 1
if dataset_name == 'TGQA':
   num_question_cat = 9  # for TGQA, there are 9 question categories, and we use avarage of them to avoid imbalance problem

for i in range(num_question_cat):
    EM_dict[i] = [0, 0]
    f1_score_dict[i] = []


num_test_samples = 10000
for i in range(num_test_samples):
    file_path = folder_path + f'/{str(i)}.json'
    if not os.path.exists(file_path):
        continue

    with open(file_path) as json_file:
        data = json.load(json_file)

    pred = data['prediction'].strip()
    pred = parse_generation(pred)

    gts = data['answer']
    gts = [gt[1:-1].strip() if gt[0] == '(' and gt[-1] == ')' else gt for gt in gts]

    if data['Q-Type'] is None:
        data['Q-Type'] = 0

    cur_f1_score = [calculate_F1(pred, gt) for gt in gts]
    f1_score_dict[data['Q-Type']].append(max(cur_f1_score))

    cur_EM = [calculate_EM(pred, gt) for gt in gts]
    EM_dict[data['Q-Type']][0] += max(cur_EM)
    EM_dict[data['Q-Type']][1] += 1


for i in range(num_question_cat):
    if EM_dict[i][1] > 0:
        EM_dict[i][0] = EM_dict[i][0]/EM_dict[i][1]




print('EM:')
print(np.mean([EM_dict[i][0] for i in range(num_question_cat) if EM_dict[i][1] > 0]), sum(EM_dict[i][1] for i in range(num_question_cat)))


# for results based on perplexity, we only need EM since we select the answer from the candidates
if not f_ppl:
    print('\nF1 score:')
    print(np.mean([np.mean(f1_score_dict[i]) for i in range(num_question_cat) if len(f1_score_dict[i]) > 0]), sum(len(f1_score_dict[i]) for i in range(num_question_cat)))