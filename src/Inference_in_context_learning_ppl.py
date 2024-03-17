import sys
import json
import random
import os
import re
import argparse
import numpy as np
import copy



import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from datasets import Dataset
import openai
import time


os.environ["WANDB_DISABLED"] = "true"






dataset_selection = 0
model_selection = 0
f_using_CoT = 0 # whether use CoT
f_ICL = 1  # whether use in-context learning during test
f_rewrite = 1 # whether rewrite existing test results
f_shorten_story = 1 # whether shorten the story


dataset_name = ['TGQA', 'TimeQA', 'TimeQA', 'TempReason', 'TempReason'][dataset_selection]
filename = ['TGSR_test.json', 'TGSR_easy_test.json', 'TGSR_hard_test.json', 'TGSR_l2_test.json', 'TGSR_l3_test.json'][dataset_selection]
Q_type = [None, None, None, 'l2', 'l3'][dataset_selection]
model_name = ['Llama-2-7b-hf', 'Llama-2-13b-hf', 'Llama-2-70b-hf'][model_selection]



def read_data(dataset_name, filename):
    file_path = f'../dataset/{dataset_name}/{filename}'
    with open(file_path) as json_file:
        data = json.load(json_file)

    # Convert list of dictionaries to the desired format
    data_dict = {'story': [item["story"] for item in data],
                 'Q': [item["question"] for item in data], 
                 'A': [item["answer"] for item in data],
                 'CoT': [item["CoT"] if "CoT" in item else None for item in data],
                 'C': [item["candidates"] if "candidates" in item else None for item in data],
                 'id': [item['id'] for item in data],
                 'Q-Type': [item['Q-Type'] if 'Q-Type' in item else Q_type for item in data]}

    # Convert your data into a dataset
    dataset = Dataset.from_dict(data_dict)

    return dataset



data_test = read_data(dataset_name, filename)
print(data_test)








def my_generate_prompt(story, Q, C, Q_type=None):
    if f_ICL:
        if dataset_name == 'TGQA':
            Q_type = f'Q{Q_type}'
        if not f_using_CoT:
            if Q_type is None:
                file_path = f'../materials/{dataset_name}/prompt_examples_ICL_SP.txt'
            else:
                file_path = f'../materials/{dataset_name}/prompt_examples_ICL_SP_{Q_type}.txt'
        else:
            if Q_type is None:
                file_path = f'../materials/{dataset_name}/prompt_examples_ICL_CoT.txt'
            else:
                file_path = f'../materials/{dataset_name}/prompt_examples_ICL_CoT_{Q_type}.txt'

        with open(file_path) as txt_file:
            prompt_examples = txt_file.read()

    story = story.replace('\n', ' ')

    if f_shorten_story:
        story = ' '.join(story.split(' ')[:2000])

    if '(' not in C[0] and ')' not in C[0]:
        C = ['( ' + cand + ' )' for cand in C]
    Q += ' Choose from ' + ', '.join(C) + '.'

    if f_ICL:
        prompt = f"Example:\n\n{prompt_examples}\n\n\n\nTest:\n\nStory: {story}\n\nQuestion: {Q}"
    else:
        prompt = f"Story: {story}\n\nQuestion: {Q}"


    if f_using_CoT:
        prompt += "\n\nAnswer: Let's think step by step.\n\n"
    else:
        prompt += "\n\nAnswer: "

    return prompt





for i in range(5):
    sample = data_test[i]
    prompt = my_generate_prompt(sample['story'], sample['Q'], sample['C'], Q_type=sample['Q-Type'])

    print(prompt)
    print('===============================')








model_name_cmp = f'meta-llama/{model_name}'
tokenizer = AutoTokenizer.from_pretrained(model_name_cmp)
tokenizer.pad_token_id = 0
tokenizer.padding_side = 'left'
model = AutoModelForCausalLM.from_pretrained(model_name_cmp,
                                            load_in_8bit=True,
                                            device_map="auto"
                                            )






def one_batch(input_prompts, samples, file_paths):
    for j in range(len(input_prompts)):
        cur_sample = samples[j]
        context_len = tokenizer(input_prompts[j], return_tensors="pt")["input_ids"].shape[1]
        op = cur_sample['C'][0]
        minloss = 10000
        for cand in cur_sample['C']:
            input_tokens = tokenizer(input_prompts[j] + cand, return_tensors="pt")["input_ids"].to("cuda")
            target_ids = input_tokens.clone()
            target_ids[:, :context_len] = -100
            with torch.no_grad():
                outputs = model(input_tokens, labels=target_ids)
                loss = outputs.loss.cpu().numpy()
                if loss < minloss:
                    minloss = copy.copy(loss)
                    op = copy.copy(cand)

        cur_sample.update({'prediction': op})

        with open(file_paths[j], 'w') as json_file:
            json.dump(cur_sample, json_file)


    return



def process_CoT(prediction):
    prediction = prediction.strip()
    for identifier in [' the answer is ', 'Answer:', ' answer is:', ' the correct answer is', ' the answers are ']:
        if identifier in prediction:
            prediction = prediction.split(identifier)[0].strip()
            break
    return prediction + ' the answer is '








folder_path = f'../results/{dataset_name}_ICL_{model_name}_ppl'
if not os.path.exists(folder_path):
    os.mkdir(folder_path)


if f_using_CoT:
    folder_path_past_res = f'../results/{dataset_name}_ICL_{model_name}'
    if not os.path.exists(folder_path_past_res):
        print('Error! Please first generate the CoT results with the command "python Inference_in_context_learning.py".')
        sys.exit()


batchsize = 8
input_prompts = []
file_paths = []
samples = []
for i in range(len(data_test)):
    file_path = f'{folder_path}/{str(i)}.json'
    if os.path.exists(file_path) and (not f_rewrite):
        continue

    sample = data_test[i]
    cur_prompt = my_generate_prompt(sample['story'], sample['Q'], sample['C'], Q_type=sample['Q-Type'])

    if f_using_CoT:
        file_path_past_res = f'{folder_path_past_res}/{str(i)}.json'
        if not os.path.exists(file_path_past_res):
            continue

        with open(file_path_past_res) as json_file:
            past_res = json.load(json_file)

        cur_prompt += process_CoT(past_res['prediction'])

    # print(cur_prompt)
    # print('-----------------------')


    input_prompts.append(cur_prompt)
    samples.append(sample)
    file_paths.append(file_path)

    if len(input_prompts) >= batchsize:
        one_batch(input_prompts, samples, file_paths)
        input_prompts = []
        file_paths = []
        samples = []


if len(input_prompts) > 0:
    one_batch(input_prompts, samples, file_paths)