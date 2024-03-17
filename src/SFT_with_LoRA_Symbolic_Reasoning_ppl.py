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
from peft import (
        get_peft_model, 
        prepare_model_for_kbit_training, 
        LoraConfig
    )
from trl import SFTTrainer
from peft import PeftModel
from datasets import Dataset


os.environ["WANDB_DISABLED"] = "true"






dataset_selection = 0
f_ICL = 1  # whether use in-context learning during test
f_rewrite = 1 # whether rewrite existing test results


dataset_name = ['TGQA', 'TimeQA_easy', 'TimeQA_hard', 'TempReason_l2', 'TempReason_l3'][dataset_selection]
Q_type = ['', 'easy', 'hard', 'l2', 'l3'][dataset_selection]
dataset_name_short = dataset_name.split('_')[0]





def read_data(dataset_name, filename, f_CoT_bs=0, f_data_aug=0):
    file_path = f'../dataset/{dataset_name}/{filename}'

    with open(file_path) as json_file:
        data = json.load(json_file)

    # Convert list of dictionaries to the desired format
    data_dict = {'story': [item["story"] for item in data],
                 'TG': [item["TG"] for item in data],
                 'Q': [item["question"] for item in data], 
                 'A': [item["answer"] for item in data],
                 'EK': [item["EK"] if "EK" in item else None for item in data],
                 'CoT': [item["CoT"] if "CoT" in item else None for item in data],
                 'C': [item["candidates"] if "candidates" in item else None for item in data],
                 'id': [item['id'] for item in data],
                 'Q-Type': [item['Q-Type'] if 'Q-Type' in item else Q_type for item in data]}

    # Convert your data into a dataset
    dataset = Dataset.from_dict(data_dict)

    return dataset






filename = ['TGSR_test.json', 'TGSR_easy_test.json', 'TGSR_hard_test.json', 'TGSR_l2_test.json', 'TGSR_l3_test.json'][dataset_selection]
data_test = read_data(dataset_name_short, filename)

print(data_test)




TG_pred = {}
path_TG_pred = f'../results/{dataset_name_short}_story_TG_trans/'
for filename in os.listdir(path_TG_pred):
    file_path = os.path.join(path_TG_pred, filename)
    with open(file_path) as json_file:
        data = json.load(json_file)
    TG_pred[data['id']] = data['prediction']







def process_id(sample_id):
    story_id = sample_id
    if dataset_name_short == 'TimeQA':
        story_id = story_id[:-2]
    if dataset_name_short == 'TempReason':
        story_id = story_id[2:-2]
    return story_id


def my_generate_prompt(TG, EK, Q, CoT, A, Q_type=None, mode=None, eos_token="</s>"):
    if isinstance(TG, list):
        TG = '\n'.join(TG)

    if f_ICL and mode == 'test':
        if dataset_name == 'TGQA':
            Q_type = f'Q{Q_type}'
        if Q_type is None:
            file_path = f'../materials/{dataset_name_short}/prompt_examples_TGSR.txt'
        else:
            file_path = f'../materials/{dataset_name_short}/prompt_examples_TGSR_{Q_type}.txt'
        with open(file_path) as txt_file:
            prompt_examples = txt_file.read()

    if f_ICL and mode == 'test':
        prompt = f"Example:\n\n{prompt_examples}\n\nTest:\n\nTimeline:\n{TG}\n\nQuestion: {Q}"
    else:
        prompt = f"Timeline:\n{TG}\n\nQuestion: {Q}"

    if EK is not None:
        if isinstance(EK, list):
            EK = '\n'.join(EK)
        prompt += f"\n\nUseful information:\n{EK}"

    prompt += "\n\nAnswer: Let's think step by step.\n\n"

    if CoT is not None:
        if isinstance(CoT, list):
            CoT = CoT[0]
        prompt += CoT

    prompt += eos_token
    return prompt







for i in range(5):
    sample = data_test[i]
    story_id = process_id(sample['id'])
    prompt = my_generate_prompt(TG_pred[story_id], sample['EK'], sample['Q'], sample['CoT'], sample['A'], Q_type=sample['Q-Type'], mode='test', eos_token="")

    print(prompt)
    print('===============================')






model_name = "meta-llama/Llama-2-13b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)


model = AutoModelForCausalLM.from_pretrained(model_name,
                                            load_in_8bit=True,
                                            device_map="auto"
                                            )









def one_batch(tokenizer, input_prompts, samples, file_paths):
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




tokenizer.pad_token_id = 0
tokenizer.padding_side = 'left'

peft_model_id = f"../model_weights/{dataset_name}_TGSR/final"
peft_model = PeftModel.from_pretrained(model, peft_model_id, torch_dtype=torch.float16, offload_folder="lora_results/lora_7/temp")

folder_path = f'../results/{dataset_name}_TGSR_ppl'
if not os.path.exists(folder_path):
    os.mkdir(folder_path)


folder_path_past_res = f'../results/{dataset_name}_TGSR'
if not os.path.exists(folder_path_past_res):
    print('Error! Please first generate the CoT results with the command "python Inference_in_context_learning.py".')
    sys.exit()



batchsize = 8
input_prompts = []
file_paths = []
samples = []
for i in range(len(data_test)):
    file_path = folder_path + f'/{str(i)}.json'
    if os.path.exists(file_path) and (not f_rewrite):
        continue

    sample = data_test[i]
    story_id = process_id(sample['id'])
    cur_prompt = my_generate_prompt(TG_pred[story_id], sample['EK'], sample['Q'], None, None, Q_type=sample['Q-Type'], mode='test', eos_token="")


    file_path_past_res = f'{folder_path_past_res}/{str(i)}.json'
    if not os.path.exists(file_path_past_res):
        continue

    with open(file_path_past_res) as json_file:
        past_res = json.load(json_file)

    cur_prompt += process_CoT(past_res['prediction'])



    input_prompts.append(cur_prompt)
    samples.append(sample)
    file_paths.append(file_path)

    if len(input_prompts) >= batchsize:
        one_batch(tokenizer, input_prompts, samples, file_paths)
        input_prompts = []
        file_paths = []
        samples = []


if len(input_prompts) > 0:
    one_batch(tokenizer, input_prompts, samples, file_paths)