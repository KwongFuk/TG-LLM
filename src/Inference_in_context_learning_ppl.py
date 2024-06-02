import sys
import json
import os
import copy

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset



os.environ["WANDB_DISABLED"] = "true"






dataset_selection = 0 # 0: TGQA, 1: TimeQA, 2: TimeQA, 3: TempReason, 4: TempReason
model_selection = 0 # 0: Llama-2-7b-hf, 1: Llama-2-13b-hf, 2: Llama-2-70b-hf
f_using_CoT = 0 # whether use CoT
f_ICL = 1  # whether use in-context learning during test
f_rewrite = 1 # whether rewrite existing test results
f_shorten_story = 1 # whether shorten the story


dataset_name = ['TGQA', 'TimeQA', 'TimeQA', 'TempReason', 'TempReason'][dataset_selection]
filename = ['TGSR_test.json', 'TGSR_easy_test.json', 'TGSR_hard_test.json', 'TGSR_l2_test.json', 'TGSR_l3_test.json'][dataset_selection]
Q_type = [None, None, None, 'l2', 'l3'][dataset_selection]
model_name = ['Llama-2-7b-hf', 'Llama-2-13b-hf', 'Llama-2-70b-hf'][model_selection]



def read_data(dataset_name, filename):
    '''
    Read the data from the json file and convert it into a dataset
    '''
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
    '''
    Generate the prompt for the model
    
    Args:
    story: str, the story
    Q: str, the question
    C: list of str, the candidates
    Q_type: str, the type of the question

    Returns:
    prompt: str, the prompt
    '''
    if f_ICL: # use in-context learning
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

    if f_shorten_story: # shorten the story
        story = ' '.join(story.split(' ')[:2000]) # only simply keep the first 2000 words

    if '(' not in C[0] and ')' not in C[0]:
        C = ['( ' + cand + ' )' for cand in C]
    Q += ' Choose from ' + ', '.join(C) + '.'

    if f_ICL:
        prompt = f"Example:\n\n{prompt_examples}\n\n\n\nTest:\n\nStory: {story}\n\nQuestion: {Q}"
    else:
        prompt = f"Story: {story}\n\nQuestion: {Q}"


    if f_using_CoT: # use CoT
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
    '''
    Given a batch of input prompts and candidates, calculate the perplexity of the candidates and choose the best one. Then save the results to the corresponding files.
    Todo: The two-round loops can be optimized to accelerate the process.
    
    Args:
    input_prompts: the input prompts, list
    samples: the samples, list
    file_paths: the file paths to save the results, list

    Returns:
    None 
    '''
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
    '''
    Remove the final answer from the CoT since we need to calculate the perplexity of the candidates.
    '''
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


batch_size = 8
input_prompts = []
file_paths = []
samples = []
for i in range(len(data_test)):
    # collect the prompts as a batch
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

    if len(input_prompts) >= batch_size:
        one_batch(input_prompts, samples, file_paths)
        input_prompts = []
        file_paths = []
        samples = []


if len(input_prompts) > 0:
    one_batch(input_prompts, samples, file_paths)