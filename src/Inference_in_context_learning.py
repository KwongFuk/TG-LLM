import sys
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
import openai
import time


os.environ["WANDB_DISABLED"] = "true"






dataset_selection = 0 # 0: TGQA, 1: TimeQA_easy, 2: TimeQA_hard, 3: TempReason_l2, 4: TempReason_l3
model_selection = 2 # 0: gpt-3.5-turbo, 1: gpt-4-1106-preview, 2: Llama-2-7b-hf, 3: Llama-2-13b-hf, 4: Llama-2-70b-hf
f_using_CoT = 0 # whether use CoT
f_ICL = 1  # whether use in-context learning during test
f_rewrite = 1 # whether rewrite existing test results
f_shorten_story = 1 # whether shorten the story


dataset_name = ['TGQA', 'TimeQA_easy', 'TimeQA_hard', 'TempReason_l2', 'TempReason_l3'][dataset_selection]
filename = ['TGSR_test.json', 'TGSR_easy_test.json', 'TGSR_hard_test.json', 'TGSR_l2_test.json', 'TGSR_l3_test.json'][dataset_selection]
Q_type = [None, None, None, 'l2', 'l3'][dataset_selection]
model_name = ['gpt-3.5-turbo', 'gpt-4-1106-preview', 'Llama-2-7b-hf', 'Llama-2-13b-hf', 'Llama-2-70b-hf'][model_selection]



def read_data(dataset_name, filename):
    '''
    Read the data from the json file and convert it into a dataset
    '''
    file_path = f'../dataset/{dataset_name.split('_')[0]}/{filename}'
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





def my_gpt_completion(openai_model, messages, timeout, max_new_tokens=128, wait_time=0):
    '''
    Use the GPT model to generate the completion
    We can add exception handling here to avoid interruption

    args:
    openai_model: the model name
    messages: the messages in the conversation
    timeout: the timeout for the request
    max_new_tokens: the maximum new tokens for the completion
    wait_time: the wait time between requests

    return:
    response: the generated response, str
    '''
    openai.api_key =   # add your own api_key here (you can look at the openai website to get one)
    completion = openai.ChatCompletion.create(model=openai_model,
                                              messages=messages,
                                              request_timeout = timeout,
                                              temperature=0.7,
                                              max_tokens=max_new_tokens
                                            )
    response = completion['choices'][0]["message"]["content"]
    time.sleep(wait_time) # wait for a while to avoid the rate limit

    return response




def my_generate_prompt(story, Q, C, Q_type=None):
    '''
    Gnerate the prompt for the model

    args:
    story: the story, str
    Q: the question, str
    C: the candidates, list
    Q_type: the question type, str

    return:
    prompt: the generated prompt, str
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






if 'Llama' in model_name:
    model_name_cmp = f'meta-llama/{model_name}'
    tokenizer = AutoTokenizer.from_pretrained(model_name_cmp)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(model_name_cmp,
                                                load_in_8bit=True,
                                                device_map="auto"
                                                )





def one_batch(input_prompts, samples, file_paths, max_new_tokens=512):
    '''
    Generate the completion for one batch of input prompts

    args:
    input_prompts: the input prompts, list
    samples: the samples, list
    file_paths: the file paths to save the results, list
    max_new_tokens: the maximum new tokens for the completion

    return:
    None
    '''
    if 'Llama' in model_name:
        input_tokens = tokenizer(input_prompts, padding='longest', return_tensors="pt")["input_ids"].to("cuda")

        with torch.cuda.amp.autocast():
            generation_output = model.generate(
                input_ids=input_tokens,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_k=10,
                top_p=0.9,
                temperature=0.3,
                repetition_penalty=1.15,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
              )


        for j in range(len(input_prompts)):
            op = tokenizer.decode(generation_output[j], skip_special_tokens=True)
            op = op[len(input_prompts[j]):]
            cur_sample = samples[j]
            cur_sample.update({'prediction': op})

            with open(file_paths[j], 'w') as json_file:
                json.dump(cur_sample, json_file)


    if 'gpt' in model_name:
        for j in range(len(input_prompts)):
            messages = []
            messages.append({"role": "user", "content": input_prompts[j]})
            op = my_gpt_completion(model_name, messages, 600, max_new_tokens=max_new_tokens, wait_time=0.5)

            cur_sample = samples[j]
            cur_sample.update({'prediction': op})

            with open(file_paths[j], 'w') as json_file:
                json.dump(cur_sample, json_file)


    return







folder_path = f'../results/{dataset_name}_ICL_{model_name}'
if not os.path.exists(folder_path):
    os.mkdir(folder_path)

batch_size = 8
input_prompts = []
file_paths = []
samples = []
for i in range(len(data_test)):
    # collect the prompts as a batch
    file_path = folder_path + f'/{str(i)}.json'
    if os.path.exists(file_path) and (not f_rewrite):
        continue

    sample = data_test[i]
    cur_prompt = my_generate_prompt(sample['story'], sample['Q'], sample['C'], Q_type=sample['Q-Type'])

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