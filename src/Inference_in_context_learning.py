import sys
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import openai
import time
from utlis import *
from tqdm import tqdm

os.environ["WANDB_DISABLED"] = "true"





######### Config #########

dataset_selection = 0  # 0: TGQA, 1: TimeQA_easy, 2: TimeQA_hard, 3: TempReason_l2, 4: TempReason_l3
model_selection = 3  # 0: gpt-3.5-turbo, 1: gpt-4-1106-preview, 2: Llama-2-7b-hf, 3: Llama-2-13b-hf, 4: Llama-2-70b-hf
f_using_CoT = True  # whether use CoT
f_ICL = True   # whether use in-context learning during test
f_rewrite = True  # whether rewrite existing test results
f_shorten_story = True  # whether shorten the story
f_print_example_prompt = True  # whether to print the example prompt for the model
f_unit_test = False  # whether to run the unit test (only for debugging)

openai.api_key = None  # add your own api_key here (you can look at the openai website to get one)

###########################


dataset_name = ['TGQA', 'TimeQA', 'TimeQA', 'TempReason', 'TempReason'][dataset_selection]
split_name = ['', '_easy', '_hard', '_l2', '_l3'][dataset_selection]
prefix = ['', 'easy_', 'hard_', 'l2_', 'l3_'][dataset_selection]
model_name = ['gpt-3.5-turbo', 'gpt-4-1106-preview', 'Llama-2-7b-hf', 'Llama-2-13b-hf', 'Llama-2-70b-hf'][model_selection]
learning_setting = 'SP' if not f_using_CoT else 'CoT'


dataset = load_dataset("sxiong/TGQA", f'{dataset_name}_TGR')
data_test = dataset[prefix + 'test']

if f_unit_test:
    data_test = create_subset(data_test, 10)

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

        if Q_type is None:
            file_path = f'../materials/{dataset_name}/prompt_examples_ICL_{learning_setting}{split_name}.txt'
        else:
            file_path = f'../materials/{dataset_name}/prompt_examples_ICL_{learning_setting}{split_name}_{Q_type}.txt'

        with open(file_path) as txt_file:
            prompt_examples = txt_file.read()

    if f_shorten_story: # shorten the story
        story = shorten_story(story)

    C = add_brackets(C)
    Q += ' Choose from ' + ', '.join(C) + '.'
    
    prompt = f"Example:\n\n{prompt_examples}\n\n\n\nTest:\n\nStory: {story}\n\nQuestion: {Q}" if f_ICL else f"Story: {story}\n\nQuestion: {Q}"
    prompt += "\n\nAnswer: Let's think step by step.\n\n" if f_using_CoT else "\n\nAnswer: "

    return prompt




if f_print_example_prompt:
    for i in range(5):
        sample = data_test[i]
        prompt = my_generate_prompt(sample['story'], sample['question'], sample['candidates'], Q_type=sample['Q-Type'])

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
    model.eval()





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







folder_path = f'../results/{dataset_name}_ICL_{learning_setting}{split_name}_{model_name}' if f_ICL else f'../results/{dataset_name}{split_name}_{model_name}'
if not os.path.exists(folder_path):
    os.mkdir(folder_path)

batch_size = 4
input_prompts = []
file_paths = []
samples = []
for i in tqdm(range(len(data_test))):
    file_path = f'{folder_path}/{str(i)}.json'
    if os.path.exists(file_path) and (not f_rewrite):
        continue

    sample = data_test[i]
    cur_prompt = my_generate_prompt(sample['story'], sample['question'], sample['candidates'], Q_type=sample['Q-Type'])

    input_prompts.append(cur_prompt)
    samples.append(sample)
    file_paths.append(file_path)

    # collect the prompts as a batch
    if len(input_prompts) >= batch_size:
        one_batch(input_prompts, samples, file_paths)
        input_prompts = []
        file_paths = []
        samples = []


if len(input_prompts) > 0:
    one_batch(input_prompts, samples, file_paths)