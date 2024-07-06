import sys
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from utlis import *
from tqdm import tqdm
from Models import *
from prompt_generation import *
import argparse

os.environ["WANDB_DISABLED"] = "true"


parser = argparse.ArgumentParser()


parser.add_argument('--dataset', type=str)
parser.add_argument('--ICL', action='store_true')
parser.add_argument('--rewrite', action='store_true')
parser.add_argument('--print_prompt', action='store_true')
parser.add_argument('--unit_test', action='store_true')


args = parser.parse_args()



######### Config #########

dataset_selection = ['TGQA', 'TimeQA_easy', 'TimeQA_hard', 'TempReason_l2', 'TempReason_l3'].index(args.dataset)
f_ICL = args.ICL   # whether use in-context learning during test
f_rewrite = args.rewrite   # whether rewrite existing test results
f_print_example_prompt = args.print_prompt   # whether to print the example prompt for the model
f_unit_test = args.unit_test   # whether to run the unit test (only for debugging)

###########################


dataset_name = ['TGQA', 'TimeQA', 'TimeQA', 'TempReason', 'TempReason'][dataset_selection]
split_name = ['', '_easy', '_hard', '_l2', '_l3'][dataset_selection]
prefix = ['', 'easy_', 'hard_', 'l2_', 'l3_'][dataset_selection]




dataset = load_dataset("sxiong/TGQA", f'{dataset_name}_TGR')
data_test = dataset[f'{prefix}test']



if f_unit_test:
    data_test = create_subset(data_test, 100)


print(data_test)


# use estimated temporal graph for test
TG_pred = obtain_TG_pred(dataset_name)


if f_print_example_prompt:
    for i in range(5):
        sample = data_test[i]
        story_id = process_id(dataset_name, sample['id'])
        if story_id in TG_pred:
            prompt = my_generate_prompt_TG_Reasoning(dataset_name, split_name, TG_pred[story_id], sample['external knowledge'], sample['question'], None, None, f_ICL, Q_type=sample['Q-Type'], mode='test')
            print(prompt)
            print('===============================')




model_name = "meta-llama/Llama-2-13b-hf" # you can change to other models
tokenizer = AutoTokenizer.from_pretrained(model_name)


model = AutoModelForCausalLM.from_pretrained(model_name,
                                            load_in_8bit=True,
                                            device_map="auto"
                                            )


tokenizer.pad_token_id = 0
tokenizer.padding_side = 'right'

peft_model_id = f"../model_weights/{dataset_name}_TGR{split_name}/final"
peft_model = PeftModel.from_pretrained(model, peft_model_id, torch_dtype=torch.float16, offload_folder="lora_results/lora_7/temp")
peft_model.eval()

folder_path = f'../results/{dataset_name}_TGR{split_name}_ppl'
if not os.path.exists(folder_path):
    os.mkdir(folder_path)


folder_path_past_res = f'../results/{dataset_name}_TGR{split_name}'
if not os.path.exists(folder_path_past_res):
    print('Error! Please first generate the CoT results with the command "python Inference_in_context_learning.py".')
    sys.exit()



batch_size = 4
input_prompts = []
file_paths = []
samples = []
for i in tqdm(range(len(data_test))):
    file_path = folder_path + f'/{str(i)}.json'
    if os.path.exists(file_path) and (not f_rewrite):
        continue

    sample = data_test[i]
    story_id = process_id(dataset_name, sample['id'])
    if story_id not in TG_pred:
        continue
    cur_prompt = my_generate_prompt_TG_Reasoning(dataset_name, split_name, TG_pred[story_id], sample['external knowledge'], sample['question'], None, None, f_ICL, Q_type=sample['Q-Type'], mode='test')

    file_path_past_res = f'{folder_path_past_res}/{str(i)}.json'
    if not os.path.exists(file_path_past_res):
        continue

    with open(file_path_past_res) as json_file:
        past_res = json.load(json_file)

    CoT, _ = parse_TGR_pred(past_res['prediction'])
    if CoT is None:
        continue

    cur_prompt += f'\n{{\n"Thought": ""{json.dumps(CoT)}"",\n"Answer":'

    input_prompts.append(cur_prompt)
    samples.append(sample)
    file_paths.append(file_path)

    # collect the prompts as a batch
    if len(input_prompts) >= batch_size:
        run_one_batch_ppl(peft_model, tokenizer, input_prompts, samples, file_paths)
        input_prompts = []
        file_paths = []
        samples = []


if len(input_prompts) > 0:
    run_one_batch_ppl(peft_model, tokenizer, input_prompts, samples, file_paths)