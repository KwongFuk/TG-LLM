import sys
import json
import random
import os
import copy



import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import Dataset, load_dataset, concatenate_datasets
from utlis import *
from tqdm import tqdm
from Models import *
from prompt_generation import *

os.environ["WANDB_DISABLED"] = "true"




######### Config #########

dataset_selection = 0  # 0: TGQA, 1: TimeQA_easy, 2: TimeQA_hard, 3: TempReason_l2, 4: TempReason_l3
f_train = True  # whether train the model
f_test = True  # whether test the model
f_CoT_bs = True  # whether use CoT bootstrapping
f_data_aug = True  # whether use data augmentation
f_ICL = True  # whether use in-context learning during test
f_rewrite = True  # whether rewrite existing test results
f_print_example_prompt = True  # whether to print the example prompt for the model
f_unit_test = False  # whether to run the unit test (only for debugging)

###########################


dataset_name = ['TGQA', 'TimeQA', 'TimeQA', 'TempReason', 'TempReason'][dataset_selection]
split_name = ['', '_easy', '_hard', '_l2', '_l3'][dataset_selection]
prefix = ['', 'easy_', 'hard_', 'l2_', 'l3_'][dataset_selection]








def read_data(dataset_name, prefix, split, f_CoT_bs=0, f_data_aug=0):
    '''
    Read the data from the given file.

    args:
        dataset_name: string, dataset name
        filename: string, file name
        f_CoT_bs: bool, whether to use CoT bootstrapping
        f_data_aug: bool, whether to use data augmentation

    return:
        dataset: Dataset, the dataset
    '''
    file_path = f'../results/{dataset_name}_TGR_CoT_bs/{prefix + split}.json'
    if (not f_CoT_bs) or (not os.path.exists(file_path)):
        dataset = load_dataset("sxiong/TGQA", f'{dataset_name}_TGR')
        dataset = dataset[prefix + split]
    else:
        with open(file_path) as json_file:
            data = json.load(json_file)

        # Convert list of dictionaries to the desired format
        data_dict = {'story': [item["story"] for item in data],
                     'TG': [item["TG"] for item in data],
                     'question': [item["question"] for item in data], 
                     'answer': [item["answer"] for item in data],
                     'external knowledge': [item["external knowledge"] for item in data],
                     'CoT': [CoT_sampling(item["CoT"], item['CoT_sample_prob']) for item in data],
                     'candidates': [item["candidates"] for item in data],
                     'id': [item['id'] for item in data],
                     'Q-Type': [item['Q-Type'] for item in data]}

        # Convert your data into a dataset
        dataset = Dataset.from_dict(data_dict)



    if f_data_aug and dataset_name in ['TGQA']:
        rel_entity_dict = collect_entity(dataset_name)
        random_entity_names = collect_entity_v2(dataset_name, rel_entity_dict)

        global_ent_mapping = {}   # we use a global mapping to ensure the consistency of entities and avoid confusion
        global_names_cnt = {}
        global_time_offset = random.randint(-20, 5)

        extra_data = [rel_entity_dict, global_ent_mapping, global_names_cnt, random_entity_names, global_time_offset]

        data_aug_dict = {'story': [], 'TG': [], 'external knowledge': [], 'question': [], 'CoT': [], 'candidates': [], 'answer': [], 'id': [], 'Q-Type': []}
        for sample in dataset:
            TG, EK, Q, CoT, C, A = data_augmentation(dataset_name, sample['TG'], sample['external knowledge'], sample['question'], sample['CoT'], 
                                                    sample['candidates'], sample['answer'], 
                                                    flag_rm_irr_edges=True, flag_change_relations=True, 
                                                    flag_change_entities=True, flag_change_times=True, extra_data=extra_data)
            data_aug_dict['story'].append(sample['story'])
            data_aug_dict['TG'].append(TG)
            data_aug_dict['external knowledge'].append(EK)
            data_aug_dict['question'].append(Q)
            data_aug_dict['CoT'].append(CoT)
            data_aug_dict['candidates'].append(C)
            data_aug_dict['answer'].append(A)
            data_aug_dict['id'].append(sample['id'])
            data_aug_dict['Q-Type'].append(sample['Q-Type'])

        dataset_aug = Dataset.from_dict(data_aug_dict)
        
        dataset = concatenate_datasets([dataset, dataset_aug])


    return dataset





data_train = read_data(dataset_name, prefix, 'train', f_CoT_bs, f_data_aug)
data_val = read_data(dataset_name, prefix, 'val', f_CoT_bs, f_data_aug)
data_test = read_data(dataset_name, prefix, 'test')


if f_unit_test:
    data_train = create_subset(data_train, 10)
    data_val = create_subset(data_val, 10)
    data_test = create_subset(data_test, 10)


print(data_train)
print(data_val)
print(data_test)




if f_test:
    # use estimated temporal graph for test
    TG_pred = obtain_TG_pred(dataset_name)




if f_print_example_prompt:
    if f_train:
        for i in range(5):
            sample = data_train[i]
            prompt = my_generate_prompt_TG_Reasoning(dataset_name, split_name, sample['TG'], sample['external knowledge'], sample['question'], sample['CoT'], sample['answer'], f_ICL, mode='train', eos_token="</s>")
            print(prompt)
            print('===============================')

    if f_test:
        for i in range(5):
            sample = data_test[i]
            story_id = process_id(dataset_name, sample['id'])
            if story_id in TG_pred:
                prompt = my_generate_prompt_TG_Reasoning(dataset_name, split_name, TG_pred[story_id], sample['external knowledge'], sample['question'], sample['CoT'], sample['answer'], f_ICL, Q_type=sample['Q-Type'], mode='test')
                print(prompt)
                print('===============================')






model_name = "meta-llama/Llama-2-13b-hf"  # you can change this to other models
tokenizer = AutoTokenizer.from_pretrained(model_name)


model = AutoModelForCausalLM.from_pretrained(model_name,
                                            load_in_8bit=True,
                                            device_map="auto"
                                            )







if f_train:
    def formatting_func(sample):
        '''
        Given the sample, generate the prompt for the model.
        '''
        output = []
        for g, e, q, cot, a in zip(sample['TG'], sample['external knowledge'], sample['question'], sample['CoT'], sample['answer']):
            op = my_generate_prompt_TG_Reasoning(dataset_name, split_name, g, e, q, cot, a, f_ICL, mode='train', eos_token="</s>")
            output.append(op)

        return output

    output_dir = f"../model_weights/{dataset_name}_TGR{split_name}"
    SFT_with_LoRA(model, tokenizer, output_dir, f_unit_test, formatting_func, data_train, data_val, 12, 2048)


if f_test:
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'left'

    peft_model_id = f"../model_weights/{dataset_name}_TGR{split_name}/final"
    peft_model = PeftModel.from_pretrained(model, peft_model_id, torch_dtype=torch.float16, offload_folder="lora_results/lora_7/temp")
    peft_model.eval()

    folder_path = f'../results/{dataset_name}_TGR{split_name}'
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    batch_size = 8

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

        input_prompts.append(cur_prompt)
        samples.append(sample)
        file_paths.append(file_path)

        # collect the prompts as a batch
        if len(input_prompts) >= batch_size:
            run_one_batch_generation(peft_model, tokenizer, input_prompts, samples, file_paths)
            input_prompts = []
            file_paths = []
            samples = []


    if len(input_prompts) > 0:
        run_one_batch_generation(peft_model, tokenizer, input_prompts, samples, file_paths)