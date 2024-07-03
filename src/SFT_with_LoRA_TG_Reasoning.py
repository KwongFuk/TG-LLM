import sys
import json
import random
import os
import copy



import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import (
        get_peft_model, 
        prepare_model_for_kbit_training, 
        LoraConfig
    )
from trl import SFTTrainer
from peft import PeftModel
from datasets import Dataset, load_dataset, concatenate_datasets
from utlis import *


os.environ["WANDB_DISABLED"] = "true"




######### Config #########

dataset_selection = 0  # 0: TGQA, 1: TimeQA_easy, 2: TimeQA_hard, 3: TempReason_l2, 4: TempReason_l3
f_train = 1  # whether train the model
f_test = 1  # whether test the model
f_CoT_bs = 1  # whether use CoT bootstrapping
f_data_aug = 1  # whether use data augmentation
f_ICL = 1  # whether use in-context learning during test
f_rewrite = 1  # whether rewrite existing test results
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
    TG_pred = {}
    path_TG_pred = f'../results/{dataset_name}_story_TG_trans/'
    for filename in os.listdir(path_TG_pred):
        file_path = os.path.join(path_TG_pred, filename)
        with open(file_path) as json_file:
            data = json.load(json_file)
        TG_pred[data['id']] = data['prediction']




def my_generate_prompt(TG, EK, Q, CoT, A, Q_type=None, mode=None, eos_token=""):
    '''
    Generate the prompt for the model.

    args:
        TG: list of strings or string, temporal graph
        EK: list of strings or string, exteral knowledge
        Q: string, question
        CoT: list of strings, chain of thought
        A: string, answer
        Q_type: string, question type
        mode: string, mode
        eos_token: string, eos token

    return:
        prompt: string, the prompt
    '''
    if isinstance(TG, list):
        TG = '\n'.join(TG)

    if f_ICL and mode == 'test':
        if dataset_name == 'TGQA':
            Q_type = f'Q{Q_type}'

        if Q_type is None:
            file_path = f'../materials/{dataset_name}/prompt_examples_TGR{split_name}.txt'
        else:
            file_path = f'../materials/{dataset_name}/prompt_examples_TGR_{Q_type}.txt'
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






if f_print_example_prompt:
    for i in range(5):
        if f_train:
            sample = data_train[i]
            prompt = my_generate_prompt(sample['TG'], sample['external knowledge'], sample['question'], sample['CoT'], sample['answer'], mode='train', eos_token="</s>")

        if f_test:
            sample = data_test[i]
            story_id = process_id(dataset_name, sample['id'])
            prompt = my_generate_prompt(TG_pred[story_id], sample['external knowledge'], sample['question'], sample['CoT'], sample['answer'], Q_type=sample['Q-Type'], mode='test')

        print(prompt)
        print('===============================')






model_name = "meta-llama/Llama-2-13b-hf"  # you can change this to other models
tokenizer = AutoTokenizer.from_pretrained(model_name)


model = AutoModelForCausalLM.from_pretrained(model_name,
                                            load_in_8bit=True,
                                            device_map="auto"
                                            )







if f_train:
    # lora config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # this should be set for finutning and batched inference
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    model.resize_token_embeddings(len(tokenizer))

    # Loading in 8 bit ..."
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    output_dir = f"../model_weights/{dataset_name}_TGR{split_name}"
    per_device_train_batch_size = 12
    gradient_accumulation_steps = 4
    per_device_eval_batch_size = 12
    eval_accumulation_steps = 4
    optim = "paged_adamw_32bit"
    save_steps = 10
    logging_steps = 10
    learning_rate = 5e-4
    max_grad_norm = 0.3
    max_steps = 5 if f_unit_test else 50
    warmup_ratio = 0.03
    evaluation_strategy="steps"
    lr_scheduler_type = "constant"

    training_args = transformers.TrainingArguments(
                output_dir=output_dir,
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                optim=optim,
                evaluation_strategy=evaluation_strategy,
                save_steps=save_steps,
                learning_rate=learning_rate,
                logging_steps=logging_steps,
                max_grad_norm=max_grad_norm,
                max_steps=max_steps,
                warmup_ratio=warmup_ratio,
                group_by_length=True,
                lr_scheduler_type=lr_scheduler_type,
                ddp_find_unused_parameters=False,
                eval_accumulation_steps=eval_accumulation_steps,
                per_device_eval_batch_size=per_device_eval_batch_size
            )


    def formatting_func(sample):
        '''
        Given the sample, generate the prompt for the model.
        '''
        output = []
        for g, e, q, cot, a in zip(sample['TG'], sample['external knowledge'], sample['question'], sample['CoT'], sample['answer']):
            op = my_generate_prompt(g, e, q, cot, a, mode='train', eos_token="</s>")
            output.append(op)

        return output


    trainer = SFTTrainer(
        model=model,
        train_dataset=data_train,
        eval_dataset=data_val,
        peft_config=lora_config,
        formatting_func=formatting_func,
        max_seq_length=2048,
        tokenizer=tokenizer,
        args=training_args
    )

    # We will also pre-process the model by upcasting the layer norms in float 32 for more stable training
    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)

    trainer.train()
    trainer.save_model(f"{output_dir}/final")



if f_test:
    def one_batch(tokenizer, input_prompts, samples, file_paths, max_new_tokens=512):
        '''
        Generate the predictions for one batch of samples and save the results.

        args:
            tokenizer: tokenizer
            input_prompts: list of strings, input prompts
            samples: list of dictionaries, samples
            file_paths: list of strings, file paths
            max_new_tokens: int, maximum number of new tokens

        return:
            None
        '''
        input_tokens = tokenizer(input_prompts, padding='longest', return_tensors="pt")["input_ids"].to("cuda")

        with torch.cuda.amp.autocast():
            generation_output = peft_model.generate(
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

        return



    tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'left'

    peft_model_id = f"../model_weights/{dataset_name}_TGR{split_name}/final"
    peft_model = PeftModel.from_pretrained(model, peft_model_id, torch_dtype=torch.float16, offload_folder="lora_results/lora_7/temp")

    folder_path = f'../results/{dataset_name}_TGR{split_name}'
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    batch_size = 8

    input_prompts = []
    file_paths = []
    samples = []
    for i in range(len(data_test)):
        file_path = folder_path + f'/{str(i)}.json'
        if os.path.exists(file_path) and (not f_rewrite):
            continue

        sample = data_test[i]
        story_id = process_id(dataset_name, sample['id'])
        cur_prompt = my_generate_prompt(TG_pred[story_id], sample['external knowledge'], sample['question'], None, None, Q_type=sample['Q-Type'], mode='test')

        input_prompts.append(cur_prompt)
        samples.append(sample)
        file_paths.append(file_path)

        # collect the prompts as a batch
        if len(input_prompts) >= batch_size:
            one_batch(tokenizer, input_prompts, samples, file_paths)
            input_prompts = []
            file_paths = []
            samples = []


    if len(input_prompts) > 0:
        one_batch(tokenizer, input_prompts, samples, file_paths)